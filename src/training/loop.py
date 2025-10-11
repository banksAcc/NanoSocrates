"""Main training and evaluation loop for the transformer model."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import amp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import wandb
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class TrainingLoop:
    """A class to encapsulate the training and validation loops.

    This class handles the complexities of model training, including:
    - Iterating over epochs and batches.
    - Gradient accumulation to simulate larger batch sizes.
    - Automatic Mixed Precision (AMP) for faster training on compatible GPUs.
    - Checkpointing the best model based on a validation metric.
    - Early stopping to prevent overfitting.
    - Logging metrics to Weights & Biases (wandb).
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: _LRScheduler | None = None,
        device: str | torch.device | None = None,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        log_every_n_steps: int = 100,
        checkpoint_path: str = "best_model.pt",
        early_stopping_patience: int = 5,
        early_stopping_metric: str = "loss",
        early_stopping_mode: Literal["min", "max"] = "min",
        wandb_run: "wandb.sdk.wandb_run.Run" | None = None,
    ):
        """Initializes the TrainingLoop.

        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            scheduler: Optional learning rate scheduler.
            device: The device to train on ('cuda', 'cpu'). If None, it will be
                auto-detected.
            use_amp: Whether to use Automatic Mixed Precision.
            grad_accum_steps: Number of steps to accumulate gradients over.
            log_every_n_steps: How often to log training metrics to wandb.
            checkpoint_path: Path to save the best model checkpoint.
            early_stopping_patience: Number of epochs to wait for improvement
                before stopping.
            early_stopping_metric: The validation metric to monitor for early
                stopping and checkpointing.
            early_stopping_mode: 'min' if a lower metric is better (e.g., loss),
                'max' if a higher metric is better (e.g., accuracy).
            wandb_run: An active wandb run object for logging.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._train_loader_len = len(train_loader)
        self.grad_accum_steps = grad_accum_steps
        self.log_every_n_steps = log_every_n_steps
        self.checkpoint_path = checkpoint_path
        self.wandb_run = wandb_run

        # Early stopping setup
        self.es_patience = early_stopping_patience
        self.es_metric = early_stopping_metric
        self.es_mode = early_stopping_mode
        self.es_counter = 0
        self.best_score = -float("inf") if self.es_mode == "max" else float("inf")

        # Auto-detect device if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # AMP setup
        self.use_amp = use_amp and self.device.type == "cuda"
        self.autocast_kwargs = {
            "device_type": self.device.type,
            "enabled": self.use_amp,
        }
        if self.use_amp:
            self.autocast_kwargs["dtype"] = torch.float16
            logger.info("Automatic Mixed Precision (AMP) enabled.")

        self.scaler = None
        if self.use_amp:
            try:
                self.scaler = amp.GradScaler(device_type=self.device.type, enabled=True)
            except TypeError:
                # Older PyTorch versions expect the legacy signature without the
                # device argument. Fall back gracefully so training still works.
                self.scaler = amp.GradScaler(enabled=True)
        self.model.to(self.device)
        
    def run(self, num_epochs: int) -> dict[str, Any]:
        """Starts and manages the training process for a given number of epochs.

        Args:
            num_epochs: The total number of epochs to train for.

        Returns:
            A dictionary containing the best score achieved and the epoch at which
            it occurred.
        """
        if self._train_loader_len == 0:
            raise ValueError(
                "Training DataLoader is empty. Check the dataset paths or preprocessing "
                "pipeline for the selected configuration."
            )

        logger.info("Starting training...")
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch()

            if self.scheduler:
                # Some schedulers use metrics (e.g., ReduceLROnPlateau)
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.es_metric])
                else:
                    self.scheduler.step()

            # Log metrics to wandb
            if self.wandb_run:
                metrics_to_log = {
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
                self.wandb_run.log(metrics_to_log)

            logger.info(f"Validation metrics: {val_metrics}")

            # Early stopping and checkpointing
            current_score = val_metrics[self.es_metric]
            if self._check_early_stopping(current_score):
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training finished. Best score: {self.best_score:.4f}")
        return {"best_score": self.best_score, "best_epoch": epoch - self.es_counter}

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Performs one full training pass over the training data.

        Returns:
            A dictionary of average training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        # Use a moving average for smoother loss reporting in tqdm
        smoothing_factor = 0.98
        smoothed_loss = 0.0
        is_first_batch = True

        pbar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )

        for i, batch in enumerate(pbar):
            batch = self._transfer_batch_to_device(batch)
            model_inputs = self._prepare_model_inputs(batch)
            step = (epoch - 1) * self._train_loader_len + i  
            with amp.autocast(**self.autocast_kwargs):
                outputs = self.model(**model_inputs)
                loss = outputs["loss"]
                if loss is None:
                    continue
                loss = loss / self.grad_accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Update progress bar with smoothed loss
            loss_item = loss.item() * self.grad_accum_steps
            total_loss += loss_item
            if is_first_batch:
                smoothed_loss = loss_item
                is_first_batch = False
            else:
                smoothed_loss = (smoothing_factor * smoothed_loss) + (1 - smoothing_factor) * loss_item
            
            pbar.set_postfix({"loss": f"{smoothed_loss:.4f}"})

            # Log to wandb periodically
            if self.wandb_run and step % self.log_every_n_steps == 0:
                log_data = {"train/step_loss": loss_item, "learning_rate": self.optimizer.param_groups[0]["lr"]}
                if "metrics" in outputs and outputs["metrics"] is not None:
                    for k, v in outputs["metrics"].items():
                        log_data[f"train/{k}"] = v
                self.wandb_run.log(log_data, step=step)

        avg_loss = total_loss / self._train_loader_len
        return {"loss": avg_loss}

    @torch.inference_mode()
    def _validate_epoch(self) -> dict[str, float]:
        """Performs one full validation pass.

        Returns:
            A dictionary of average validation metrics.
        """
        self.model.eval()
        metrics_agg = defaultdict(float)
        total_count = 0

        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            batch = self._transfer_batch_to_device(batch)
            model_inputs = self._prepare_model_inputs(batch)
            with amp.autocast(**self.autocast_kwargs):
                outputs = self.model(**model_inputs)

            if outputs["loss"] is not None:
                metrics_agg["loss"] += outputs["loss"].item() * len(batch["input_ids"])
            
            if "metrics" in outputs and outputs["metrics"] is not None:
                for k, v in outputs["metrics"].items():
                    metrics_agg[k] += v * len(batch["input_ids"])

            total_count += len(batch["input_ids"])

        if total_count == 0:
            logger.warning(
                "Validation DataLoader produced no batches; returning empty metrics."
            )
            return {}

        # Average the metrics over the entire dataset
        avg_metrics = {k: v / total_count for k, v in metrics_agg.items()}
        return avg_metrics

    def _check_early_stopping(self, current_score: float) -> bool:
        """Checks if early stopping criteria are met and saves the best model.

        Args:
            current_score: The validation score from the current epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        is_better = (current_score < self.best_score) if self.es_mode == "min" else (current_score > self.best_score)

        if is_better:
            self.best_score = current_score
            self.es_counter = 0
            logger.info(f"New best score: {self.best_score:.4f}. Saving model...")

            checkpoint: dict[str, object] = {"model": self.model.state_dict()}
            export_fn = getattr(self.model, "export_config", None)
            if callable(export_fn):
                try:
                    checkpoint["config"] = export_fn()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Unable to export model config for checkpoint: %s", exc)

            torch.save(checkpoint, self.checkpoint_path)
        else:
            self.es_counter += 1
            logger.info(f"No improvement. Early stopping counter: {self.es_counter}/{self.es_patience}")

        return self.es_counter >= self.es_patience

    def _transfer_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Moves a batch of data to the configured device."""
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @staticmethod
    def _prepare_model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Filters a collated batch to only the tensors consumed by the model."""

        allowed_keys = {
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "labels",
            "mask_positions",
            "mask_lengths",
        }
        return {k: v for k, v in batch.items() if k in allowed_keys}