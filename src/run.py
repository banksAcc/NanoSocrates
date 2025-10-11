"""Entry point per addestrare o forzare l'overfit del modello NanoSocrates."""

from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tokenizers import Tokenizer

from src.data.builders import build_and_cache_datasets
from src.model.transformer import TinySeq2Seq
from src.training.dataloaders import create_multitask_dataloader
from src.training.loop import TrainingLoop
from src.training.scheduler import create_scheduler
from src.utils.config import (
    add_common_overrides,
    apply_overrides,
    apply_toy_paths,
    load_yaml,
)
from src.utils.wandb_utils import maybe_init_wandb

LOGGER = logging.getLogger(__name__)


def _get_pad_id(tokenizer: Tokenizer) -> int:
    pad = tokenizer.token_to_id("<pad>")
    if pad is None:
        raise ValueError("Tokenizer privo di <pad>: rigenera il BPE includendo <pad>.")
    return int(pad)


def set_seed(seed: int) -> None:
    """Assicura riproducibilità impostando tutti i seed noti."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - dipende da hw
        torch.cuda.manual_seed_all(seed)
    LOGGER.info("Seed fissato a %d", seed)


def _build_model(cfg: Dict[str, Any], tokenizer: Tokenizer) -> TinySeq2Seq:
    """Costruisce il Transformer rispettando le scelte presenti nel config."""

    pad_id = _get_pad_id(tokenizer)
    model = TinySeq2Seq(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=int(cfg.get("d_model", 384)),
        nhead=int(cfg.get("nhead", 6)),
        num_encoder_layers=int(cfg.get("enc_layers", 3)),
        num_decoder_layers=int(cfg.get("dec_layers", 3)),
        dim_feedforward=int(cfg.get("ff_dim", 1536)),
        dropout=float(cfg.get("dropout", 0.1)),
        pad_id=pad_id,
        tie_embeddings=True,
        use_mla=bool(cfg.get("use_mla", False)),
        use_rope=bool(cfg.get("use_rope", False)),
        interleave_ratio=float(cfg.get("interleave_ratio", 0.0)),
        max_position_embeddings=int(cfg.get("max_len", 256)),
        compute_span_metrics=bool(cfg.get("compute_span_metrics", False)),
        architecture=str(cfg.get("architecture", "vanilla")),
        relative_attention_num_buckets=int(cfg.get("relative_attention_num_buckets", 32)),
        relative_attention_max_distance=int(cfg.get("relative_attention_max_distance", 128)),
        layer_norm_epsilon=float(cfg.get("layer_norm_epsilon", 1e-6)),
    )
    LOGGER.info("Modello creato con %s parametri", f"{sum(p.numel() for p in model.parameters()):,}")
    return model


def _build_optimizer(cfg: Dict[str, Any], model: TinySeq2Seq) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )


def _build_scheduler(
    cfg: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    name = cfg.get("scheduler")
    if not name or total_steps <= 0:
        return None
    return create_scheduler(
        str(name),
        optimizer,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),
        total_steps=total_steps,
        min_lr_ratio=float(cfg.get("min_lr_ratio", 0.0)),
    )


def _prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml(args.cfg)
    if getattr(args, "toy", False):
        cfg = apply_toy_paths(cfg)
        LOGGER.info("[toy] uso i dataset compatti in data/processed/toy")
    cfg = apply_overrides(cfg, args.override)
    return cfg


def run_training(cfg: Dict[str, Any], *, overfit: bool = False) -> None:
    """Esegue l'intero ciclo di training partendo da un config strutturato."""

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    tokenizer_path = cfg.get("tokenizer_file") or cfg.get("data", {}).get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("Specificare 'tokenizer_file' nel config di training.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    dataset_payload = build_and_cache_datasets(cfg, tokenizer)
    train_dataset = dataset_payload["train"]
    val_dataset = dataset_payload["validation"]
    ratios = dataset_payload.get("ratios") or train_dataset.task_fractions()

    if overfit:
        limit = int(cfg.get("overfit_samples", cfg.get("batch_size", 8)))
        train_dataset = train_dataset.select_first(limit)
        val_dataset = train_dataset
        ratios = train_dataset.task_fractions()
        LOGGER.info("Modalità overfit attiva: uso i primi %d esempi", len(train_dataset))

    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))

    train_loader = create_multitask_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        ratios=ratios,
        num_workers=num_workers,
        shuffle=not overfit,
    )
    val_loader = create_multitask_dataloader(
        val_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        ratios=ratios,
        num_workers=num_workers,
        shuffle=False,
    )

    steps_per_epoch = max(1, math.ceil(len(train_dataset) / batch_size))
    total_steps = steps_per_epoch * int(cfg.get("num_epochs", 1))

    model = _build_model(cfg, tokenizer)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer, total_steps)

    save_dir = Path(cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / ("overfit.pt" if overfit else "best.pt")

    run, wandb_module = maybe_init_wandb(cfg)

    requested_device = str(cfg.get("device", "cuda") or "cuda").lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA non disponibile: eseguo il training su CPU")
        device = "cpu"
    else:
        device = requested_device
    use_amp = bool(cfg.get("use_amp", True)) and device == "cuda"

    loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_amp=use_amp,
        grad_accum_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        log_every_n_steps=int(cfg.get("log_every_n_steps", 50)),
        checkpoint_path=str(checkpoint_path),
        early_stopping_patience=int(cfg.get("early_stopping", {}).get("patience", 5)),
        early_stopping_metric=str(cfg.get("early_stopping", {}).get("metric", "loss")),
        early_stopping_mode=str(cfg.get("early_stopping", {}).get("mode", "min")),
        wandb_run=run,
    )

    num_epochs = int(cfg.get("num_epochs", 1))
    if num_epochs <= 0:
        LOGGER.info("num_epochs=0: pipeline configurata, nessun batch elaborato.")
    else:
        loop.run(num_epochs=num_epochs)

    if run is not None and wandb_module is not None:
        try:
            wandb_module.finish()
        except Exception as exc:  # pragma: no cover - dipende da env
            LOGGER.warning("Chiusura wandb fallita: %s", exc)

    LOGGER.info("Training completato. Checkpoint salvato in %s", checkpoint_path.resolve())


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _prepare_config(args)
    run_training(cfg, overfit=False)


def cmd_overfit(args: argparse.Namespace) -> None:
    cfg = _prepare_config(args)
    run_training(cfg, overfit=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline di training NanoSocrates")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Addestra il modello sul dataset indicato")
    add_common_overrides(p_train)

    p_overfit = sub.add_parser("overfit", help="Forza l'overfit di un singolo batch")
    add_common_overrides(p_overfit)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "overfit":
        cmd_overfit(args)
    else:  # pragma: no cover - guardia difensiva
        raise ValueError(f"Comando sconosciuto: {args.command}")


if __name__ == "__main__":
    main()
