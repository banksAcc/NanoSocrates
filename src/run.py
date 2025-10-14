"""Entry point per addestrare, valutare o forzare l'overfit del modello."""

from __future__ import annotations

import argparse
from enum import Enum
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tokenizers import Tokenizer

from src.decoding.base import decode_to_text
from src.eval.evaluate import evaluate_from_config, load_model_and_tokenizer
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
    resolve_checkpoint_reference,
)
from src.utils.wandb_utils import flatten_eval_metrics, maybe_init_wandb
from src.tokenizer.tokenizer_io import ensure_runtime_special_tokens

LOGGER = logging.getLogger(__name__)

TASK_MARKERS = {
    "text2rdf": "<Text2RDF>",
    "rdf2text": "<RDF2Text>",
    "rdfcomp2": "<CONTINUERDF>",
    "rdfcomp1": "<MASK>",
}


def _get_pad_id(tokenizer: Tokenizer) -> int:
    """Look up the pad token id and fail fast if the vocabulary is incomplete."""
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
    """Create the default AdamW optimiser using hyperparameters from *cfg*."""
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
    """Construct the learning-rate scheduler only when it is explicitly requested."""
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
    """Load the configuration file and apply CLI overrides/toy shortcuts."""
    cfg = load_yaml(args.cfg)
    if getattr(args, "toy", False):
        cfg = apply_toy_paths(cfg)
        LOGGER.info("[toy] uso i dataset compatti in data/processed/toy")
    cfg = apply_overrides(cfg, args.override)
    cfg = resolve_checkpoint_reference(cfg)
    return cfg


def _maybe_swap_to_eval_config(cfg_path: Path, cfg: Dict[str, Any] | None) -> Path:
    """Ritorna il config di valutazione equivalente a quello di training se presente."""
    if cfg and ("checkpoint" in cfg or "tasks" in cfg or "datasets" in cfg):
        return cfg_path
    if cfg_path.parent.name == "train":
        candidate = Path("configs") / "eval" / cfg_path.name
        if candidate.exists():
            LOGGER.info(
                "Config di training rilevato (%s): uso %s per la valutazione.",
                cfg_path,
                candidate,
            )
            return candidate
    return cfg_path


def _print_report(report: Dict[str, object]) -> None:
    """Pretty-print evaluation results grouping metrics by split and task."""
    print("=== Evaluation Report ===")
    for split_name, split_payload in report.get("splits", {}).items():
        print(f"\n[{split_name}]")
        tasks_payload = split_payload.get("tasks", {}) if isinstance(split_payload, dict) else {}
        if not tasks_payload:
            print("  (nessun task)")
            continue
        if "avg_loss" in split_payload:
            print(f"  avg_loss: {split_payload['avg_loss']:.4f}")
        for task_name, task_payload in tasks_payload.items():
            loss = task_payload.get("loss") if isinstance(task_payload, dict) else None
            samples = task_payload.get("num_samples", 0) if isinstance(task_payload, dict) else 0
            if loss is not None:
                print(f"  - {task_name} (n={samples}) loss={loss:.4f}")
            else:
                print(f"  - {task_name} (n={samples})")
            metrics = task_payload.get("metrics", {}) if isinstance(task_payload, dict) else {}
            for inner_task, vals in metrics.items():
                numbers = [
                    f"{metric_name}={vals[metric_name]:.2f}"
                    for metric_name in sorted(vals)
                    if metric_name != "samples"
                ]
                if numbers:
                    print(f"      · {inner_task}: {', '.join(numbers)}")
                print(f"        samples={vals.get('samples', 0)}")


def run_training(cfg: Dict[str, Any], *, overfit: bool = False) -> None:
    """Esegue l'intero ciclo di training partendo da un config strutturato."""
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    tokenizer_path = cfg.get("tokenizer_file") or cfg.get("data", {}).get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("Specificare 'tokenizer_file' nel config di training.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ensure_runtime_special_tokens(tokenizer)

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

    num_epochs = int(cfg.get("num_epochs", 1))
    grad_accum_steps = max(1, int(cfg.get("gradient_accumulation_steps", 1)))

    try:
        batches_per_epoch = len(train_loader)
    except TypeError:
        dataset_length = len(train_dataset)
        batches_per_epoch = math.ceil(dataset_length / batch_size) if batch_size > 0 else 0

    if batches_per_epoch == 0 or num_epochs <= 0:
        total_steps = 0
    else:
        optimizer_steps_per_epoch = math.ceil(batches_per_epoch / grad_accum_steps)
        total_steps = optimizer_steps_per_epoch * num_epochs

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
        grad_accum_steps=grad_accum_steps,
        log_every_n_steps=int(cfg.get("log_every_n_steps", 50)),
        checkpoint_path=str(checkpoint_path),
        early_stopping_patience=int(cfg.get("early_stopping", {}).get("patience", 5)),
        early_stopping_metric=str(cfg.get("early_stopping", {}).get("metric", "loss")),
        early_stopping_mode=str(cfg.get("early_stopping", {}).get("mode", "min")),
        wandb_run=run,
    )

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
    """Entry point for the ``train`` sub-command."""
    cfg = _prepare_config(args)
    run_training(cfg, overfit=False)


def cmd_overfit(args: argparse.Namespace) -> None:
    """Entry point for the ``overfit`` sub-command."""
    cfg = _prepare_config(args)
    run_training(cfg, overfit=True)

def enum_to_str(obj):
    if isinstance(obj, Enum):
        return obj.name
    return str(obj)

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Entry point for the ``evaluate`` sub-command that saves metrics to disk."""
    cfg_path = Path(args.cfg)
    raw_cfg = load_yaml(cfg_path)
    effective_cfg_path = _maybe_swap_to_eval_config(cfg_path, raw_cfg)
    args_dict = vars(args).copy()
    args_dict["cfg"] = str(effective_cfg_path)
    args = argparse.Namespace(**args_dict)

    cfg = _prepare_config(args)

    output_path = args.output or cfg.get("output_json")
    if output_path is None:
        output_path = effective_cfg_path.with_suffix(".report.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_run, wandb_module = maybe_init_wandb(cfg)
    try:
        report = evaluate_from_config(cfg)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=enum_to_str)
        LOGGER.info("Report salvato in %s", output_path.resolve())
        _print_report(report)

        if wandb_run is not None:
            flat_metrics = flatten_eval_metrics(report)
            try:
                wandb_run.log(flat_metrics)
            except Exception as exc:  # pragma: no cover - logging best effort
                LOGGER.warning("Log su Weights & Biases fallito: %s", exc)
    finally:
        if wandb_run is not None and wandb_module is not None:
            try:
                wandb_module.finish()
            except Exception as exc:  # pragma: no cover - dipende da env
                LOGGER.warning("Chiusura wandb fallita: %s", exc)


def _prepare_predict_input(text: str, task: str | None) -> str:
    """Attach the task marker to *text* when necessary for decoding."""
    text = text.strip()
    if task:
        marker = TASK_MARKERS.get(task)
        if marker and marker not in text:
            if task == "rdfcomp1" and "<MASK>" not in text:
                text = f"{text} <MASK>".strip()
            else:
                text = f"{text} {marker}".strip()
    return text


def cmd_predict(args: argparse.Namespace) -> None:
    """Entry point for the ``predict`` sub-command."""
    overrides: Dict[str, Any] = {}
    if getattr(args, "model_override", None):
        overrides = apply_overrides({}, args.model_override)

    model, tokenizer, device, _ = load_model_and_tokenizer(
        args.tokenizer,
        args.checkpoint,
        device=args.device,
        overrides=overrides,
    )

    prepared_input = _prepare_predict_input(args.input, args.task)
    output = decode_to_text(
        model,
        tokenizer,
        prepared_input,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    print(output)


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser with the train/overfit/evaluate commands."""
    parser = argparse.ArgumentParser(description="Pipeline di training NanoSocrates")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Addestra il modello sul dataset indicato")
    add_common_overrides(p_train)

    p_overfit = sub.add_parser("overfit", help="Forza l'overfit di un singolo batch")
    add_common_overrides(p_overfit)

    p_eval = sub.add_parser("evaluate", help="Valuta un checkpoint sui task indicati")
    add_common_overrides(p_eval)
    p_eval.add_argument("--output", help="File JSON per salvare il report", default=None)

    p_predict = sub.add_parser("predict", help="Genera una predizione per un singolo input")
    p_predict.add_argument("--checkpoint", required=True)
    p_predict.add_argument("--tokenizer", required=True)
    p_predict.add_argument("--input", required=True, help="Input testuale o RDF linearizzato")
    p_predict.add_argument("--task", choices=sorted(TASK_MARKERS))
    p_predict.add_argument("--device", default="cuda")
    p_predict.add_argument("--max-new-tokens", type=int, default=128)
    p_predict.add_argument(
        "--model-override",
        nargs="*",
        default=[],
        help="Override opzionali dei parametri del modello (chiave=valore)",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the selected sub-command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "overfit":
        cmd_overfit(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:  # pragma: no cover - guardia difensiva
        raise ValueError(f"Comando sconosciuto: {args.command}")


if __name__ == "__main__":
    main()
