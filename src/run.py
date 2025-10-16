"""Minimal CLI per addestrare e valutare il modello NanoSocrates."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tokenizers import Tokenizer

from src.data.builders import build_and_cache_datasets
from src.model.transformer import TinySeq2Seq
from src.tokenizer.tokenizer_io import ensure_runtime_special_tokens
from src.training.simple_training import build_dataloader, evaluate_model, train_one_epoch
from src.utils.config import (
    add_common_overrides,
    apply_overrides,
    apply_toy_paths,
    load_yaml,
    resolve_checkpoint_reference,
)

LOGGER = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - dipende dall'hw della CI
        torch.cuda.manual_seed_all(seed)


def _select_device(want: str | None) -> torch.device:
    want = (want or "cuda").lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_tokenizer_from_config(cfg: Dict[str, Any]) -> Tokenizer:
    path = cfg.get("tokenizer_file") or cfg.get("data", {}).get("tokenizer_path")
    if not path:
        raise ValueError("Specificare 'tokenizer_file' nel file di configurazione.")
    tokenizer = Tokenizer.from_file(str(path))
    ensure_runtime_special_tokens(tokenizer)
    return tokenizer


def _pad_id(tokenizer: Tokenizer) -> int:
    pad = tokenizer.token_to_id("<pad>")
    if pad is None:
        raise ValueError("Tokenizer privo di <pad>: rigenera il vocabolario includendo <pad>.")
    return int(pad)


def _build_model(cfg: Dict[str, Any], tokenizer: Tokenizer, device: torch.device) -> TinySeq2Seq:
    pad_id = _pad_id(tokenizer)
    model = TinySeq2Seq(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 4)),
        num_encoder_layers=int(cfg.get("enc_layers", 2)),
        num_decoder_layers=int(cfg.get("dec_layers", 2)),
        dim_feedforward=int(cfg.get("ff_dim", 1024)),
        dropout=float(cfg.get("dropout", 0.1)),
        pad_id=pad_id,
        tie_embeddings=True,
        max_position_embeddings=int(cfg.get("max_len", 256)),
        relative_attention_num_buckets=int(cfg.get("relative_attention_num_buckets", 32)),
        relative_attention_max_distance=int(cfg.get("relative_attention_max_distance", 128)),
        layer_norm_epsilon=float(cfg.get("layer_norm_epsilon", 1e-6)),
    )
    return model.to(device)


def _load_model_from_checkpoint(
    tokenizer: Tokenizer,
    checkpoint: str,
    device: torch.device,
    fallback_cfg: Dict[str, Any],
) -> TinySeq2Seq:
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
        saved_cfg = payload.get("config", {})
    else:
        state_dict = payload
        saved_cfg = {}

    merged_cfg = dict(fallback_cfg)
    merged_cfg.update(saved_cfg)

    model = _build_model(merged_cfg, tokenizer, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _save_checkpoint(path: Path, model: TinySeq2Seq) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.export_config()}, path)


def _format_metrics(metrics: Dict[str, Any]) -> str:
    return f"loss={metrics['loss']:.4f} exact_match={metrics['exact_match'] * 100:.2f}%"


def _prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml(args.cfg)
    if getattr(args, "toy", False):
        cfg = apply_toy_paths(cfg)
    cfg = apply_overrides(cfg, args.override)
    cfg = resolve_checkpoint_reference(cfg)
    return cfg


def run_training(cfg: Dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    tokenizer = _get_tokenizer_from_config(cfg)
    datasets = build_and_cache_datasets(cfg, tokenizer)
    train_dataset = datasets["train"]
    val_dataset = datasets.get("validation")
    test_dataset = datasets.get("test")
    pad_id = _pad_id(tokenizer)

    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))
    train_loader = build_dataloader(
        train_dataset,
        tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = (
        build_dataloader(val_dataset, tokenizer, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_dataset
        else None
    )
    test_loader = (
        build_dataloader(test_dataset, tokenizer, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if test_dataset
        else None
    )

    device = _select_device(cfg.get("device"))
    LOGGER.info("Uso il device %s", device)

    model = _build_model(cfg, tokenizer, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    num_epochs = int(cfg.get("num_epochs", 1))
    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
    max_grad_norm = cfg.get("max_grad_norm")

    save_dir = Path(cfg.get("save_dir", "checkpoints"))
    checkpoint_path = save_dir / "best.pt"

    best_val_loss: float | None = None
    best_val_metrics: Dict[str, Any] | None = None
    best_epoch = 0

    if num_epochs <= 0:
        LOGGER.info("num_epochs=%d: salto la fase di training.", num_epochs)
        _save_checkpoint(checkpoint_path, model)
    else:
        for epoch in range(1, num_epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                grad_accum_steps=grad_accum,
                max_grad_norm=float(max_grad_norm) if max_grad_norm else None,
            )
            LOGGER.info("Epoch %d train %s", epoch, _format_metrics(train_metrics))

            if val_loader is not None:
                val_metrics = evaluate_model(
                    model,
                    val_loader,
                    device,
                    pad_id=pad_id,
                )
                LOGGER.info("Epoch %d val %s", epoch, _format_metrics(val_metrics))
                if best_val_loss is None or val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_val_metrics = val_metrics
                    best_epoch = epoch
                    _save_checkpoint(checkpoint_path, model)
            else:
                LOGGER.info("Epoch %d completed", epoch)

        if val_loader is None:
            _save_checkpoint(checkpoint_path, model)

    if best_val_metrics:
        LOGGER.info(
            "Miglior validazione all'epoch %d: %s",
            best_epoch,
            _format_metrics(best_val_metrics),
        )
        LOGGER.info("Checkpoint salvato in %s", checkpoint_path)
    else:
        LOGGER.info("Checkpoint salvato in %s", checkpoint_path)

    if test_loader is not None:
        if checkpoint_path.exists():
            eval_model = _load_model_from_checkpoint(tokenizer, str(checkpoint_path), device, cfg)
        else:
            eval_model = model
        test_metrics = evaluate_model(eval_model, test_loader, device, pad_id=pad_id)
        LOGGER.info("Test %s", _format_metrics(test_metrics))
        tasks = test_metrics.get("tasks", {})
        for task, info in tasks.items():
            LOGGER.info(
                "  - %s exact_match=%.2f%% (n=%d)",
                task,
                info["exact_match"] * 100,
                info["samples"],
            )


def run_evaluation(cfg: Dict[str, Any], checkpoint: str, splits: list[str]) -> None:
    tokenizer = _get_tokenizer_from_config(cfg)
    datasets = build_and_cache_datasets(cfg, tokenizer)

    device = _select_device(cfg.get("device"))
    LOGGER.info("Uso il device %s", device)

    model = _load_model_from_checkpoint(tokenizer, checkpoint, device, cfg)
    pad_id = _pad_id(tokenizer)

    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))

    for split in splits:
        dataset = datasets.get(split)
        if dataset is None:
            LOGGER.warning("Split '%s' non disponibile nel config", split)
            continue
        loader = build_dataloader(
            dataset,
            tokenizer,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        metrics = evaluate_model(model, loader, device, pad_id=pad_id)
        LOGGER.info("[%s] %s", split, _format_metrics(metrics))
        tasks = metrics.get("tasks", {})
        for task, info in tasks.items():
            LOGGER.info(
                "  - %s exact_match=%.2f%% (n=%d)",
                task,
                info["exact_match"] * 100,
                info["samples"],
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline semplificata NanoSocrates")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Addestra il modello sui dataset indicati")
    add_common_overrides(p_train)

    p_eval = sub.add_parser("evaluate", help="Valuta un checkpoint esistente")
    add_common_overrides(p_eval)
    p_eval.add_argument("--checkpoint", required=False, help="Path al checkpoint da valutare")
    p_eval.add_argument(
        "--splits",
        nargs="*",
        default=["validation", "test"],
        help="Lista di split su cui calcolare le metriche (default: validation test)",
    )

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()

    if args.command == "train":
        cfg = _prepare_config(args)
        run_training(cfg)
    elif args.command == "evaluate":
        cfg = _prepare_config(args)
        checkpoint = args.checkpoint or cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "Specifica il checkpoint da valutare (via --checkpoint o nel file di config con 'checkpoint')."
            )
        run_evaluation(cfg, str(checkpoint), list(args.splits))
    else:  # pragma: no cover - guardia difensiva
        raise ValueError(f"Comando sconosciuto: {args.command}")


if __name__ == "__main__":
    main()
