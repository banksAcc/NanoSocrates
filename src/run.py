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
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from src.decoding.base import decode_to_text
from src.eval.evaluate import evaluate_from_config, load_model_and_tokenizer
from src.data.builders import build_and_cache_datasets
from src.model.transformer import TinySeq2Seq
from src.training.dataloaders import (
    StaticBatchLoader,
    create_multitask_dataloader,
    materialise_single_batch,
)
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

STRUCTURED_RDF_TASKS = frozenset({"text2rdf", "rdfcomp2"})


def _decode_for_logging(tokenizer: Tokenizer, sequence: Any, pad_value: int | None) -> str:
    """Decode *sequence* removing tokens equal to *pad_value* when provided."""

    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()

    if pad_value is not None:
        filtered = [int(tok) for tok in sequence if int(tok) != int(pad_value)]
    else:
        filtered = [int(tok) for tok in sequence]

    if not filtered:
        return ""

    try:
        return tokenizer.decode(filtered, skip_special_tokens=False)
    except Exception as exc:  # pragma: no cover - logging best effort
        LOGGER.warning("Decodifica fallita durante la preview del batch: %s", exc)
        return ""


def _preview_dataloader_batch(
    loader: DataLoader,
    tokenizer: Tokenizer,
    *,
    limit: int,
) -> None:
    """Log a preview of the first batch emitted by *loader*."""

    if limit <= 0:
        LOGGER.info("Anteprima batch disattivata perché limit=%d", limit)
        return

    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        LOGGER.warning("Impossibile mostrare il batch: il DataLoader è vuoto.")
        return

    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        LOGGER.warning(
            "Token <pad> non trovato nel tokenizer: impossibile rimuovere il padding dalla preview."
        )

    label_pad_id = pad_id if pad_id is not None else -100

    batch_size = batch["input_ids"].size(0)
    LOGGER.info("Anteprima del primo batch (%d esempi)", batch_size)

    labels = batch.get("labels")
    if isinstance(labels, torch.Tensor):
        non_pad = int((labels != label_pad_id).sum().item())
        LOGGER.info("Token da predire nel batch (escludendo pad): %d", non_pad)

    tasks = batch.get("tasks") or []
    raw_inputs = batch.get("raw_input") or []
    raw_targets = batch.get("raw_target") or []

    max_examples = min(batch_size, limit)
    for idx in range(max_examples):
        example_labels = batch["labels"][idx]
        tokens_to_predict = int((example_labels != label_pad_id).sum().item())
        decoded_input = _decode_for_logging(tokenizer, batch["input_ids"][idx], pad_id)
        decoded_labels = _decode_for_logging(tokenizer, example_labels, label_pad_id)

        header = f"Esempio {idx}"
        if tasks:
            header += f" | task={tasks[idx]}"
        LOGGER.info(header)
        LOGGER.info("  Token da predire: %d", tokens_to_predict)
        if raw_inputs and raw_inputs[idx]:
            LOGGER.info("  Input grezzo: %s", raw_inputs[idx])
        LOGGER.info("  Input decodificato: %s", decoded_input)
        if raw_targets and raw_targets[idx]:
            LOGGER.info("  Target grezzo: %s", raw_targets[idx])
        LOGGER.info("  Labels decodificate: %s", decoded_labels)

    if max_examples < batch_size:
        LOGGER.info(
            "(Anteprima limitata a %d esempi su %d: usare --print-batch-limit per aumentare)",
            max_examples,
            batch_size,
        )


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
        max_position_embeddings=int(cfg.get("max_len", 256)),
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


def run_training(
    cfg: Dict[str, Any],
    *,
    overfit: bool = False,
    overfit_steps: int | None = None,
    preview_batch: bool = False,
    preview_batch_limit: int = 3,
) -> None:
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

    if overfit:
        cached_batch = materialise_single_batch(
            train_dataset, tokenizer=tokenizer, batch_size=batch_size
        )
        batch_size = cached_batch["input_ids"].size(0)
        LOGGER.info(
            "Modalità overfit: riutilizzo un batch statico da %d esempi",
            batch_size,
        )
        train_loader = StaticBatchLoader(cached_batch)
        val_loader = StaticBatchLoader(cached_batch)
    else:
        train_loader = create_multitask_dataloader(
            train_dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            ratios=ratios,
            num_workers=num_workers,
            shuffle=True,
        )
        val_loader = create_multitask_dataloader(
            val_dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            ratios=ratios,
            num_workers=num_workers,
            shuffle=False,
        )

    if preview_batch:
        _preview_dataloader_batch(train_loader, tokenizer, limit=preview_batch_limit)

    num_epochs = int(cfg.get("num_epochs", 1))
    # If overfitting, allow mapping an explicit step count to epochs (1 batch/epoch)
    if overfit and overfit_steps is not None and overfit_steps > 0:
        num_epochs = int(overfit_steps)
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
    """Entry point for the ``overfit`` sub-command.

    Applies safe defaults for overfitting a single batch: disables regularisation
    (dropout/weight decay), disables schedulers/warmup, increases logging
    frequency, and maps --steps to epochs when provided.
    """
    base_cfg = _prepare_config(args)
    cfg: Dict[str, Any] = dict(base_cfg)

    # Friendly defaults for overfit mode (can be overridden via --override)
    cfg.setdefault("dropout", 0.0)
    cfg["weight_decay"] = 0.0
    cfg["scheduler"] = ""  # disable scheduler
    cfg["warmup_ratio"] = 0.0
    cfg.setdefault("log_every_n_steps", 1)
    cfg.setdefault("num_workers", 0)

    # Ensure single-batch dataset by default
    if "batch_size" in cfg and "overfit_samples" not in cfg:
        try:
            cfg["overfit_samples"] = int(cfg["batch_size"])
        except Exception:
            cfg["overfit_samples"] = 8

    # Disable early stopping by setting a very large patience
    early = dict(cfg.get("early_stopping", {}) or {})
    early["patience"] = int(1e9)
    cfg["early_stopping"] = early

    overfit_steps = getattr(args, "steps", None)
    if overfit_steps is not None and overfit_steps > 0:
        cfg["num_epochs"] = int(overfit_steps)

    preview_limit = int(getattr(args, "print_batch_limit", 3))
    run_training(
        cfg,
        overfit=True,
        overfit_steps=overfit_steps,
        preview_batch=bool(getattr(args, "print_batch", False)),
        preview_batch_limit=preview_limit,
    )

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
    enforce_rdf = (args.task in STRUCTURED_RDF_TASKS) if args.task else False
    if getattr(args, "enforce_rdf_grammar", False):
        enforce_rdf = True
    if getattr(args, "disable_rdf_grammar", False):
        enforce_rdf = False

    output = decode_to_text(
        model,
        tokenizer,
        prepared_input,
        max_new_tokens=args.max_new_tokens,
        device=device,
        use_beam_search=args.use_beam_search,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        early_stopping=not args.no_early_stopping,
        enforce_rdf_grammar=enforce_rdf,
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
    p_overfit.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Numero di aggiornamenti consecutivi sullo stesso batch (default: 200)",
    )
    p_overfit.add_argument(
        "--print-batch",
        action="store_true",
        help="Stampa il primo batch del DataLoader per ispezionare input e label",
    )
    p_overfit.add_argument(
        "--print-batch-limit",
        type=int,
        default=3,
        help="Numero massimo di esempi da mostrare quando si usa --print-batch (default: 3)",
    )

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
    p_predict.add_argument("--use-beam-search", action="store_true")
    p_predict.add_argument("--beam-size", type=int, default=4)
    p_predict.add_argument("--length-penalty", type=float, default=1.0)
    p_predict.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p_predict.add_argument("--repetition-penalty", type=float, default=1.1)
    p_predict.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disabilita l'early stopping durante il beam search",
    )
    p_predict.add_argument(
        "--enforce-rdf-grammar",
        action="store_true",
        help="Forza il vincolo grammaticale RDF anche per task non strutturati",
    )
    p_predict.add_argument(
        "--disable-rdf-grammar",
        action="store_true",
        help="Disattiva il vincolo grammaticale RDF anche per task strutturati",
    )
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
