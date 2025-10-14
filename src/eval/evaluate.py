"""Valutazione dei checkpoint con spiegazioni passo-passo."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from src.decoding.base import decode_to_text
from src.eval.metrics import compute_accuracy, compute_text_generation_metrics, compute_triple_metrics
from src.model.transformer import TinySeq2Seq
from src.tokenizer.tokenizer_io import TokWrapper
from src.training.dataloaders import JsonlSeq2Seq, pad_collate

LOGGER = logging.getLogger(__name__)

def _select_device(want: Optional[str]) -> str:
    """Sceglie "cuda" solo se disponibile, altrimenti ripiega su CPU."""

    want = (want or "cuda").lower()
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalise_text(text: str) -> str:
    """Pulisce testo generato eliminando padding e newline."""

    if not text:
        return ""
    cleaned = [tok for tok in text.replace("\n", " ").split() if tok and tok != "<pad>"]
    return " ".join(cleaned).strip()


def _summarise_lengths(lengths: Iterable[int]) -> Dict[str, float]:
    """Calcola statistiche robuste (min, max, media, mediana) su una sequenza di lunghezze."""

    values = [int(max(0, length)) for length in lengths]
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "zeros": 0}

    values.sort()
    count = len(values)
    zeros = sum(1 for value in values if value == 0)
    mean = float(sum(values) / count)
    if count % 2:
        median = float(values[count // 2])
    else:
        median = float(values[count // 2 - 1] + values[count // 2]) / 2.0

    return {
        "count": count,
        "min": int(values[0]),
        "max": int(values[-1]),
        "mean": mean,
        "median": median,
        "zeros": int(zeros),
    }


def _load_model_from_checkpoint(
    checkpoint_path: str,
    tokenizer: TokWrapper,
    device: str,
    overrides: Optional[Mapping[str, object]] = None,
) -> tuple[TinySeq2Seq, Dict[str, object]]:
    """Ricostruisce il modello a partire da un checkpoint salvato."""

    overrides = dict(overrides or {})
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, Mapping):
        saved_cfg: MutableMapping[str, object] = dict(ckpt.get("config", {}))
        state_dict = ckpt.get("model", ckpt)
    else:
        saved_cfg = {}
        state_dict = ckpt
    saved_cfg.update(overrides)

    required = ["d_model", "nhead", "enc_layers", "dec_layers", "ff_dim", "dropout"]
    missing = [k for k in required if k not in saved_cfg]
    if missing:
        raise ValueError(
            f"Checkpoint privo dei parametri modello {missing}. Fornisci override espliciti."
        )

    model = TinySeq2Seq(
        vocab_size=tokenizer.vocab_size(),
        d_model=int(saved_cfg["d_model"]),
        nhead=int(saved_cfg["nhead"]),
        num_encoder_layers=int(saved_cfg["enc_layers"]),
        num_decoder_layers=int(saved_cfg["dec_layers"]),
        dim_feedforward=int(saved_cfg["ff_dim"]),
        dropout=float(saved_cfg["dropout"]),
        pad_id=tokenizer.pad_id,
        tie_embeddings=True,
        use_mla=bool(saved_cfg.get("use_mla", False)),
        use_rope=bool(saved_cfg.get("use_rope", False)),
        interleave_ratio=float(saved_cfg.get("interleave_ratio", 0.0)),
        max_position_embeddings=int(saved_cfg.get("max_len", 256)),
        compute_span_metrics=bool(saved_cfg.get("compute_span_metrics", False)),
        architecture=str(saved_cfg.get("architecture", "vanilla")),
        relative_attention_num_buckets=int(saved_cfg.get("relative_attention_num_buckets", 32)),
        relative_attention_max_distance=int(saved_cfg.get("relative_attention_max_distance", 128)),
        layer_norm_epsilon=float(saved_cfg.get("layer_norm_epsilon", 1e-6)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, dict(saved_cfg)


def load_model_and_tokenizer(
    tokenizer_file: str,
    checkpoint_path: str,
    *,
    device: Optional[str] = None,
    overrides: Optional[Mapping[str, object]] = None,
) -> tuple[TinySeq2Seq, TokWrapper, str, Dict[str, object]]:
    """Carica coppia tokenizer+modello e restituisce anche il device scelto."""

    device_sel = _select_device(device)
    tokenizer = TokWrapper(tokenizer_file)
    model, saved_cfg = _load_model_from_checkpoint(
        checkpoint_path, tokenizer, device_sel, overrides
    )
    return model, tokenizer, device_sel, saved_cfg


@torch.no_grad()
def _compute_loss(model: TinySeq2Seq, dataloader: DataLoader, device: str) -> float:
    """Calcola la loss media iterando sul dataloader indicato."""

    total = 0.0
    steps = 0
    for batch in dataloader:
        inp = batch["input_ids"].to(device, non_blocking=True)
        att = batch["attention_mask"].to(device, non_blocking=True)
        lab = batch["labels"].to(device, non_blocking=True)
        extra = {}
        if "mask_positions" in batch:
            extra["mask_positions"] = batch["mask_positions"].to(device, non_blocking=True)
        if "mask_lengths" in batch:
            extra["mask_lengths"] = batch["mask_lengths"].to(device, non_blocking=True)
        out = model(inp, att, labels=lab, **extra)
        loss = out.get("loss")
        if loss is None:
            continue
        total += float(loss.item())
        steps += 1
    if steps == 0:
        return 0.0
    return total / steps


@torch.no_grad()
def _generate_predictions(
    model: TinySeq2Seq,
    tokenizer: TokWrapper,
    dataset: JsonlSeq2Seq,
    device: str,
    max_new_tokens: int,
    *,
    normalise: bool = True,
    return_raw: bool = False,
) -> tuple[List[str], List[str], List[str]] | tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Genera predizioni greedy restituendo anche i task di provenienza."""

    predictions: List[str] = []
    references: List[str] = []
    raw_predictions: List[str] = []
    raw_references: List[str] = []
    tasks: List[str] = []
    token_sequences: List[List[int]] = []

    for ex in dataset.items:
        source: str = ""
        target: str = ""
        task_name: str | None = None

        if isinstance(ex, Mapping):
            source = str(
                ex.get("input")
                or ex.get("source")
                or ex.get("raw_input")
                or ""
            )
            target = str(
                ex.get("target")
                or ex.get("output")
                or ex.get("label")
                or ex.get("raw_target")
                or ""
            )
            task_name = ex.get("task") or ex.get("task_name")
        else:
            source = str(
                getattr(ex, "input_text", None)
                or getattr(ex, "input", "")
            )
            target = str(
                getattr(ex, "target_text", None)
                or getattr(ex, "target", None)
                or getattr(ex, "label", "")
            )
            task_name = getattr(ex, "task", None)

        if not source:
            continue

        pred, token_ids = decode_to_text(
            model,
            tokenizer,
            source,
            max_new_tokens=max_new_tokens,
            device=device,
            min_new_tokens=min_new_tokens,
            debug=debug_generation,
            return_ids=True,
        )
        raw_predictions.append(pred)
        raw_references.append(target)
        if normalise:
            predictions.append(_normalise_text(pred))
            references.append(_normalise_text(target))
        else:
            predictions.append(pred)
            references.append(target)
        tasks.append(str(task_name or "unknown"))
    if return_raw:
        return predictions, references, tasks, raw_predictions, raw_references
    return predictions, references, tasks


def _extract_example_fields(example) -> Tuple[str, str, str, Optional[str]]:
    """Return raw input/target/task/film strings from a dataset item."""

    if isinstance(example, Mapping):
        source = str(
            example.get("raw_input")
            or example.get("input")
            or example.get("source")
            or ""
        )
        target = str(
            example.get("raw_target")
            or example.get("target")
            or example.get("output")
            or example.get("label")
            or ""
        )
        task_name = str(example.get("task") or example.get("task_name") or "")
        film = example.get("film")
    else:
        source = str(
            getattr(example, "input_text", None)
            or getattr(example, "input", None)
            or ""
        )
        target = str(
            getattr(example, "target_text", None)
            or getattr(example, "target", None)
            or getattr(example, "label", None)
            or ""
        )
        task_name = str(getattr(example, "task", "") or "")
        film = getattr(example, "film", None)
    return source, target, task_name, film


def _build_preview_records(
    dataset: JsonlSeq2Seq,
    predictions: Sequence[str],
    references: Sequence[str],
    raw_predictions: Sequence[str],
    tasks: Sequence[str],
    *,
    limit: int,
    per_task: bool = True,
) -> Dict[str, List[Dict[str, object]]]:
    """Create a limited set of per-task preview records for debugging."""

    if limit <= 0:
        return {}

    previews: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    total_added = 0

    for idx, (pred, ref, raw_pred, task_tag) in enumerate(
        zip(predictions, references, raw_predictions, tasks)
    ):
        bucket = str(task_tag or "unknown")
        if per_task:
            if counters[bucket] >= limit:
                continue
        else:
            if total_added >= limit:
                break
        source, target, dataset_task, film = _extract_example_fields(dataset.items[idx])
        record = {
            "index": idx,
            "task": bucket,
            "dataset_task": dataset_task,
            "film": film,
            "input": source,
            "target": target,
            "prediction": raw_pred,
            "prediction_normalised": pred,
            "target_normalised": ref,
            "exact_match": bool(pred == ref),
        }
        previews[bucket].append(record)
        counters[bucket] += 1
        total_added += 1
    return dict(previews)


def _write_preview_files(
    previews: Mapping[str, List[Dict[str, object]]],
    *,
    output_dir: Path,
    split: str,
    task: str,
) -> Optional[Path]:
    """Persist preview records to disk and return the output path."""

    if not previews:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / f"{split}_{task}.jsonl"
    with preview_path.open("w", encoding="utf-8") as f:
        for task_name, records in previews.items():
            for record in records:
                payload = {"group": task_name, **record}
                json.dump(payload, f, ensure_ascii=False)
                f.write("\n")
    return preview_path


def _group_by_task(
    predictions: Iterable[str],
    references: Iterable[str],
    tasks: Iterable[str],
) -> Dict[str, Dict[str, List[str]]]:
    """Aggrega predizioni e riferimenti per task nominale."""

    buckets: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"pred": [], "ref": []})
    for pred, ref, task in zip(predictions, references, tasks):
        buckets[task]["pred"].append(pred)
        buckets[task]["ref"].append(ref)
    return buckets


def _metrics_for_task(task: str, preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Seleziona automaticamente la metrica corretta in base al task."""

    if task == "rdf2text":
        return compute_text_generation_metrics(preds, refs)
    if task in {"text2rdf", "rdfcomp2"}:
        return compute_triple_metrics(preds, refs)
    if task == "rdfcomp1":
        return compute_accuracy(preds, refs)
    # fallback: treat as generation
    return compute_text_generation_metrics(preds, refs)


def _normalise_tasks_config(raw_tasks) -> Dict[str, Dict[str, str]]:
    """Uniforma i vari formati YAML in un dizionario standard."""

    if isinstance(raw_tasks, Mapping):
        return {str(name): dict(cfg) for name, cfg in raw_tasks.items()}
    tasks_dict: Dict[str, Dict[str, str]] = {}
    for entry in raw_tasks or []:
        if isinstance(entry, Mapping):
            name = entry.get("name") or entry.get("task")
            if not name:
                raise ValueError("Ogni task deve avere un campo 'name' o 'task'.")
            tasks_dict[str(name)] = dict(entry)
        else:
            raise ValueError("Formato task non valido. Usa dict o lista di dict.")
    return tasks_dict


def evaluate_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    """Esegue la valutazione completa seguendo i path indicati nel config."""

    if "checkpoint" not in config:
        raise ValueError("Il config di valutazione richiede 'checkpoint'.")
    if "tokenizer_file" not in config:
        raise ValueError("Il config di valutazione richiede 'tokenizer_file'.")
    if "tasks" not in config and "datasets" not in config:
        raise ValueError("Specifica la sezione 'tasks' con i path val/test per task.")

    model_overrides = config.get("model") or {}
    model, tokenizer, device, saved_cfg = load_model_and_tokenizer(
        str(config["tokenizer_file"]),
        str(config["checkpoint"]),
        device=str(config.get("device", "cuda")),
        overrides=model_overrides,
    )

    max_len = int(config.get("max_len", saved_cfg.get("max_len", 256)))
    decode_cfg = config.get("decoding") or {}
    max_new_tokens = int(decode_cfg.get("max_new_tokens", max_len))
    min_new_tokens_raw = decode_cfg.get("min_new_tokens", 1)
    try:
        min_new_tokens = max(0, int(min_new_tokens_raw))
    except (TypeError, ValueError):
        min_new_tokens = 1
    debug_generation_raw = decode_cfg.get(
        "debug_generation",
        decode_cfg.get("debug_token_ids", decode_cfg.get("debug", False)),
    )
    if isinstance(debug_generation_raw, str):
        debug_generation = debug_generation_raw.lower() not in {"false", "0", "no"}
    else:
        debug_generation = bool(debug_generation_raw)

    preview_cfg = config.get("preview") or {}
    preview_limit_raw = preview_cfg.get("limit", config.get("preview_samples", 0))
    try:
        preview_limit = int(preview_limit_raw)
    except (TypeError, ValueError):
        preview_limit = 0
    per_task = preview_cfg.get("per_task", True)
    if isinstance(per_task, str):
        per_task = per_task.lower() not in {"false", "0", "no"}
    else:
        per_task = bool(per_task)
    preview_output_dir_raw = (
        preview_cfg.get("output_dir")
        or preview_cfg.get("dir")
        or config.get("preview_output_dir")
    )
    preview_output_dir = Path(preview_output_dir_raw) if preview_output_dir_raw else None

    preview_cfg = config.get("preview") or {}
    preview_limit_raw = preview_cfg.get("limit", config.get("preview_samples", 0))
    try:
        preview_limit = int(preview_limit_raw)
    except (TypeError, ValueError):
        preview_limit = 0
    per_task = preview_cfg.get("per_task", True)
    if isinstance(per_task, str):
        per_task = per_task.lower() not in {"false", "0", "no"}
    else:
        per_task = bool(per_task)
    preview_output_dir_raw = (
        preview_cfg.get("output_dir")
        or preview_cfg.get("dir")
        or config.get("preview_output_dir")
    )
    preview_output_dir = Path(preview_output_dir_raw) if preview_output_dir_raw else None

    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = device == "cuda"

    tasks_cfg = _normalise_tasks_config(
        config.get("tasks") if config.get("tasks") is not None else config.get("datasets")
    )

    enable_entity_spans = bool(
        config.get(
            "enable_entity_spans",
            saved_cfg.get("enable_entity_spans", False),
        )
    )

    collate = partial(pad_collate, pad_id=tokenizer.pad_id)

    report: Dict[str, object] = {
        "checkpoint": os.path.abspath(str(config["checkpoint"])),
        "device": device,
        "splits": {},
    }

    for split in ("val", "test"):
        split_payload: Dict[str, object] = {"tasks": {}}
        weighted_loss = 0.0
        total_samples = 0
        for task_name, cfg_task in tasks_cfg.items():
            path = cfg_task.get(split)
            if not path:
                continue
            dataset = JsonlSeq2Seq(
                str(path),
                tokenizer=tokenizer,
                max_len=max_len,
                enable_entity_spans=enable_entity_spans,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            loss = _compute_loss(model, dataloader, device)
            want_preview = preview_limit > 0
            if want_preview:
                preds, refs, task_tags, raw_preds, _ = _generate_predictions(
                    model,
                    tokenizer,
                    dataset,
                    device,
                    max_new_tokens,
                    return_raw=True,
                )
            else:
                preds, refs, task_tags = _generate_predictions(
                    model,
                    tokenizer,
                    dataset,
                    device,
                    max_new_tokens,
                    return_raw=False,
                )
                raw_preds = preds
            grouped = _group_by_task(preds, refs, task_tags)
            metrics_payload: Dict[str, object] = {}
            for t_name, bucket in grouped.items():
                metrics = _metrics_for_task(t_name, bucket["pred"], bucket["ref"])
                metrics_payload[t_name] = {
                    **{k: float(v) for k, v in metrics.items()},
                    "samples": len(bucket["pred"]),
                }

            if not metrics_payload:
                metrics_payload[task_name] = {"samples": len(dataset)}

            previews = {}
            if want_preview:
                previews = _build_preview_records(
                    dataset,
                    preds,
                    refs,
                    raw_preds,
                    task_tags,
                    limit=preview_limit,
                    per_task=per_task,
                )

            split_payload["tasks"][task_name] = {
                "path": str(path),
                "loss": float(loss),
                "num_samples": len(dataset),
                "metrics": metrics_payload,
                "diagnostics": diagnostics,
            }
            if previews:
                split_payload["tasks"][task_name]["preview"] = previews
                if preview_output_dir is not None:
                    preview_file = _write_preview_files(
                        previews,
                        output_dir=preview_output_dir,
                        split=split,
                        task=task_name,
                    )
                    if preview_file is not None:
                        split_payload["tasks"][task_name]["preview_file"] = str(
                            preview_file
                        )
            weighted_loss += loss * len(dataset)
            total_samples += len(dataset)

        if total_samples > 0:
            split_payload["avg_loss"] = float(weighted_loss / total_samples)
            split_payload["num_samples"] = total_samples
        report["splits"][split] = split_payload

    return report
