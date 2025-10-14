"""Valutazione dei checkpoint con spiegazioni passo-passo."""

from __future__ import annotations

import json
import logging
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


STRUCTURAL_DECODE_TOKENS: tuple[str, ...] = (
    "<SUBJ>",
    "<PRED>",
    "<OBJ>",
    "<OBJ_LIST>",
    "|",
    "<Text2RDF>",
    "<RDF2Text>",
    "<CONTINUERDF>",
    "<MASK>",
)

TEXTUAL_OUTPUT_TASKS: frozenset[str] = frozenset({"rdf2text", "rdfcomp1"})


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
    min_new_tokens: int = 0,
    *,
    normalise: bool = True,
    return_raw: bool = False,
    debug_generation: bool = False,
) -> (
    tuple[List[str], List[str], List[str]]
    | tuple[List[str], List[str], List[str], List[str], List[str], List[List[int]]]
):
    """Genera predizioni greedy restituendo anche i task di provenienza."""

    predictions: List[str] = []
    references: List[str] = []
    raw_predictions: List[str] = []
    raw_references: List[str] = []
    tasks: List[str] = []
    token_sequences: List[List[int]] = []

    pad_token_id = getattr(tokenizer, "pad_id", None)
    structural_token_ids: Tuple[int, ...] = tuple(
        int(token_id)
        for token_id in (
            tokenizer.token_to_id(token) for token in STRUCTURAL_DECODE_TOKENS
        )
        if token_id is not None
    )
    base_forbidden_ids: Tuple[int, ...]
    if pad_token_id is None:
        base_forbidden_ids = structural_token_ids
    else:
        base_forbidden_ids = tuple(
            dict.fromkeys(structural_token_ids + (int(pad_token_id),))
        )

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

        task_slug = str(task_name or "").lower()
        forbidden_ids = base_forbidden_ids if task_slug in TEXTUAL_OUTPUT_TASKS else ()

        pred, token_ids = decode_to_text(
            model,
            tokenizer,
            source,
            max_new_tokens=max_new_tokens,
            device=device,
            min_new_tokens=min_new_tokens,
            debug=debug_generation,
            return_ids=True,
            forbidden_token_ids=forbidden_ids,
        )
        raw_predictions.append(pred)
        raw_references.append(target)
        token_sequences.append(list(token_ids))
        if normalise:
            predictions.append(_normalise_text(pred))
            references.append(_normalise_text(target))
        else:
            predictions.append(pred)
            references.append(target)
        tasks.append(str(task_name or "unknown"))
    if return_raw:
        return (
            predictions,
            references,
            tasks,
            raw_predictions,
            raw_references,
            token_sequences,
        )
    return predictions, references, tasks


def _extract_example_fields(example) -> Tuple[str, str, str, Optional[str]]:
    """Return raw input/target/task/film strings from a dataset item."""

    def _normalise_optional_str(value):
        if value is None:
            return None
        return str(value)

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
        film = _normalise_optional_str(example.get("film"))
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
        film = _normalise_optional_str(getattr(example, "film", None))
    return source, target, task_name, film


def _build_preview_records(
    dataset: JsonlSeq2Seq,
    predictions: Sequence[str],
    references: Sequence[str],
    raw_predictions: Sequence[str],
    tasks: Sequence[str],
    token_ids: Sequence[Sequence[int]],
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

    for idx, (pred, ref, raw_pred, task_tag, token_seq) in enumerate(
        zip(predictions, references, raw_predictions, tasks, token_ids)
    ):
        bucket = str(task_tag or "unknown")
        if per_task:
            if counters[bucket] >= limit:
                continue
        else:
            if total_added >= limit:
                break
        source, target, dataset_task, film = _extract_example_fields(dataset.items[idx])
        example = dataset.items[idx]
        record: Dict[str, object] = {
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
            "input_tokens": len(getattr(example, "input_ids", []) or []),
            "target_tokens": len(getattr(example, "label_ids", []) or []),
            "prediction_chars": len(raw_pred.strip()),
            "prediction_tokens": len(raw_pred.split()),
            "prediction_token_ids": list(token_seq),
        }
        warnings: List[str] = []
        if not raw_pred.strip():
            warnings.append("empty_prediction")
        if len(getattr(example, "input_ids", [])) >= getattr(dataset, "max_len", 0) > 0:
            warnings.append("input_truncated")
        if len(getattr(example, "label_ids", [])) >= getattr(dataset, "max_len", 0) > 0:
            warnings.append("target_truncated")
        if warnings:
            record["warnings"] = warnings
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
                (
                    preds,
                    refs,
                    task_tags,
                    raw_preds,
                    _raw_refs,
                    token_ids,
                ) = _generate_predictions(
                    model,
                    tokenizer,
                    dataset,
                    device,
                    max_new_tokens,
                    min_new_tokens,
                    return_raw=True,
                    debug_generation=debug_generation,
                )
            else:
                preds, refs, task_tags = _generate_predictions(
                    model,
                    tokenizer,
                    dataset,
                    device,
                    max_new_tokens,
                    min_new_tokens,
                    return_raw=False,
                    debug_generation=debug_generation,
                )
                raw_preds = preds
                token_ids = [[] for _ in preds]
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
                    token_ids,
                    limit=preview_limit,
                    per_task=per_task,
                )

            input_token_lengths = [len(example.input_ids) for example in dataset.items]
            target_token_lengths = [len(example.label_ids) for example in dataset.items]
            prediction_char_lengths = [len(pred.strip()) for pred in raw_preds]
            prediction_token_lengths = [len(pred.split()) for pred in raw_preds]

            diagnostics: Dict[str, object] = {
                "input_tokens": _summarise_lengths(input_token_lengths),
                "target_tokens": _summarise_lengths(target_token_lengths),
                "prediction_chars": _summarise_lengths(prediction_char_lengths),
                "prediction_tokens": _summarise_lengths(prediction_token_lengths),
                "inputs_truncated": int(
                    sum(1 for length in input_token_lengths if length >= dataset.max_len)
                ),
                "targets_truncated": int(
                    sum(1 for length in target_token_lengths if length >= dataset.max_len)
                ),
            }

            try:
                unique_inputs = len({example.input_text for example in dataset.items})
            except TypeError:
                unique_inputs = len(dataset.items)
            diagnostics["unique_inputs"] = int(unique_inputs)
            diagnostics["duplicate_inputs"] = int(len(dataset) - unique_inputs)

            if dataset.items:
                longest_input_idx = max(
                    range(len(dataset.items)),
                    key=lambda i: len(dataset.items[i].input_ids),
                )
                longest_example = dataset.items[longest_input_idx]
                diagnostics["longest_input_index"] = int(longest_input_idx)
                diagnostics["longest_input_tokens"] = int(len(longest_example.input_ids))
                diagnostics["longest_input_chars"] = int(len(longest_example.input_text))
                diagnostics["longest_input_film"] = (
                    str(longest_example.film)
                    if getattr(longest_example, "film", None) is not None
                    else None
                )
                
            empty_predictions = diagnostics["prediction_chars"]["zeros"]
            total_predictions = diagnostics["prediction_chars"]["count"]
            if total_predictions:
                diagnostics["empty_prediction_ratio"] = float(
                    empty_predictions / total_predictions
                )
            if empty_predictions:
                LOGGER.warning(
                    "Lo split %s del task %s ha prodotto %d predizioni vuote su %d esempi (%.1f%%).",
                    split,
                    task_name,
                    empty_predictions,
                    total_predictions,
                    100.0 * empty_predictions / total_predictions if total_predictions else 0.0,
                )

            split_payload["tasks"][task_name] = {
                "path": str(path),
                "loss": float(loss),
                "num_samples": len(dataset),
                "metrics": metrics_payload,
                # attach computed diagnostics, not the pyparsing enum
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
