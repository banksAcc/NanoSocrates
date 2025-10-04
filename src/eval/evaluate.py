"""Orchestratore di valutazione su split val/test."""
from __future__ import annotations

import os
from collections import defaultdict
from functools import partial
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import torch
from torch.utils.data import DataLoader

from src.decoding.base import decode_to_text
from src.eval.metrics import (
    compute_accuracy,
    compute_text_generation_metrics,
    compute_triple_metrics,
)
from src.model.transformer import TinySeq2Seq
from src.tokenizer.tokenizer_io import TokWrapper
from src.training.dataloaders import JsonlSeq2Seq, pad_collate

def _select_device(want: Optional[str]) -> str:
    want = (want or "cuda").lower()
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalise_text(text: str) -> str:
    if not text:
        return ""
    cleaned = [tok for tok in text.replace("\n", " ").split() if tok and tok != "<pad>"]
    return " ".join(cleaned).strip()


def _load_model_from_checkpoint(
    checkpoint_path: str,
    tokenizer: TokWrapper,
    device: str,
    overrides: Optional[Mapping[str, object]] = None,
) -> tuple[TinySeq2Seq, Dict[str, object]]:
    overrides = dict(overrides or {})
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_cfg: MutableMapping[str, object] = dict(ckpt.get("config", {}))
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
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, dict(saved_cfg)


def load_model_and_tokenizer(
    tokenizer_file: str,
    checkpoint_path: str,
    *,
    device: Optional[str] = None,
    overrides: Optional[Mapping[str, object]] = None,
) -> tuple[TinySeq2Seq, TokWrapper, str, Dict[str, object]]:
    device_sel = _select_device(device)
    tokenizer = TokWrapper(tokenizer_file)
    model, saved_cfg = _load_model_from_checkpoint(
        checkpoint_path, tokenizer, device_sel, overrides
    )
    return model, tokenizer, device_sel, saved_cfg


@torch.no_grad()
def _compute_loss(model: TinySeq2Seq, dataloader: DataLoader, device: str) -> float:
    total = 0.0
    steps = 0
    for batch in dataloader:
        inp = batch["input_ids"].to(device, non_blocking=True)
        att = batch["attention_mask"].to(device, non_blocking=True)
        lab = batch["labels"].to(device, non_blocking=True)
        out = model(inp, att, labels=lab)
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
) -> tuple[List[str], List[str], List[str]]:
    predictions: List[str] = []
    references: List[str] = []
    tasks: List[str] = []

    for ex in dataset.items:
        pred = decode_to_text(
            model,
            tokenizer,
            ex.input,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        predictions.append(_normalise_text(pred))
        references.append(_normalise_text(ex.target))
        tasks.append(ex.task or "unknown")
    return predictions, references, tasks


def _group_by_task(
    predictions: Iterable[str],
    references: Iterable[str],
    tasks: Iterable[str],
) -> Dict[str, Dict[str, List[str]]]:
    buckets: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"pred": [], "ref": []})
    for pred, ref, task in zip(predictions, references, tasks):
        buckets[task]["pred"].append(pred)
        buckets[task]["ref"].append(ref)
    return buckets


def _metrics_for_task(task: str, preds: List[str], refs: List[str]) -> Dict[str, float]:
    if task == "rdf2text":
        return compute_text_generation_metrics(preds, refs)
    if task in {"text2rdf", "rdfcomp2"}:
        return compute_triple_metrics(preds, refs)
    if task == "rdfcomp1":
        return compute_accuracy(preds, refs)
    # fallback: treat as generation
    return compute_text_generation_metrics(preds, refs)


def _normalise_tasks_config(raw_tasks) -> Dict[str, Dict[str, str]]:
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

    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = device == "cuda"

    tasks_cfg = _normalise_tasks_config(
        config.get("tasks") if config.get("tasks") is not None else config.get("datasets")
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
            dataset = JsonlSeq2Seq(str(path), tokenizer=tokenizer, max_len=max_len)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            loss = _compute_loss(model, dataloader, device)
            preds, refs, task_tags = _generate_predictions(
                model, tokenizer, dataset, device, max_new_tokens
            )
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

            split_payload["tasks"][task_name] = {
                "path": str(path),
                "loss": float(loss),
                "num_samples": len(dataset),
                "metrics": metrics_payload,
            }
            weighted_loss += loss * len(dataset)
            total_samples += len(dataset)

        if total_samples > 0:
            split_payload["avg_loss"] = float(weighted_loss / total_samples)
            split_payload["num_samples"] = total_samples
        report["splits"][split] = split_payload

    return report
