"""
Costruzione dei 4 dataset (Text2RDF, RDF2Text, RDF Completion 1 & 2).
- Linearizza RDF con tag.
- Applica spanned masking semplice per Comp-1 (maschera l'intero oggetto).
- Crea contesto/continuazione per Comp-2 (split 60/40).
"""

from __future__ import annotations
from typing import Iterable, Dict, List, Tuple, Mapping, Any
import random

from .serialization import linearize
from src.training.dataloaders import JsonlSeq2Seq, MultiTaskDataset

# Token di task (prefisso in input)
TASK_T2RDF = "<Text2RDF>"
TASK_RDF2TEXT = "<RDF2Text>"
TASK_CONT = "<CONTINUERDF>"
TASK_MASK = "<MASK>"

def build_text2rdf(pairs_iter: Iterable[dict], max_len: int = 384):
    """Input: testo + <Text2RDF>; Target: RDF linearizzato."""
    for ex in pairs_iter:
        rdf = linearize(ex["triples"])
        yield {
            "film": ex["film"],
            "input": ex["text"].strip() + " " + TASK_T2RDF,
            "target": rdf,
        }

def build_rdf2text(pairs_iter: Iterable[dict], max_len: int = 384):
    """Input: RDF linearizzato + <RDF2Text>; Target: testo."""
    for ex in pairs_iter:
        rdf = linearize(ex["triples"])
        yield {
            "film": ex["film"],
            "input": rdf + " " + TASK_RDF2TEXT,
            "target": ex["text"].strip(),
        }

def build_comp1(pairs_iter: Iterable[dict], max_len: int = 384, mask_token: str = "<MASK>"):
    """
    RDF Completion 1 (masked): maschera l'intero oggetto di una tripla a caso.
    Input: RDF_masked + <MASK>; Target: l'oggetto mascherato.
    """
    for ex in pairs_iter:
        triples = list(ex["triples"])
        if not triples:
            continue
        s, p, o = random.choice(triples)
        rdf = linearize(triples)
        # Nota: replace 1 sola volta per evitare di mascherare occorrenze multiple dello stesso o.
        rdf_masked = rdf.replace(o, mask_token, 1)
        yield {
            "film": ex["film"],
            "input": rdf_masked + " " + TASK_MASK,
            "target": o,
        }

def build_comp2(pairs_iter: Iterable[dict], max_len: int = 384):
    """
    RDF Completion 2 (continuation): split triples in contesto (60%) e continuazione (40%).
    Input: RDF(contesto) + <CONTINUERDF>; Target: RDF(continuazione)
    """
    for ex in pairs_iter:
        triples = list(ex["triples"])
        if len(triples) < 2:
            continue
        split = max(1, int(0.6 * len(triples)))
        ctx = linearize(triples[:split])
        nxt = linearize(triples[split:])
        yield {
            "film": ex["film"],
            "input": ctx + " " + TASK_CONT,
            "target": nxt,
        }


def _select_task_name(dataset: JsonlSeq2Seq, fallback: str) -> str:
    """Returns the canonical task name for a dataset instance."""

    if dataset.items:
        return dataset.items[0].task
    return fallback


def build_and_cache_datasets(
    config: Mapping[str, Any],
    tokenizer: Any,
) -> Dict[str, MultiTaskDataset | Dict[str, float]]:
    """Materialises training/validation datasets based on a YAML configuration."""

    max_len = int(config.get("max_len", 256))
    dataset_specs = config.get("datasets")

    train_items: List[Any] = []
    val_items: List[Any] = []
    ratios: Dict[str, float] = {}

    if dataset_specs:
        for entry in dataset_specs:
            if not isinstance(entry, Mapping):
                raise ValueError("Ogni dataset deve essere un mapping con path 'train'/'val'.")
            train_path = entry.get("train")
            val_path = entry.get("val") or entry.get("validation")
            if not train_path or not val_path:
                raise ValueError("Specifica i path 'train' e 'val' per ogni dataset.")
            task_hint = entry.get("name") or entry.get("task")
            weight = float(entry.get("weight", 1.0))

            train_ds = JsonlSeq2Seq(str(train_path), tokenizer, max_len=max_len, task=task_hint)
            val_ds = JsonlSeq2Seq(str(val_path), tokenizer, max_len=max_len, task=task_hint)

            task_name = _select_task_name(train_ds, str(task_hint or train_path))
            ratios[task_name] = weight

            train_items.extend(train_ds.items)
            val_items.extend(val_ds.items)
    else:
        train_path = config.get("train_file") or config.get("train_path")
        val_path = (
            config.get("val_file")
            or config.get("validation_file")
            or config.get("val_path")
        )
        if not train_path or not val_path:
            raise ValueError(
                "Config di training privo di 'train_file'/'val_file' o sezione 'datasets'."
            )
        task_hint = config.get("task") or config.get("name")
        weight = float(config.get("weight", 1.0))

        train_ds = JsonlSeq2Seq(str(train_path), tokenizer, max_len=max_len, task=task_hint)
        val_ds = JsonlSeq2Seq(str(val_path), tokenizer, max_len=max_len, task=task_hint)

        task_name = _select_task_name(train_ds, str(task_hint or train_path))
        ratios[task_name] = weight

        train_items.extend(train_ds.items)
        val_items.extend(val_ds.items)

    if not train_items:
        raise ValueError("Nessun esempio disponibile nel dataset di training.")

    train_dataset = MultiTaskDataset(train_items)
    val_dataset = MultiTaskDataset(val_items or train_items)

    return {
        "train": train_dataset,
        "validation": val_dataset,
        "ratios": ratios,
    }
