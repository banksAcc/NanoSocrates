#!/usr/bin/env python
"""Utility script to build the canonical NanoSocrates dataset and splits."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Iterable, List

from src.data.builders import build_comp1, build_comp2, build_rdf2text, build_text2rdf
from src.data.pairing import pair_and_filter
from src.utils.config import load_yaml
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

LOGGER = get_logger("build_dataset")


def split_by_film(pairs: List[dict], split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 13):
    """Split the dataset by film id to avoid leaking a title across different splits."""
    films = [p["film"] for p in pairs]
    rng = random.Random(seed)
    rng.shuffle(films)
    n = len(films)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    train_ids = set(films[:n_train])
    val_ids = set(films[n_train : n_train + n_val])
    test_ids = set(films[n_train + n_val :])

    out = {"train": [], "val": [], "test": []}
    for example in pairs:
        film_id = example["film"]
        if film_id in train_ids:
            out["train"].append(example)
        elif film_id in val_ids:
            out["val"].append(example)
        else:
            out["test"].append(example)
    return out


def _add_n_triples(example: dict) -> dict:
    """Return a shallow copy of *example* with an explicit ``n_triples`` field."""
    enriched = dict(example)
    enriched["n_triples"] = len(enriched.get("triples", []))
    return enriched


def main() -> None:
    """Materialise the canonical dataset splits and optional task-specific files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/data/build.yaml")
    parser.add_argument("--dbp", required=True, help="data/raw/dbpedia_triples.jsonl")
    parser.add_argument("--wiki", required=True, help="data/raw/wikipedia_intro.jsonl")
    parser.add_argument("--outdir", required=True, help="data/processed/")
    parser.add_argument(
        "--emit_tasks",
        action="store_true",
        help="(opzionale) scrivi anche i 4 task per split",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("shuffle_seed", 13))
    random.seed(seed)

    dbpedia_cfg_path = cfg.get("dbpedia_config")
    predicate_priority: List[str] = []
    if dbpedia_cfg_path:
        try:
            dbpedia_cfg = load_yaml(dbpedia_cfg_path)
            predicate_priority = list(dbpedia_cfg.get("predicates_whitelist", []))
        except FileNotFoundError:
            LOGGER.warning(
                "DBpedia config not found at %s. Predicate priority will be empty.",
                dbpedia_cfg_path,
            )

    predicate_caps_override = cfg.get("predicate_caps_override") or {}
    predicate_caps_override = {
        predicate: int(cap) for predicate, cap in predicate_caps_override.items()
    }
    predicate_object_cap = cfg.get("predicate_object_cap")
    predicate_object_cap = int(predicate_object_cap) if predicate_object_cap is not None else None

    # Carica stream grezzi dalle sorgenti DBpedia/Wikipedia.
    triples_stream = list(read_jsonl(args.dbp))
    texts_stream = list(read_jsonl(args.wiki))
    LOGGER.info("Loaded triples: %d ; texts: %d", len(triples_stream), len(texts_stream))

    # Appaia triple e testi ed effettua un filtro di qualità minimo.
    pairs = list(
        pair_and_filter(
            triples_stream,
            texts_stream,
            min_triples=int(cfg.get("min_triples_per_film", 3)),
            allowed_languages=cfg.get("allowed_languages"),
            predicate_object_cap=predicate_object_cap,
            predicate_caps_override=predicate_caps_override,
            predicate_priority=predicate_priority,
        )
    )
    LOGGER.info("Paired examples (films): %d", len(pairs))

    if not pairs:
        LOGGER.warning("No pairs available after filtering. Check inputs/configs.")
        return

    # Persist canonical pairs and precomputed splits for downstream scripts.
    os.makedirs("data/interim", exist_ok=True)

    write_jsonl("data/interim/pairs.all.jsonl", (_add_n_triples(pair) for pair in pairs))
    LOGGER.info("Wrote data/interim/pairs.all.jsonl")

    ratios = tuple(cfg.get("train_val_test_split", [0.8, 0.1, 0.1]))
    splits = split_by_film(pairs, ratios, seed)
    with open("data/interim/splits.json", "w", encoding="utf-8") as fh:
        json.dump(
            {split: [example["film"] for example in items] for split, items in splits.items()},
            fh,
            ensure_ascii=False,
            indent=2,
        )
    LOGGER.info("Wrote data/interim/splits.json")

    for split_name, examples in splits.items():
        write_jsonl(f"data/interim/pairs.{split_name}.jsonl", (_add_n_triples(example) for example in examples))
        LOGGER.info("Wrote data/interim/pairs.%s.jsonl (%d films)", split_name, len(examples))

    if args.emit_tasks:
        os.makedirs(args.outdir, exist_ok=True)

        def dump_task(task_name: str, records: Iterable[dict], split: str) -> None:
            """Write a task-specific JSONL file for a given data *split*."""
            path = os.path.join(args.outdir, f"{task_name}.{split}.jsonl")
            write_jsonl(path, records)
            LOGGER.info("Wrote %s.%s to %s", task_name, split, path)

        for split_name, examples in splits.items():
            max_len = cfg.get("max_seq_len", 384)
            dump_task("text2rdf", build_text2rdf(examples, max_len), split_name)
            dump_task("rdf2text", build_rdf2text(examples, max_len), split_name)
            dump_task("rdfcomp1", build_comp1(examples, max_len), split_name)
            dump_task("rdfcomp2", build_comp2(examples, max_len), split_name)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
