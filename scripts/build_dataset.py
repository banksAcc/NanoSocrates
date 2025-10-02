#!/usr/bin/env python
"""
Costruisce il dataset canonico:
- Pairs per film: {"film","text","triples", "n_triples"}
- Split per film: splits.json con liste di film per train/val/test
Opzionale: emette anche i 4 task per split (flag --emit_tasks)

Usage:
  python scripts/build_dataset.py \
    --config configs/data/build.yaml \
    --dbp data/raw/dbpedia_triples.jsonl \
    --wiki data/raw/wikipedia_intro.jsonl \
    --outdir data/processed \
    [--emit_tasks]
"""
from __future__ import annotations
import argparse, os, random, json
from typing import List, Dict, Iterable

from utils.io import read_jsonl, write_jsonl
from utils.config import load_yaml
from utils.logging import get_logger
from data.pairing import pair_and_filter

# Task builders (usati solo se --emit_tasks)
from data.builders import build_text2rdf, build_rdf2text, build_comp1, build_comp2

logger = get_logger("build_dataset")

def split_by_film(pairs: List[dict], split_ratios=(0.8, 0.1, 0.1), seed=13):
    """
    Split deterministico per-film: shuffla lista di film, taglia in (train,val,test).
    Restituisce dict: {"train":[...], "val":[...], "test":[...]}
    """
    films = [p["film"] for p in pairs]
    rng = random.Random(seed)
    rng.shuffle(films)
    n = len(films)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])
    train_ids = set(films[:n_train])
    val_ids   = set(films[n_train:n_train+n_val])
    test_ids  = set(films[n_train+n_val:])

    out = {"train": [], "val": [], "test": []}
    for ex in pairs:
        f = ex["film"]
        if f in train_ids: out["train"].append(ex)
        elif f in val_ids: out["val"].append(ex)
        else:              out["test"].append(ex)
    return out

def _add_n_triples(ex: dict) -> dict:
    ex2 = dict(ex)
    ex2["n_triples"] = len(ex2.get("triples", []))
    return ex2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/data/build.yaml")
    ap.add_argument("--dbp", required=True, help="data/raw/dbpedia_triples.jsonl")
    ap.add_argument("--wiki", required=True, help="data/raw/wikipedia_intro.jsonl")
    ap.add_argument("--outdir", required=True, help="data/processed/")
    ap.add_argument("--emit_tasks", action="store_true", help="(opzionale) scrivi anche i 4 task per split")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("shuffle_seed", 13))
    random.seed(seed)

    # Carica stream grezzi
    triples_stream = list(read_jsonl(args.dbp))
    texts_stream   = list(read_jsonl(args.wiki))
    logger.info(f"Loaded triples: {len(triples_stream)} ; texts: {len(texts_stream)}")

    # Pairing + filtro qualità
    pairs = list(pair_and_filter(
        triples_stream,
        texts_stream,
        min_triples=int(cfg.get("min_triples_per_film", 3)),
    ))
    logger.info(f"Paired examples (films): {len(pairs)}")

    if not pairs:
        logger.warning("No pairs available after filtering. Check inputs/configs.")
        return

    # Persisti la “verità” canonica (pairs all + split)
    os.makedirs("data/interim", exist_ok=True)

    write_jsonl("data/interim/pairs.all.jsonl", (_add_n_triples(p) for p in pairs))
    logger.info("Wrote data/interim/pairs.all.jsonl")

    ratios = tuple(cfg.get("train_val_test_split", [0.8, 0.1, 0.1]))
    splits = split_by_film(pairs, ratios, seed)
    with open("data/interim/splits.json", "w", encoding="utf-8") as f:
        json.dump({k: [ex["film"] for ex in v] for k, v in splits.items()}, f, ensure_ascii=False, indent=2)
    logger.info("Wrote data/interim/splits.json")

    # (comodo per debug) pairs per split
    for k, exs in splits.items():
        write_jsonl(f"data/interim/pairs.{k}.jsonl", (_add_n_triples(p) for p in exs))
        logger.info(f"Wrote data/interim/pairs.{k}.jsonl ({len(exs)} films)")

    # Opzionale: emetti anche i 4 task materializzati per split (come prima)
    if args.emit_tasks:
        os.makedirs(args.outdir, exist_ok=True)

        def dump_task(task_name: str, records: Iterable[dict], split: str):
            path = os.path.join(args.outdir, f"{task_name}.{split}.jsonl")
            write_jsonl(path, records)
            logger.info(f"Wrote {task_name}.{split} to {path}")

        for split, examples in splits.items():
            dump_task("text2rdf",  build_text2rdf(examples, cfg.get("max_seq_len", 384)), split)
            dump_task("rdf2text",  build_rdf2text(examples, cfg.get("max_seq_len", 384)), split)
            dump_task("rdfcomp1",  build_comp1(examples,  cfg.get("max_seq_len", 384)), split)
            dump_task("rdfcomp2",  build_comp2(examples,  cfg.get("max_seq_len", 384)), split)

    logger.info("Done.")

if __name__ == "__main__":
    main()
