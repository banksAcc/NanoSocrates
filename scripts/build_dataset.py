#!/usr/bin/env python
"""
Costruisce i 4 dataset a partire da:
- triple (DBpedia)  -> data/raw/dbpedia_triples.jsonl
- testi  (Wikipedia)-> data/raw/wikipedia_intro.jsonl

Passi:
1) Pairing & filtri (min_triples, testo non vuoto)
2) Split per film (train/val/test)
3) Costruzione task:
   - text2rdf.{split}.jsonl
   - rdf2text.{split}.jsonl
   - rdfcomp1.{split}.jsonl
   - rdfcomp2.{split}.jsonl
"""
from __future__ import annotations
import argparse, os, random
from typing import List, Dict, Iterable

#from ..src.utils.io import read_jsonl, write_jsonl

from utils.io import read_jsonl, write_jsonl
from utils.config import load_yaml
from utils.logging import get_logger
from data.pairing import pair_and_filter
from data.builders import build_text2rdf, build_rdf2text, build_comp1, build_comp2

logger = get_logger("build_dataset")

def split_by_film(pairs: List[dict], split_ratios=(0.8, 0.1, 0.1), seed=13):
    """
    Split deterministico per-film: shuffla la lista di film, poi taglia in (train,val,test).
    Restituisce dict: {"train":[...], "val":[...], "test":[...]}
    """
    films = [p["film"] for p in pairs]
    random.Random(seed).shuffle(films)
    n = len(films)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])
    train_ids = set(films[:n_train])
    val_ids   = set(films[n_train:n_train+n_val])
    test_ids  = set(films[n_train+n_val:])

    out = {"train": [], "val": [], "test": []}
    for ex in pairs:
        if ex["film"] in train_ids:
            out["train"].append(ex)
        elif ex["film"] in val_ids:
            out["val"].append(ex)
        else:
            out["test"].append(ex)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/data/build.yaml")
    ap.add_argument("--dbp", required=True, help="data/raw/dbpedia_triples.jsonl")
    ap.add_argument("--wiki", required=True, help="data/raw/wikipedia_intro.jsonl")
    ap.add_argument("--outdir", required=True, help="data/processed/")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("shuffle_seed", 13))
    random.seed(seed)

    # 1) Carica stream
    triples_stream = list(read_jsonl(args.dbp))
    texts_stream = list(read_jsonl(args.wiki))
    logger.info(f"Loaded triples: {len(triples_stream)} ; texts: {len(texts_stream)}")

    # 2) Pairing & filtri qualità
    pairs_iter = pair_and_filter(
        triples_stream,
        texts_stream,
        min_triples=int(cfg.get("min_triples_per_film", 3)),
    )
    pairs = list(pairs_iter)
    logger.info(f"Paired examples: {len(pairs)}")

    if not pairs:
        logger.warning("No pairs available after filtering. Check your inputs/configs.")
        return

    # 3) Split per film (deterministico)
    ratios = tuple(cfg.get("train_val_test_split", [0.8, 0.1, 0.1]))
    splits = split_by_film(pairs, ratios, seed)
    for k in ("train", "val", "test"):
        logger.info(f"{k}: {len(splits[k])} films")

    # 4) Costruzione dataset per split
    os.makedirs(args.outdir, exist_ok=True)
    def dump_task(task_name: str, records: Iterable[dict], split: str):
        path = os.path.join(args.outdir, f"{task_name}.{split}.jsonl")
        write_jsonl(path, records)
        logger.info(f"Wrote {task_name}.{split} to {path}")

    # Per ogni split, costruiamo e salviamo i quattro task
    for split, examples in splits.items():
        dump_task("text2rdf",  build_text2rdf(examples, cfg.get("max_seq_len", 384)), split)
        dump_task("rdf2text",  build_rdf2text(examples, cfg.get("max_seq_len", 384)), split)
        dump_task("rdfcomp1",  build_comp1(examples,  cfg.get("max_seq_len", 384)), split)
        dump_task("rdfcomp2",  build_comp2(examples,  cfg.get("max_seq_len", 384)), split)

    logger.info(f"All done. Processed datasets in {args.outdir}")

if __name__ == "__main__":
    main()
