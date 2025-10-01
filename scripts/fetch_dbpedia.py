#!/usr/bin/env python
"""
Estrae triple 1-hop da DBpedia con whitelist di predicati.
Scrive JSONL: {film, dir, p, o}.
"""
import argparse
from utils.io import write_jsonl
from data.dbpedia import fetch_triples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/data/dbpedia.yaml")
    ap.add_argument("--out", required=True, help="path JSONL per output triple")
    args = ap.parse_args()

    write_jsonl(args.out, fetch_triples(args.config))
    print(f"[OK] Wrote triples to {args.out}")

if __name__ == "__main__":
    main()
