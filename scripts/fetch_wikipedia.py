#!/usr/bin/env python
"""
Per ogni film (dbr:...) presente nel file triple, scarica il primo paragrafo (summary) da Wikipedia.
Fallback opzionale a dbo:abstract@lang se abilitato in configs/data/wikipedia.yaml.
Scrive JSONL: {film, text}.
"""
import argparse
from utils.io import read_jsonl, write_jsonl
from data.wikipedia import fetch_intro_records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/data/wikipedia.yaml")
    ap.add_argument("--in", dest="inp", required=True, help="JSONL triple: data/raw/dbpedia_triples.jsonl")
    ap.add_argument("--out", required=True, help="JSONL testi: data/raw/wikipedia_intro.jsonl")
    args = ap.parse_args()

    # Set di film unici presenti nelle triple
    seen = set()
    films = (r["film"] for r in read_jsonl(args.inp))
    films_unique = (f for f in films if (f not in seen and not seen.add(f)))

    write_jsonl(args.out, fetch_intro_records(args.config, films_unique))
    print(f"[OK] Wrote wiki intros to {args.out}")

if __name__ == "__main__":
    main()
