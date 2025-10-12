"""Train the NanoSocrates byte-pair encoding tokenizer from JSONL corpora."""

import argparse
import os
import yaml
from src.tokenizer.train_bpe import train_bpe

def main() -> None:
    """Train a BPE tokenizer using the configuration specified in YAML."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer/bpe_default.yaml")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        C = yaml.safe_load(f)
    os.makedirs(os.path.dirname(C["out"]) or ".", exist_ok=True)
    train_bpe(
        glob_pat=C["glob"],
        out_path=C["out"],
        vocab_size=int(C["vocab_size"]),
        min_freq=int(C.get("min_freq", 2)),
        special_tokens=C["special_tokens"],
    )

if __name__ == "__main__":
    main()
