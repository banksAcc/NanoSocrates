"""Tools to train a byte-pair encoding tokenizer on NanoSocrates datasets."""

from __future__ import annotations

import glob
import json
from typing import Iterable, Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def _iter_text(files: Iterable[str]) -> Iterator[str]:
    """Yield every input and target string stored in the provided JSONL files."""
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                # We feed both the input and the target text to maximise the
                # coverage of the resulting vocabulary.
                yield record.get("input", "")
                yield record.get("target", "")


def train_bpe(
    glob_pat: str,
    out_path: str,
    vocab_size: int,
    min_freq: int,
    special_tokens,
) -> None:
    """Train a BPE tokenizer on JSONL files matching *glob_pat*."""
    files = glob.glob(glob_pat)
    if not files:
        raise FileNotFoundError(f"Nessun file trovato per il pattern: {glob_pat}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["<unk>", "<pad>"] + list(special_tokens),
        show_progress=True,
    )

    tokenizer.train_from_iterator(_iter_text(files), trainer=trainer)

    # Configure padding at decode time so sequences shorter than the batch size
    # receive explicit pad tokens instead of implicit zeros.
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[("<pad>", tokenizer.token_to_id("<pad>"))],
    )
    tokenizer.save(out_path)

    print(f"[tokenizer] saved -> {out_path}")
