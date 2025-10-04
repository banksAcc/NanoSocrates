from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from training.dataloaders import JsonlSeq2Seq, _infer_task, pad_collate
from tokenizer.tokenizer_io import TokWrapper


def test_infer_task_rdfcomp1_from_mask_marker():
    assert _infer_task("rdfcomp1.train.jsonl", "<MASK> some prompt") == "rdfcomp1"


def test_jsonlseq2seq_entity_spans_present():
    tok = TokWrapper(str(PROJECT_ROOT / "data" / "vocab" / "bpe.json"))
    dataset = JsonlSeq2Seq(
        str(PROJECT_ROOT / "data" / "processed" / "rdfcomp1.train.jsonl"),
        tokenizer=tok,
        max_len=64,
        enable_entity_spans=True,
    )
    item = dataset[0]
    # the first token should be <SOT> for rdfcomp1 examples
    assert item["labels"][0].item() == tok.token_to_id("<SOT>")
    assert item["mask_positions"].numel() == 1
    assert item["mask_lengths"].numel() == 1
    pos = item["mask_positions"][0].item()
    span_len = item["mask_lengths"][0].item()
    assert pos == 0
    assert span_len > 0
    expected_ids = tok.encode(dataset.items[0].target.strip())
    assert item["labels"][1 : 1 + span_len].tolist() == expected_ids[:span_len]


def test_pad_collate_adds_mask_tensors():
    tok = TokWrapper(str(PROJECT_ROOT / "data" / "vocab" / "bpe.json"))
    dataset = JsonlSeq2Seq(
        str(PROJECT_ROOT / "data" / "processed" / "rdfcomp1.train.jsonl"),
        tokenizer=tok,
        max_len=64,
        enable_entity_spans=True,
    )
    batch = [dataset[0], dataset[1]]
    collated = pad_collate(batch, pad_id=tok.pad_id)
    assert "mask_positions" in collated
    assert "mask_lengths" in collated
    assert collated["mask_positions"].dtype == torch.long
    assert collated["mask_lengths"].dtype == torch.long
    assert collated["mask_positions"].shape[0] == len(batch)
    assert collated["mask_positions"].shape == collated["mask_lengths"].shape
