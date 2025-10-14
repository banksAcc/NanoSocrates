"""Utility script to inspect a single batch emitted by the DataLoader/DataCollator."""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, Sequence

import torch
from tokenizers import Tokenizer

from src.data.builders import build_and_cache_datasets
from src.tokenizer.tokenizer_io import ensure_runtime_special_tokens
from src.training.dataloaders import create_multitask_dataloader
from src.utils.config import apply_overrides, apply_toy_paths, load_yaml


def _decode_sequence(
    tokenizer: Tokenizer,
    sequence: Sequence[int],
    pad_id: int,
) -> str:
    """Decode *sequence* removing ``pad_id`` tokens but keeping other specials."""

    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
    filtered = [int(tok) for tok in sequence if int(tok) != pad_id]
    if not filtered:
        return ""
    return tokenizer.decode(filtered, skip_special_tokens=False)


def _contains_mask_token(text: str) -> bool:
    """Return True when the decoded text contains a mask token marker."""

    markers = {"[MASK]", "<MASK>", "<mask>", "[mask]"}
    return any(marker in text for marker in markers)


def _prepare_config(path: str, overrides: Iterable[str], toy: bool) -> Dict[str, object]:
    """Load the YAML configuration and optionally apply overrides/toy shortcuts."""

    cfg = load_yaml(path)
    if toy:
        cfg = apply_toy_paths(cfg)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def inspect_batch(
    cfg_path: str,
    *,
    split: str,
    batch_size: int,
    toy: bool,
    overrides: Iterable[str],
    limit: int,
) -> None:
    """Materialise one batch and print debugging information about it."""

    cfg = _prepare_config(cfg_path, overrides, toy)

    tokenizer_path = cfg.get("tokenizer_file") or cfg.get("data", {}).get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("Il config deve specificare 'tokenizer_file'.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ensure_runtime_special_tokens(tokenizer)

    datasets = build_and_cache_datasets(cfg, tokenizer)
    dataset = datasets["train" if split == "train" else "validation"]
    ratios = datasets.get("ratios") or dataset.task_fractions()

    loader = create_multitask_dataloader(
        dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        ratios=ratios,
        num_workers=0,
        shuffle=False,
    )

    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:  # pragma: no cover - defensive
        raise RuntimeError("Il DataLoader non ha prodotto batch.") from exc

    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Il tokenizer non contiene il token <pad> richiesto per il padding.")
    label_pad_id = pad_id

    print("=== Batch Debug Report ===")
    print(f"Split: {split} | Batch size: {batch['input_ids'].size(0)}")
    total_predict_tokens = int((batch["labels"] != label_pad_id).sum().item())
    print(f"Numero totale di token da predire (escludendo <pad>): {total_predict_tokens}")
    if total_predict_tokens == 0:
        print("⚠️  Attenzione: nessun token da predire nel batch!")

    tasks = batch.get("tasks", [])
    raw_inputs = batch.get("raw_input", [])
    raw_targets = batch.get("raw_target", [])
    films = batch.get("films", [])
    mask_positions = batch.get("mask_positions")
    mask_lengths = batch.get("mask_lengths")

    num_examples = min(batch["input_ids"].size(0), limit)
    for idx in range(num_examples):
        decoded_input = _decode_sequence(tokenizer, batch["input_ids"][idx], pad_id)
        decoded_labels = _decode_sequence(tokenizer, batch["labels"][idx], label_pad_id)
        example_predict_tokens = int((batch["labels"][idx] != label_pad_id).sum().item())

        print("\n--- Esempio", idx, "---")
        if tasks:
            print(f"Task: {tasks[idx]}")
        if films:
            print(f"Film: {films[idx]}")
        if raw_inputs:
            print(f"Input grezzo: {raw_inputs[idx]}")
        if raw_targets:
            print(f"Target grezzo: {raw_targets[idx]}")

        print(f"Input decodificato: {decoded_input}")
        print(f"Contiene token [MASK]/<MASK>: {_contains_mask_token(decoded_input)}")
        print(f"Labels decodificate: {decoded_labels}")
        print(f"Numero di token da predire (esempio): {example_predict_tokens}")

        if mask_positions is not None and mask_lengths is not None:
            positions = mask_positions[idx].tolist()
            lengths = mask_lengths[idx].tolist()
            predicted = sum(int(length) for length in lengths)
            print(f"Posizioni maschera: {positions}")
            print(f"Lunghezze maschera: {lengths} (totale={predicted})")

    if num_examples < batch["input_ids"].size(0):
        print(
            f"\n(mostrati solo i primi {num_examples} esempi su {batch['input_ids'].size(0)}: usa --limit per modificare)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ispeziona un singolo batch del DataLoader")
    parser.add_argument(
        "--config",
        default="configs/train/baseline.yaml",
        help="Config di training da cui ricavare tokenizer e dataset",
    )
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--batch-size", type=int, default=4, help="Dimensione del batch da ispezionare")
    parser.add_argument("--limit", type=int, default=4, help="Numero di esempi da stampare dal batch")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override opzionali in formato chiave=valore (come per src/run.py)",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Usa i dataset compatti in data/processed/toy definiti nel config",
    )

    args = parser.parse_args()

    inspect_batch(
        args.config,
        split=args.split,
        batch_size=args.batch_size,
        toy=args.toy,
        overrides=args.override,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
