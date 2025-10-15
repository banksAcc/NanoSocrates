"""Inspect a masked language modelling batch before feeding it to the model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from src.training.mlm_datamodule import MLMDataModule
from src.utils.special_tokens import REQUIRED_SPECIAL_TOKENS, ensure_required_special_tokens


def _load_texts(paths: Iterable[Path]) -> List[str]:
    texts: List[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def _load_toy_texts(
    path: Path,
    *,
    field: str,
    sample_size: Optional[int],
    seed: Optional[int],
) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Il file del toy set '{path}' non esiste. Costruiscilo con scripts.build_toy_subset."
        )

    examples: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                examples.append(value.strip())

    if not examples:
        raise RuntimeError(
            f"Il file '{path}' non contiene campi testuali validi nella colonna '{field}'."
        )

    if sample_size is not None:
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:sample_size]

    return examples
  
def _load_tokenizer(identifier: str) -> PreTrainedTokenizerBase:
    path = Path(identifier)
    if path.exists():
        if path.is_dir():
            tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        else:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(path),
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<MASK>",
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(identifier, use_fast=True)

    ensure_required_special_tokens(tokenizer)

    if tokenizer.pad_token_id is None:
        raise ValueError("Il tokenizer deve definire un token di padding (<pad>).")

    return tokenizer


def _check_special_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
    ids = {}
    for token in REQUIRED_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            raise RuntimeError(f"Token speciale mancante dal vocabolario: {token}")
        if token_id in ids.values():
            clash = [name for name, value in ids.items() if value == token_id][0]
            raise RuntimeError(
                f"Token speciali con ID duplicato: {token} e {clash} condividono l'id {token_id}"
            )
        ids[token] = token_id


def _check_padding_labels(batch: dict[str, torch.Tensor], pad_id: int) -> None:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    padding_positions = input_ids == pad_id
    if padding_positions.any():
        labels_on_padding = labels.masked_select(padding_positions)
        if not torch.all(labels_on_padding == -100):
            raise RuntimeError(
                "Le labels nei token di padding devono essere impostate a -100."
            )


def _check_attention_mask(batch: dict[str, torch.Tensor], pad_id: int) -> None:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise RuntimeError("La attention_mask deve contenere solo valori 0 o 1.")

    padding_positions = input_ids == pad_id
    if padding_positions.any():
        if not torch.all(attention_mask.masked_select(padding_positions) == 0):
            raise RuntimeError("La attention_mask deve essere 0 sui token di padding.")

    real_token_positions = ~padding_positions
    if real_token_positions.any():
        if not torch.all(attention_mask.masked_select(real_token_positions) == 1):
            raise RuntimeError("La attention_mask deve essere 1 sui token reali.")


def _format_tensor(tensor: torch.Tensor) -> str:
    return str(tensor.cpu().tolist())


def inspect_batch(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer(args.tokenizer)
    texts: List[str] = list(args.text)
    if args.text_file:
        text_paths = [Path(path) for path in args.text_file]
        texts.extend(_load_texts(text_paths))

    if args.toy:
        toy_path = Path(args.toy_path)
        toy_texts = _load_toy_texts(
            toy_path,
            field=args.toy_field,
            sample_size=args.toy_sample,
            seed=args.toy_seed,
        )
        texts.extend(toy_texts)

    if not texts:
        raise ValueError(
            "Fornisci almeno un testo tramite --text, --text-file oppure abilita --toy."
        )

    datamodule = MLMDataModule(
        tokenizer,
        train_texts=texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
        shuffle=False,
    )
    datamodule.setup()

    loader = datamodule.train_dataloader()
    batch = next(iter(loader))

    print("=== Batch MLM Debug ===")
    print(f"Token speciali richiesti: {REQUIRED_SPECIAL_TOKENS}")
    _check_special_tokens(tokenizer)
    print("✅ Token speciali con ID univoci.")

    pad_id = tokenizer.pad_token_id
    _check_padding_labels(batch, pad_id)
    print("✅ Labels sui token di padding impostate a -100.")

    _check_attention_mask(batch, pad_id)
    print("✅ attention_mask valida (0 sul padding, 1 sui token reali).")

    print("\ninput_ids:\n", _format_tensor(batch["input_ids"]))
    print("\nlabels:\n", _format_tensor(batch["labels"]))
    print("\nattention_mask:\n", _format_tensor(batch["attention_mask"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ispeziona e valida un batch per il training masked language modelling.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Nome o percorso del tokenizer da utilizzare.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Testo da includere nel dataset (può essere passato più volte).",
    )
    parser.add_argument(
        "--text-file",
        action="append",
        default=[],
        help="File con un esempio per riga da usare come dataset.",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help=(
            "Aggiunge automaticamente testi dal toy set JSONL (utile per debug rapido)."
        ),
    )
    parser.add_argument(
        "--toy-path",
        default="data/processed/toy/rdf2text.train.jsonl",
        help="Percorso del file JSONL da cui pescare esempi toy.",
    )
    parser.add_argument(
        "--toy-field",
        default="target",
        help="Campo del JSONL da usare come testo (es. target, input, text).",
    )
    parser.add_argument(
        "--toy-sample",
        type=int,
        default=None,
        help="Numero massimo di esempi toy da caricare (default: tutti).",
    )
    parser.add_argument(
        "--toy-seed",
        type=int,
        default=13,
        help="Seed per il campionamento casuale dal toy set.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Dimensione del batch da estrarre.")
    parser.add_argument("--max-length", type=int, default=128, help="Massima lunghezza di tokenizzazione.")
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="Probabilità di masking passata al DataCollatorForLanguageModeling.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    inspect_batch(args)


if __name__ == "__main__":
    main()
