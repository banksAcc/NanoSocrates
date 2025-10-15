#!/usr/bin/env python
"""Utility to split dataset pairs by film id."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence, cast

from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

LOGGER = get_logger("split_by_film")

SPLIT_NAMES = ("train", "val", "test")


def split_by_film(
    pairs: Sequence[Mapping[str, Any]],
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 13,
) -> dict[str, list[dict[str, Any]]]:
    """Split the dataset by film id to avoid leaking a title across splits."""
    if len(split_ratios) != len(SPLIT_NAMES):
        raise ValueError(
            f"Expected {len(SPLIT_NAMES)} ratios for {SPLIT_NAMES}, received {split_ratios}."
        )

    ratios_sum = sum(split_ratios)
    if not math.isclose(ratios_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        LOGGER.warning(
            "The split ratios sum to %.6f instead of 1.0. They will be applied as-is.",
            ratios_sum,
        )

    films = [p["film"] for p in pairs]
    rng = random.Random(seed)
    rng.shuffle(films)

    n = len(films)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train_ids = set(films[:n_train])
    val_ids = set(films[n_train : n_train + n_val])
    test_ids = set(films[n_train + n_val :])

    _ensure_disjoint_film_sets(train_ids, val_ids, test_ids)

    out: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for example in pairs:
        film_id = example.get("film")
        if film_id is None:
            raise KeyError("Missing 'film' key in example; cannot build film-based splits.")

        if film_id in train_ids:
            out["train"].append(dict(example))
        elif film_id in val_ids:
            out["val"].append(dict(example))
        else:
            out["test"].append(dict(example))

    _ensure_assignments_disjoint(out)
    return out


def _ensure_disjoint_film_sets(
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    test_ids: Iterable[str],
) -> None:
    """Validate that the candidate film id sets do not overlap."""
    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    overlaps = defaultdict(set)
    for film_id in train_ids & val_ids:
        overlaps[film_id].update({"train", "val"})
    for film_id in train_ids & test_ids:
        overlaps[film_id].update({"train", "test"})
    for film_id in val_ids & test_ids:
        overlaps[film_id].update({"val", "test"})

    if overlaps:
        formatted = ", ".join(
            f"{film} ({', '.join(sorted(splits))})" for film, splits in sorted(overlaps.items())
        )
        raise ValueError(
            f"Film assegnati a più split durante il calcolo delle partizioni: {formatted}"
        )


def _ensure_assignments_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Validate that no film appears in more than one emitted split."""
    film_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, examples in splits.items():
        for example in examples:
            film_id = example.get("film")
            if film_id is None:
                continue
            film_to_splits[str(film_id)].add(split_name)

    overlaps = {film: sorted(names) for film, names in film_to_splits.items() if len(names) > 1}
    if overlaps:
        formatted = ", ".join(
            f"{film} ({', '.join(names)})" for film, names in sorted(overlaps.items())
        )
        raise ValueError(
            f"Film duplicati in più split dopo la partizione: {formatted}"
        )


def _parse_ratios(value: str) -> tuple[float, float, float]:
    try:
        ratios = tuple(float(part) for part in value.split(","))
    except ValueError as exc:  # pragma: no cover - defensive path
        raise argparse.ArgumentTypeError("--ratios richiede tre valori float separati da virgole") from exc

    if len(ratios) != len(SPLIT_NAMES):
        raise argparse.ArgumentTypeError(
            f"--ratios richiede {len(SPLIT_NAMES)} valori, ricevuto {len(ratios)}"
        )
    return cast(tuple[float, float, float], ratios)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suddivide pairs JSONL per film id.")
    parser.add_argument("--pairs", required=True, help="Percorso a pairs.all.jsonl")
    parser.add_argument("--outdir", required=True, help="Directory di destinazione per gli split")
    parser.add_argument(
        "--ratios",
        type=_parse_ratios,
        default=(0.8, 0.1, 0.1),
        help="Rapporti train,val,test separati da virgole (default: 0.8,0.1,0.1)",
    )
    parser.add_argument("--seed", type=int, default=13, help="Seed per lo shuffle")
    args = parser.parse_args()

    LOGGER.info("Carico pairs da %s", args.pairs)
    pairs = list(read_jsonl(args.pairs))
    LOGGER.info("Esempi caricati: %d", len(pairs))

    if not pairs:
        raise ValueError("Il file di input non contiene esempi da suddividere.")

    splits = split_by_film(pairs, args.ratios, args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    for split_name in SPLIT_NAMES:
        output_path = os.path.join(args.outdir, f"pairs.{split_name}.jsonl")
        LOGGER.info("Scrivo %s (%d esempi)", output_path, len(splits[split_name]))
        write_jsonl(splits[split_name], output_path)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
