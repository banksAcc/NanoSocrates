#!/usr/bin/env python
"""Deduplicate DBpedia triples per film with lightweight normalisation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from src.data.serialization import normalize_prefix
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

LOGGER = get_logger("dedup_triples")


def _normalize_value(value: str | None) -> str:
    """Return a canonical representation of a triple component."""

    if value is None:
        return ""
    return normalize_prefix(value)


def _alias_key(value: str | None) -> str:
    """Return a relaxed normalisation to coalesce simple aliases."""

    canonical = _normalize_value(value)
    return canonical.replace("_", " ").casefold()


def _deduplicate(records: Iterable[dict]) -> Tuple[List[dict], int]:
    """Deduplicate triples grouped by film while keeping a stable order."""

    kept: List[dict] = []
    removed = 0

    seen_by_film: Dict[str, set[Tuple[str, str, str]]] = defaultdict(set)
    canonical_by_film: Dict[str, str] = {}

    for record in records:
        film_raw = record.get("film")
        pred_raw = record.get("p")
        obj_raw = record.get("o")

        if film_raw is None or pred_raw is None or obj_raw is None:
            LOGGER.debug("Skipping malformed record: %s", record)
            continue

        film_alias = _alias_key(film_raw)
        film_canonical = _normalize_value(film_raw)
        canonical_by_film.setdefault(film_alias, film_canonical)

        predicate_canonical = _normalize_value(pred_raw)
        object_canonical = _normalize_value(obj_raw)

        key = (
            record.get("dir", "out"),
            predicate_canonical.replace("_", " ").casefold(),
            object_canonical.replace("_", " ").casefold(),
        )

        if key in seen_by_film[film_alias]:
            removed += 1
            continue

        seen_by_film[film_alias].add(key)

        normalised_record = dict(record)
        normalised_record["film"] = canonical_by_film[film_alias]
        normalised_record["p"] = predicate_canonical
        normalised_record["o"] = object_canonical
        kept.append(normalised_record)

    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSONL file")
    parser.add_argument(
        "--out",
        dest="output_path",
        help="Optional output path. Defaults to in-place overwrite of --in.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path or input_path

    records = list(read_jsonl(input_path))
    LOGGER.info("Loaded %d triples from %s", len(records), input_path)

    deduped_records, removed = _deduplicate(records)
    LOGGER.info("Removed %d duplicate triples", removed)
    LOGGER.info("Writing %d triples to %s", len(deduped_records), output_path)

    write_jsonl(output_path, deduped_records)


if __name__ == "__main__":
    main()
