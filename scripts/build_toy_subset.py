#!/usr/bin/env python
"""Utility per estrarre un sottoinsieme "toy" dei dataset JSONL.

Seleziona un numero ridotto di film dai JSONL completi (per split) e
scrive i relativi file filtrati sotto ``data/processed/toy`` (o una
cartella personalizzata). L'obiettivo è avere un set consistente e
rappresentativo per debug rapido, riproducendo la struttura dei 4 task:
``text2rdf``, ``rdf2text``, ``rdfcomp1`` e ``rdfcomp2``.

Usage (default: 20 film, output in data/processed/toy):

    python -m scripts.build_toy_subset \
        --pairs data/interim/pairs.all.jsonl \
        --splits data/interim/splits.json \
        --processed-dir data/processed \
        --outdir data/processed/toy

I file generati mantengono la convenzione ``{task}.{split}.jsonl`` e
viene scritto anche un ``manifest.json`` con il riepilogo dei film
selezionati.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from utils.io import read_jsonl, write_jsonl
from utils.logging import get_logger

LOGGER = get_logger("build_toy_subset")
TASKS = ("text2rdf", "rdf2text", "rdfcomp1", "rdfcomp2")
SPLITS = ("train", "val", "test")


def _load_pairs(path: Path) -> Dict[str, dict]:
    """Indicizza le coppie film→payload (con n_triples) dal JSONL."""
    pairs: Dict[str, dict] = {}
    for record in read_jsonl(str(path)):
        film = record.get("film")
        if not film:
            continue
        pairs[film] = record
    return pairs


def _load_splits(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    # normalizza: assicura che gli split richiesti esistano anche se vuoti
    return {split: list(payload.get(split, [])) for split in SPLITS}


def _evenly_pick(entries: Sequence[dict], k: int) -> List[str]:
    """Seleziona ``k`` film distribuiti lungo l'intervallo di n_triples."""
    if k <= 0:
        return []
    ordered = sorted(entries, key=lambda ex: (ex.get("n_triples", 0), ex.get("film", "")))
    total = len(ordered)
    if total == 0:
        return []
    if total <= k:
        return [ex["film"] for ex in ordered]
    if k == 1:
        return [ordered[total // 2]["film"]]

    desired_positions = [i * (total - 1) / (k - 1) for i in range(k)]
    used: Set[int] = set()
    selected: List[str] = []

    for pos in desired_positions:
        idx = int(round(pos))
        if idx >= total:
            idx = total - 1
        if idx in used:
            chosen = None
            offset = 1
            while chosen is None and (idx - offset >= 0 or idx + offset < total):
                lower = idx - offset
                upper = idx + offset
                if lower >= 0 and lower not in used:
                    chosen = lower
                    break
                if upper < total and upper not in used:
                    chosen = upper
                    break
                offset += 1
            if chosen is None:
                for fallback in range(total):
                    if fallback not in used:
                        chosen = fallback
                        break
            idx = chosen if chosen is not None else idx
        used.add(idx)
        selected.append(ordered[idx]["film"])

    return selected


def _compute_split_allocation(split_sizes: Dict[str, int], total_needed: int) -> Dict[str, int]:
    total_films = sum(split_sizes.values())
    if total_films == 0:
        raise ValueError("Nessun film disponibile nei JSONL sorgente")
    total_needed = min(total_needed, total_films)

    allocations: Dict[str, int] = {split: 0 for split in SPLITS}
    remainders: List[tuple[float, str]] = []
    positive_splits = sum(1 for size in split_sizes.values() if size > 0)

    assigned = 0
    for split, size in split_sizes.items():
        if size <= 0:
            continue
        exact = (total_needed * size) / total_films
        base_floor = int(math.floor(exact))
        if total_needed >= positive_splits:
            base = max(1, base_floor)
        else:
            base = base_floor
        allocations[split] = min(size, base)
        assigned += allocations[split]
        remainders.append((exact - allocations[split], split))

    # Aggiusta (in eccesso o difetto) usando i resti
    diff = total_needed - assigned
    if diff > 0:
        for _, split in sorted(remainders, reverse=True):
            if diff <= 0:
                break
            if split_sizes[split] > allocations[split]:
                allocations[split] += 1
                diff -= 1
    elif diff < 0:
        for _, split in sorted(remainders):
            if diff >= 0:
                break
            if allocations[split] > 0:
                allocations[split] -= 1
                diff += 1

    allocations = _rebalance_for_minimums(allocations, split_sizes, total_needed)

    # Garantisci che non si superi il numero di film disponibili nello split
    for split, alloc in allocations.items():
        allocations[split] = min(split_sizes.get(split, 0), alloc)
    return allocations


def _rebalance_for_minimums(
    allocations: Dict[str, int], split_sizes: Dict[str, int], total_needed: int
) -> Dict[str, int]:
    desired_min = {
        "val": min(4, split_sizes.get("val", 0)),
        "test": min(4, split_sizes.get("test", 0)),
    }
    train_min = 1 if split_sizes.get("train", 0) > 0 else 0
    min_total = train_min + sum(desired_min.values())
    if total_needed < min_total:
        return allocations

    if allocations.get("train", 0) < train_min:
        allocations["train"] = train_min

    for split, minimum in desired_min.items():
        if minimum <= 0:
            continue
        current = allocations.get(split, 0)
        if current >= minimum:
            continue
        deficit = minimum - current
        gained = 0
        for candidate in SPLITS:
            if candidate == split:
                continue
            target_min = train_min if candidate == "train" else 0
            available = allocations.get(candidate, 0) - target_min
            if available <= 0:
                continue
            take = min(available, deficit)
            if take <= 0:
                continue
            allocations[candidate] = allocations.get(candidate, 0) - take
            deficit -= take
            gained += take
            if deficit <= 0:
                break
        allocations[split] = current + gained
    return allocations


def select_films(pairs_path: Path, splits_path: Path, total_films: int) -> Dict[str, List[str]]:
    pairs = _load_pairs(pairs_path)
    splits = _load_splits(splits_path)
    split_sizes = {split: sum(1 for film in films if film in pairs) for split, films in splits.items()}
    allocations = _compute_split_allocation(split_sizes, total_films)

    chosen: Dict[str, List[str]] = {split: [] for split in SPLITS}
    for split, films in splits.items():
        available_entries = [pairs[f] for f in films if f in pairs]
        needed = allocations.get(split, 0)
        if needed <= 0:
            continue
        picked = _evenly_pick(available_entries, needed)
        chosen[split] = picked
        LOGGER.info("Split %s → %d film selezionati", split, len(picked))
    return chosen


def _filter_records(source_path: Path, dest_path: Path, keep_films: Set[str]) -> int:
    if not source_path.exists():
        LOGGER.warning("File sorgente mancante: %s", source_path)
        return 0
    records = [rec for rec in read_jsonl(str(source_path)) if rec.get("film") in keep_films]
    write_jsonl(str(dest_path), records)
    return len(records)


def dump_toy_dataset(
    processed_dir: Path,
    out_dir: Path,
    selected: Dict[str, List[str]],
    tasks: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, Dict[str, int]] = {}
    for task in tasks:
        counts[task] = {}
        for split in SPLITS:
            keep = set(selected.get(split, []))
            src = processed_dir / f"{task}.{split}.jsonl"
            dst = out_dir / f"{task}.{split}.jsonl"
            kept = _filter_records(src, dst, keep)
            counts[task][split] = kept
            LOGGER.info("%s.%s → %d esempi", task, split, kept)
    return counts


def write_manifest(out_dir: Path, selected: Dict[str, List[str]], counts: Dict[str, Dict[str, int]], source_root: Path):
    payload = {
        "total_films": sum(len(v) for v in selected.values()),
        "per_split": {split: sorted(films) for split, films in selected.items()},
        "tasks": counts,
        "source_processed_dir": str(source_root),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Manifest salvato in %s", manifest_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Costruisci il toy set a 20 film")
    ap.add_argument("--pairs", default="data/interim/pairs.all.jsonl", help="JSONL completo con text+triples per film")
    ap.add_argument("--splits", default="data/interim/splits.json", help="Split originale (JSON)")
    ap.add_argument("--processed-dir", default="data/processed", help="Cartella con i JSONL completi per task")
    ap.add_argument("--outdir", default="data/processed/toy", help="Cartella di output per il toy set")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), help="Lista di task da includere")
    ap.add_argument("--films", type=int, default=20, help="Numero di film totali da selezionare")
    return ap.parse_args()


def main():
    args = parse_args()
    pairs_path = Path(args.pairs)
    splits_path = Path(args.splits)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.outdir)

    selected = select_films(pairs_path, splits_path, args.films)
    counts = dump_toy_dataset(processed_dir, out_dir, selected, args.tasks)
    write_manifest(out_dir, selected, counts, processed_dir)
    LOGGER.info("Done. Toy dataset disponibile in %s", out_dir)


if __name__ == "__main__":
    main()
