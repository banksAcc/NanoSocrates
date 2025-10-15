"""Utility CLI per auditare dataset e task JSONL."""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.utils.io import read_jsonl
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def iter_pairs_files(pairs_arg: str) -> List[Path]:
    """Resolve `pairs.*.jsonl` files from a directory or explicit pattern."""

    path = Path(pairs_arg)
    if path.is_dir():
        pattern = str(path / "pairs.*.jsonl*")
        files = sorted(Path(p) for p in glob.glob(pattern))
    else:
        # Accept explicit file path or glob pattern.
        files = sorted(Path(p) for p in glob.glob(str(path)))
        if not files and path.exists():
            files = [path]

    LOGGER.info("Risolti %d file di coppie", len(files))
    for f in files:
        LOGGER.debug("- %s", f)
    return files


def iter_task_files(tasks_dir: str | None) -> List[Path]:
    if not tasks_dir:
        return []

    base = Path(tasks_dir)
    if not base.exists():
        LOGGER.warning("La directory task %s non esiste", tasks_dir)
        return []

    files = sorted(
        p
        for pattern in ("*.jsonl", "*.jsonl.gz")
        for p in base.glob(pattern)
        if p.is_file()
    )
    LOGGER.info("Risolti %d file di task", len(files))
    for f in files:
        LOGGER.debug("- %s", f)
    return files


def audit_pairs(files: Iterable[Path]) -> Tuple[Counter, Counter, Counter]:
    pred_counter: Counter[str] = Counter()
    obj_counter: Counter[str] = Counter()
    subj_counter: Counter[str] = Counter()
    total_triples = 0

    for path in files:
        LOGGER.info("Analisi file coppie: %s", path)
        for record in read_jsonl(str(path)):
            triples = record.get("triples") or []
            for triple in triples:
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                subj, pred, obj = triple
                subj_counter[str(subj)] += 1
                pred_counter[str(pred)] += 1
                obj_counter[str(obj)] += 1
                total_triples += 1

    LOGGER.info("Triple totali processate: %d", total_triples)
    return pred_counter, obj_counter, subj_counter


def whitespace_length(text: str) -> int:
    return len(text.split()) if text else 0


def audit_task_file(path: Path, max_len: int) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {
        "input": {"count": 0, "total": 0, "min": None, "max": 0, "exceed": 0},
        "target": {"count": 0, "total": 0, "min": None, "max": 0, "exceed": 0},
    }
    missing_fields = Counter()

    for record in read_jsonl(str(path)):
        for field in ("input", "target"):
            value = record.get(field)
            if value is None:
                missing_fields[field] += 1
                continue
            length = whitespace_length(str(value))
            field_stats = stats[field]
            field_stats["count"] += 1
            field_stats["total"] += length
            field_stats["max"] = max(field_stats["max"], length)
            if field_stats["min"] is None:
                field_stats["min"] = length
            else:
                field_stats["min"] = min(field_stats["min"], length)
            if length > max_len:
                field_stats["exceed"] += 1

    for field in stats:
        if stats[field]["min"] is None:
            stats[field]["min"] = 0

    if missing_fields:
        LOGGER.warning("%s: campi mancanti %s", path.name, dict(missing_fields))

    return stats


def format_stats(name: str, stats: Dict[str, Dict[str, float]], max_len: int) -> str:
    lines = [f"=== {name} ==="]
    for field, field_stats in stats.items():
        count = field_stats["count"]
        if count == 0:
            lines.append(f"{field}: nessun record valido")
            continue
        avg = field_stats["total"] / count if count else 0
        exceed_pct = (field_stats["exceed"] / count * 100) if count else 0.0
        lines.append(
            (
                f"{field} → esempi={count} min={int(field_stats['min'])} "
                f"avg={avg:.2f} max={int(field_stats['max'])} | "
                f"> {max_len} token: {exceed_pct:.2f}%"
            )
        )
    return "\n".join(lines)


def print_counter(title: str, counter: Counter, limit: int = 20) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for item, count in counter.most_common(limit):
        print(f"{count:>6}  {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset RDF/Text")
    parser.add_argument("--pairs", required=True, help="Directory, file o glob dei pairs.*.jsonl")
    parser.add_argument("--tasks-dir", help="Directory contenente i task JSONL")
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Soglia token per segnalare esempi troppo lunghi (default: 512)",
    )
    args = parser.parse_args()

    pair_files = iter_pairs_files(args.pairs)
    if pair_files:
        pred_counter, obj_counter, subj_counter = audit_pairs(pair_files)
        print("=== Analisi pairs ===")
        print_counter("Top 20 predicati", pred_counter)
        print_counter("Top 20 oggetti", obj_counter)
        print_counter("Top 20 soggetti", subj_counter)
    else:
        LOGGER.warning("Nessun file pairs trovato per %s", args.pairs)

    task_files = iter_task_files(args.tasks_dir)
    for task_path in task_files:
        LOGGER.info("Analisi file task: %s", task_path)
        stats = audit_task_file(task_path, args.max_len)
        name = task_path.stem
        print()
        print(format_stats(name, stats, args.max_len))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

