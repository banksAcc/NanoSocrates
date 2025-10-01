"""
Funzioni di I/O robuste per JSONL (supporto gzip), con API semplici.
"""

from __future__ import annotations
import json, gzip, os
from typing import Iterable, Dict, Any, Iterator

def open_text(path: str, mode: str = "rt", encoding: str = "utf-8"):
    """Apre file di testo o .gz trasparente."""
    return gzip.open(path, mode, encoding=encoding) if path.endswith(".gz") else open(path, mode, encoding=encoding)

def read_jsonl(path: str) -> Iterator[dict]:
    """Legge un JSON Lines e restituisce un iteratore di dict."""
    with open_text(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, records: Iterable[Dict[str, Any]]):
    """Scrive un iterabile di dict in JSON Lines."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open_text(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
