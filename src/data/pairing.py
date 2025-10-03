"""
Pairing & filtro qualità:
- Unisce triple (per film) e testo (intro) in un unico record.
- rimuove duplicati di triple, scarta film con testo mancante o poche triple.
- normalizza le triple in forma (s, p, o), sia per relazioni uscenti che entranti.
Output: {"film", "text", "triples": [(s, p, o), ...]}
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple
from utils.logging import get_logger

logger = get_logger("pairing")

def pair_and_filter(
    triples_stream: Iterable[dict],
    texts_stream: Iterable[dict],
    min_triples: int = 3,
) -> Iterator[dict]:
    triples_by_film: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for r in triples_stream:
        direction = r.get("dir", "out")
        if direction == "in":
            triple = (r["o"], r["p"], r["film"])
        else:
            triple = (r["film"], r["p"], r["o"])
        triples_by_film[r["film"]].append(triple)

    texts_by_film: Dict[str, str] = {}
    for r in texts_stream:
        if r.get("text"):
            texts_by_film[r["film"]] = r["text"].strip()

    kept, dropped_no_text, dropped_few_triples = 0, 0, 0
    for film, triples in triples_by_film.items():
        text = texts_by_film.get(film, "")
        if not text:
            dropped_no_text += 1
            continue

        # rimuove duplicati mantenendo ordine (dict.fromkeys su tuple (s, p, o))
        triples_unique = list(dict.fromkeys(triples))
        if len(triples_unique) < min_triples:
            dropped_few_triples += 1
            continue

        kept += 1
        yield {"film": film, "text": text, "triples": triples_unique}

    logger.info(f"Pairing: kept={kept}, no_text={dropped_no_text}, few_triples={dropped_few_triples}")
