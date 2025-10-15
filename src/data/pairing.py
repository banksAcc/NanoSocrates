"""
Pairing & filtro qualità:
- Unisce triple (per film) e testo (intro) in un unico record.
- rimuove duplicati di triple, scarta film con testo mancante o poche triple.
- normalizza le triple in forma (film, p, o) così che il film resti sempre il soggetto.
Output: {"film", "text", "triples": [(film, p, o), ...]}
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from ..utils.logging import get_logger

logger = get_logger("pairing")

def pair_and_filter(
    triples_stream: Iterable[dict],
    texts_stream: Iterable[dict],
    min_triples: int = 3,
    allowed_languages: Optional[Iterable[str]] = None,
    predicate_object_cap: Optional[int] = None,
    predicate_caps_override: Optional[Dict[str, int]] = None,
    predicate_priority: Optional[Iterable[str]] = None,
) -> Iterator[dict]:
    """Join triple and text streams keeping only well-formed film records.

    The returned triples always expose the film as the explicit subject,
    regardless of the original triple direction provided upstream. When
    ``allowed_languages`` is provided, only films with at least one
    ``dbo:language`` triple matching the allowed URIs are kept. When a
    ``predicate_object_cap`` (and optional overrides) is provided, triples are
    grouped by predicate and truncated preserving the priority order supplied
    via ``predicate_priority``.
    """
    triples_by_film: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    languages_by_film: Dict[str, Set[str]] = defaultdict(set)
    predicate_caps: Dict[str, int] = {
        key: int(value) for key, value in (predicate_caps_override or {}).items()
    }
    priority_index: Dict[str, int] = {}
    if predicate_priority is not None:
        priority_index = {predicate: idx for idx, predicate in enumerate(predicate_priority)}
    discarded_by_predicate: Dict[str, int] = defaultdict(int)
    if allowed_languages is None:
        allowed_languages_set: Optional[Set[str]] = None
    elif isinstance(allowed_languages, str):
        allowed_languages_set = {allowed_languages}
    else:
        allowed_languages_set = set(allowed_languages)
    for r in triples_stream:
        film_id = r["film"]
        triple = (film_id, r["p"], r["o"])
        # We normalise triples so that the film is always the subject. Incoming
        # edges are conceptually rewritten to keep a consistent (film, predicate,
        # object) structure for downstream consumers.
        triples_by_film[film_id].append(triple)
        if r.get("p") == "dbo:language" and r.get("o"):
            languages_by_film[film_id].add(r["o"])

    texts_by_film: Dict[str, str] = {}
    for r in texts_stream:
        if r.get("text"):
            texts_by_film[r["film"]] = r["text"].strip()

    kept, dropped_no_text, dropped_few_triples, dropped_language = 0, 0, 0, 0
    for film, triples in triples_by_film.items():
        text = texts_by_film.get(film, "")
        if not text:
            dropped_no_text += 1
            continue

        if allowed_languages_set is not None:
            languages = languages_by_film.get(film, set())
            if not languages.intersection(allowed_languages_set):
                dropped_language += 1
                continue

        # Remove duplicates but preserve the original ordering so downstream
        # tasks can still align triples with textual references.
        triples_unique = list(dict.fromkeys(triples))
        if len(triples_unique) < min_triples:
            dropped_few_triples += 1
            continue

        predicate_buckets: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        first_seen: Dict[str, int] = {}
        for idx, triple in enumerate(triples_unique):
            predicate = triple[1]
            predicate_buckets[predicate].append(triple)
            if predicate not in first_seen:
                first_seen[predicate] = idx

        capped_triples: List[Tuple[str, str, str]] = []
        for predicate in sorted(
            predicate_buckets,
            key=lambda p: (priority_index.get(p, float("inf")), first_seen[p]),
        ):
            bucket = predicate_buckets[predicate]
            limit = predicate_caps.get(
                predicate, predicate_object_cap if predicate_object_cap is not None else None
            )
            if limit is not None:
                capped = bucket[: int(limit)]
                discarded_by_predicate[predicate] += max(0, len(bucket) - len(capped))
            else:
                capped = bucket
            capped_triples.extend(capped)

        if len(capped_triples) < min_triples:
            dropped_few_triples += 1
            continue

        kept += 1
        yield {"film": film, "text": text, "triples": capped_triples}

    logger.info(
        "Pairing: kept=%d, no_text=%d, few_triples=%d, disallowed_language=%d",
        kept,
        dropped_no_text,
        dropped_few_triples,
        dropped_language,
    )
    if discarded_by_predicate:
        for predicate, count in sorted(discarded_by_predicate.items(), key=lambda item: item[0]):
            logger.info(
                "Pairing: predicate_cap predicate=%s discarded=%d", predicate, count
            )
