"""
Adapters: generano i 4 task on-the-fly a partire da Pairs per film.
Deterministici dato un seed (niente random globale).
"""
from __future__ import annotations
from typing import Iterable, Dict, List, Tuple, Iterator
from hashlib import blake2b

from data.serialization import linearize

TASK_T2RDF = "<Text2RDF>"
TASK_RDF2TEXT = "<RDF2Text>"
TASK_CONT = "<CONTINUERDF>"
TASK_MASK = "<MASK>"

def _film_seed(film: str, base_seed: int = 13) -> int:
    """Seed deterministico per film (stabile su macchine diverse)."""
    h = blake2b(digest_size=8)
    h.update(f"{base_seed}|{film}".encode("utf-8"))
    return int.from_bytes(h.digest(), "big")

def text2rdf(pairs_iter: Iterable[dict]) -> Iterator[dict]:
    for ex in pairs_iter:
        yield {
            "film": ex["film"],
            "input": ex["text"].strip() + " " + TASK_T2RDF,
            "target": linearize(ex["triples"]),
            "task": "Text2RDF",
            "id": f"{ex['film']}#T2RDF"
        }

def rdf2text(pairs_iter: Iterable[dict]) -> Iterator[dict]:
    for ex in pairs_iter:
        yield {
            "film": ex["film"],
            "input": linearize(ex["triples"]) + " " + TASK_RDF2TEXT,
            "target": ex["text"].strip(),
            "task": "RDF2Text",
            "id": f"{ex['film']}#RDF2Text"
        }

def comp1(pairs_iter: Iterable[dict], base_seed: int = 13) -> Iterator[dict]:
    """
    Maschera l'oggetto di UNA tripla per film in modo deterministico: indice = seed % len(triples).
    """
    for ex in pairs_iter:
        triples = list(ex["triples"])
        if not triples:
            continue
        idx = _film_seed(ex["film"], base_seed) % len(triples)
        s, p, o = triples[idx]
        rdf = linearize(triples)
        rdf_masked = rdf.replace(o, TASK_MASK, 1)
        yield {
            "film": ex["film"],
            "input": rdf_masked + " " + TASK_MASK,
            "target": o,
            "task": "RDFComp1",
            "id": f"{ex['film']}#Comp1"
        }

def comp2(pairs_iter: Iterable[dict], base_seed: int = 13, ratio: float = 0.6) -> Iterator[dict]:
    """
    Continuation deterministica: split point = 1 + (seed % (len-1)) oppure floor(len*ratio).
    Qui usiamo ratio fisso per stabilità.
    """
    for ex in pairs_iter:
        triples = list(ex["triples"])
        if len(triples) < 2:
            continue
        split = max(1, int(ratio * len(triples)))
        ctx = linearize(triples[:split])
        nxt = linearize(triples[split:])
        yield {
            "film": ex["film"],
            "input": ctx + " " + TASK_CONT,
            "target": nxt,
            "task": "RDFComp2",
            "id": f"{ex['film']}#Comp2"
        }
