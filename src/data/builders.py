"""
Costruzione dei 4 dataset (Text2RDF, RDF2Text, RDF Completion 1 & 2).
- Linearizza RDF con tag.
- Applica spanned masking semplice per Comp-1 (maschera l'intero oggetto).
- Crea contesto/continuazione per Comp-2 (split 60/40).
"""

from __future__ import annotations
from typing import Iterable, Dict, List, Tuple
import random

from data.serialization import linearize

# Token di task (prefisso in input)
TASK_T2RDF = "<Text2RDF>"
TASK_RDF2TEXT = "<RDF2Text>"
TASK_CONT = "<CONTINUERDF>"
TASK_MASK = "<MASK>"

def build_text2rdf(pairs_iter: Iterable[dict], max_len: int = 384):
    """Input: testo + <Text2RDF>; Target: RDF linearizzato."""
    for ex in pairs_iter:
        rdf = linearize(ex["triples"])
        yield {
            "film": ex["film"],
            "input": ex["text"].strip() + " " + TASK_T2RDF,
            "target": rdf,
        }

def build_rdf2text(pairs_iter: Iterable[dict], max_len: int = 384):
    """Input: RDF linearizzato + <RDF2Text>; Target: testo."""
    for ex in pairs_iter:
        rdf = linearize(ex["triples"])
        yield {
            "film": ex["film"],
            "input": rdf + " " + TASK_RDF2TEXT,
            "target": ex["text"].strip(),
        }

def build_comp1(pairs_iter: Iterable[dict], max_len: int = 384, mask_token: str = "<MASK>"):
    """
    RDF Completion 1 (masked): maschera l'intero oggetto di una tripla a caso.
    Input: RDF_masked + <MASK>; Target: l'oggetto mascherato.
    """
    for ex in pairs_iter:
        triples = list(ex["triples"])
        if not triples:
            continue
        s, p, o = random.choice(triples)
        rdf = linearize(triples)
        # Nota: replace 1 sola volta per evitare di mascherare occorrenze multiple dello stesso o.
        rdf_masked = rdf.replace(o, mask_token, 1)
        yield {
            "film": ex["film"],
            "input": rdf_masked + " " + TASK_MASK,
            "target": o,
        }

def build_comp2(pairs_iter: Iterable[dict], max_len: int = 384):
    """
    RDF Completion 2 (continuation): split triples in contesto (60%) e continuazione (40%).
    Input: RDF(contesto) + <CONTINUERDF>; Target: RDF(continuazione)
    """
    for ex in pairs_iter:
        triples = list(ex["triples"])
        if len(triples) < 2:
            continue
        split = max(1, int(0.6 * len(triples)))
        ctx = linearize(triples[:split])
        nxt = linearize(triples[split:])
        yield {
            "film": ex["film"],
            "input": ctx + " " + TASK_CONT,
            "target": nxt,
        }
