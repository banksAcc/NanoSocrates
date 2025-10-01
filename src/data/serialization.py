"""
Serializzazione RDF (linearizzazione) e parser inverso.
- Linearizziamo triple (S,P,O) in una sequenza con tag speciali.
- Parser inverso tollerante per riportare a (S,P,O).
"""

from __future__ import annotations
from typing import List, Tuple

# Token speciali richiesti dalla traccia
SOT, EOT = "<SOT>", "<EOT>"
SUBJ, PRED, OBJ = "<SUBJ>", "<PRED>", "<OBJ>"

def normalize_prefix(x: str) -> str:
    """
    Converte URI DBpedia in prefissi compatti dbr:/dbo: quando possibile.
    Altri valori (letterali, altri namespace) restano intatti.
    """
    if x.startswith("http://dbpedia.org/resource/"):
        return "dbr:" + x.split("/")[-1]
    if x.startswith("http://dbpedia.org/ontology/"):
        return "dbo:" + x.split("/")[-1]
    return x

def linearize(triples: List[Tuple[str, str, str]]) -> str:
    """
    Linearizza le triple con ordine deterministico per stabilità del training.
    Formato: <SOT> <SUBJ> s <PRED> p <OBJ> o <EOT> ...
    """
    triples_sorted = sorted(triples, key=lambda t: (t[1], t[2]))
    parts: List[str] = []
    for s, p, o in triples_sorted:
        parts.extend([SOT, SUBJ, s, PRED, p, OBJ, o, EOT])
    return " ".join(parts)

def parse_rdf(serialized: str) -> List[Tuple[str, str, str]]:
    """
    Parser inverso tollerante: estrae (s,p,o) dai tag.
    Se trova frammenti incompleti, li salta.
    """
    toks = serialized.strip().split()
    out: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(toks):
        if toks[i] == SOT:
            try:
                # Atteso: <SOT> <SUBJ> s <PRED> p <OBJ> o <EOT>
                s = toks[i + 2]
                p = toks[i + 4]
                o = toks[i + 6]
                # Avanza fino a <EOT>
                j = i + 7
                while j < len(toks) and toks[j] != EOT:
                    j += 1
                out.append((s, p, o))
                i = j + 1
            except Exception:
                i += 1
        else:
            i += 1
    return out
