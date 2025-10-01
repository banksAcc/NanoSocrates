"""
DBpedia client:
- Costruisce query SPARQL per estrarre triple 1-hop filtrate (whitelist).
- Supporta batching con LIMIT/OFFSET.
- Restituisce dict: {film, dir, p, o} con prefissi normalizzati.
"""

from __future__ import annotations
from typing import Dict, Iterable, Iterator, List, Tuple
from utils.config import load_yaml
from utils.logging import get_logger
from data.serialization import normalize_prefix

logger = get_logger("dbpedia")

# Template di query per archi uscenti: (?film ?p ?o)
Q_OUT = """PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT ?film ?p ?o WHERE {
  ?film a dbo:Film .
  VALUES ?p { %PREDICATES% }
  ?film ?p ?o .
}
LIMIT %LIMIT%
OFFSET %OFFSET%"""

# Template di query per archi entranti: (?s ?p ?film)
Q_IN = """PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT ?film ?p ?s WHERE {
  ?film a dbo:Film .
  VALUES ?p { %PREDICATES% }
  ?s ?p ?film .
}
LIMIT %LIMIT%
OFFSET %OFFSET%"""

def _fmt_predicates(preds: List[str]) -> str:
    """Converte la lista in blocco VALUES: 'dbo:director dbo:starring ...'."""
    return " ".join(preds)

def _paged_queries(template: str, predicates: List[str], batch_size: int, limit_total: int):
    """
    Genera query con LIMIT/OFFSET per coprire fino a limit_total righe.
    """
    pred_str = _fmt_predicates(predicates)
    emitted = 0
    offset = 0
    while emitted < limit_total:
        limit = min(batch_size, limit_total - emitted)
        q = (
            template
            .replace("%PREDICATES%", pred_str)
            .replace("%LIMIT%", str(limit))
            .replace("%OFFSET%", str(offset))
        )
        yielded = (limit, offset)
        yield q, yielded
        emitted += limit
        offset += limit

def fetch_triples(config_path: str) -> Iterator[Dict[str, str]]:
    """
    Esegue query SPARQL (con batching) e yielda triple 1-hop.
    Output: {film, dir, p, o}; 'dir' in {"out", "in"}.
    """
    cfg = load_yaml(config_path)
    endpoint = cfg.get("endpoint", "https://dbpedia.org/sparql")
    preds = cfg["predicates_whitelist"]
    limit_total = int(cfg.get("limit_total", 100000))
    batch_size = int(cfg.get("batch_size", 5000))
    timeout_s = int(cfg.get("timeout_s", 60))
    direction = cfg.get("direction", "out")

    # Lazy import per evitare dipendenze non necessarie a import-time.
    from SPARQLWrapper import SPARQLWrapper, JSON

    def run(template: str, dir_tag: str):
        nonlocal endpoint
        for q, (limit, offset) in _paged_queries(template, preds, batch_size, limit_total):
            logger.info(f"DBpedia {dir_tag}: LIMIT {limit} OFFSET {offset}")
            sp = SPARQLWrapper(endpoint)
            sp.setTimeout(timeout_s)
            sp.setReturnFormat(JSON)
            sp.setMethod("POST")
            sp.setQuery(q)
            res = sp.query().convert()
            bindings = res.get("results", {}).get("bindings", [])
            if not bindings:
                # Se una pagina è vuota, continuiamo: DBpedia non garantisce densità uniforme.
                continue
            for b in bindings:
                film = normalize_prefix(b["film"]["value"])
                p = normalize_prefix(b["p"]["value"])
                if dir_tag == "out":
                    o = normalize_prefix(b["o"]["value"])
                else:
                    o = normalize_prefix(b["s"]["value"])
                yield {"film": film, "dir": dir_tag, "p": p, "o": o}

    # Uscente sempre; entrante opzionale se direction == "both"
    yield from run(Q_OUT, "out")
    if direction == "both":
        yield from run(Q_IN, "in")
