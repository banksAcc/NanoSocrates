"""
Wikipedia client:
- Per ogni risorsa DBpedia (dbr:Title) prova a scaricare il "page summary" REST (primo paragrafo).
- In caso di fallimento, opzionale fallback all'abstract DBpedia @lang.
Output: {film, text}
"""

from __future__ import annotations
import time, urllib.parse
from typing import Dict, Iterator, Iterable, Optional
import requests

from utils.config import load_yaml
from utils.logging import get_logger

logger = get_logger("wikipedia")

def iri_to_title(iri: str) -> str:
    """
    Converte 'http://dbpedia.org/resource/Inception' in 'Inception'
    (decodifica URL e sostituisce underscore con spazi).
    """
    iri = iri[:-1] if iri.endswith("/") else iri
    title = iri.split("/")[-1]
    return urllib.parse.unquote(title).replace("_", " ")

def _wiki_summary(session: requests.Session, lang: str, title: str, timeout: int) -> Optional[str]:
    """Chiama la REST summary API e restituisce l'estratto se presente."""
    base = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
    url = base + urllib.parse.quote(title)
    r = session.get(url, timeout=timeout, headers={"accept": "application/json"})
    if r.status_code == 200:
        js = r.json()
        text = js.get("extract") or ""
        return text.strip() or None
    return None

def _dbpedia_abstract(endpoint: str, film_iri: str, lang: str, timeout: int) -> Optional[str]:
    """
    Fallback: recupera dbo:abstract@lang via SPARQL.
    """
    q = f"""PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?abs WHERE {{
  <{film_iri}> dbo:abstract ?abs .
  FILTER (langMatches(lang(?abs), "{lang}"))
}}
LIMIT 1"""
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
        sp = SPARQLWrapper(endpoint)
        sp.setReturnFormat(JSON)
        sp.setTimeout(timeout)
        sp.setMethod("POST")
        sp.setQuery(q)
        res = sp.query().convert()
        bindings = res.get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["abs"]["value"].strip()
    except Exception:
        return None
    return None

def fetch_intro_records(config_path: str, films_iter: Iterable[str]) -> Iterator[Dict[str, str]]:
    """
    Per ogni film IRI (dbr:...), prova summary; se mancante e abilitato, fallback abstract.
    """
    cfg = load_yaml(config_path)
    lang = cfg.get("lang", "en")
    timeout = int(cfg.get("timeout_s", 30))
    max_retries = int(cfg.get("max_retries", 2))
    enable_fallback = bool(cfg.get("enable_dbpedia_abstract_fallback", True))
    dbp_endpoint = cfg.get("dbpedia_endpoint", "https://dbpedia.org/sparql")

    session = requests.Session()

    for film_iri in films_iter:
        title = iri_to_title(film_iri)
        text: Optional[str] = None

        # REST summary con piccoli retry.
        for attempt in range(max_retries + 1):
            try:
                text = _wiki_summary(session, lang, title, timeout)
                if text:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5 * attempt)

        # Fallback opzionale su DBpedia abstract.
        if not text and enable_fallback:
            text = _dbpedia_abstract(dbp_endpoint, film_iri.replace("dbr:", "http://dbpedia.org/resource/"), lang, timeout)

        if text:
            yield {"film": film_iri, "text": text}
        # Se anche fallback fallisce, silenziosamente salta: il pairing scarterà i film senza testo.
