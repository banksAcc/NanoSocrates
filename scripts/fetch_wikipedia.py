#!/usr/bin/env python
"""
fetch_wikipedia.py — Simple & Fast
- Solo Wikipedia REST summary (no fallback).
- User-Agent esplicito (richiesto da Wikimedia).
- ThreadPoolExecutor, progress bar, cache JSONL.
- Logging chiaro e file dei fallimenti.

Usage:
  python scripts/fetch_wikipedia.py \
    --config configs/data/wikipedia.yaml \
    --in data/raw/dbpedia_triples.jsonl \
    --out data/raw/wikipedia_intro.jsonl \
    [--max 2000] [--workers 8] [--cache data/interim/wiki_cache.jsonl] \
    [--logfails data/interim/wiki_failures.jsonl]
"""
from __future__ import annotations
import argparse, json, os, time, urllib.parse, threading
from typing import Dict, Iterable, Iterator, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from utils.io import read_jsonl, write_jsonl
from utils.config import load_yaml
from utils.logging import get_logger

logger = get_logger("wikipedia_fetch_simple")

# ------------------------------- cache -------------------------------

def load_cache(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                film, text = r.get("film"), r.get("text")
                if film and text:
                    out[film] = text
            except Exception:
                pass
    return out

def append_cache(path: str, items: Iterable[Dict[str, str]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------- helpers ------------------------------

def iri_to_title(iri: str) -> str:
    """
    Converte:
      - 'dbr:Cabaret_(1972_film)' -> 'Cabaret (1972 film)'
      - 'http://dbpedia.org/resource/Cabaret_(1972_film)' -> 'Cabaret (1972 film)'
    """
    iri = iri[:-1] if iri.endswith("/") else iri
    if iri.startswith("dbr:"):
        title = iri.split(":", 1)[1]
    elif iri.startswith("http://dbpedia.org/resource/") or iri.startswith("https://dbpedia.org/resource/"):
        title = iri.rsplit("/", 1)[-1]
    else:
        title = iri
    return urllib.parse.unquote(title).replace("_", " ")

# -------------------------- HTTP session ----------------------------

_tls = threading.local()

def get_session(user_agent: str, timeout: int) -> requests.Session:
    """Session per-thread con UA e timeout; retry semplice a livello applicativo."""
    if getattr(_tls, "session", None) is None:
        s = requests.Session()
        s.headers.update({
            "accept": "application/json",
            "user-agent": user_agent or "NanoSocrates/0.1 (contact: you@example.com)"
        })
        _tls.session = s
    _tls.timeout = timeout
    return _tls.session

# ---------------------- Wikipedia REST summary ----------------------

def wiki_rest_summary(title: str, lang: str, session: requests.Session, timeout: int) -> Tuple[Optional[str], int, str]:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    r = session.get(url, timeout=timeout)
    if r.status_code == 200:
        js = r.json()
        text = (js.get("extract") or "").strip()
        return (text if text else None), 200, url
    return None, r.status_code, url

# -------------------------------- main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", type=str, default="data/interim/wiki_cache.jsonl")
    ap.add_argument("--logfails", type=str, default="data/interim/wiki_failures.jsonl")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    lang = cfg.get("lang", "en")
    timeout = int(cfg.get("timeout_s", 20))
    max_retries = int(cfg.get("max_retries", 2))
    workers = int(args.workers or cfg.get("workers", 8))
    ua = cfg.get("user_agent", "NanoSocrates/0.1 (contact: you@example.com)")

    # Film unici
    seen = set()
    films_all: List[str] = []
    for r in read_jsonl(args.inp):
        f = r["film"]
        if f not in seen:
            seen.add(f); films_all.append(f)
    if args.max:
        films_all = films_all[:args.max]

    logger.info(f"Unique films to fetch: {len(films_all)} (lang={lang}, workers={workers})")

    # Cache
    cache = load_cache(args.cache)
    hits = sum(1 for f in films_all if f in cache)
    logger.info(f"Cache hits: {hits}")
    targets = [f for f in films_all if f not in cache]

    results: Dict[str, str] = dict(cache)
    failures: List[Dict[str, str]] = []

    def task(film_iri: str) -> Optional[Dict[str, str]]:
        title = iri_to_title(film_iri)
        s = get_session(ua, timeout)
        code, url = None, ""
        # retry minimale applicativo (no fallback)
        for attempt in range(max_retries + 1):
            text, code, url = wiki_rest_summary(title, lang, s, timeout)
            if code == 200 and text:
                return {"film": film_iri, "text": text}
            # 429 (Too Many Requests): backoff lineare leggero
            if code == 429:
                time.sleep(0.5 * (attempt + 1))
                continue
            # 404 o altri: piccolo backoff e ritenta
            time.sleep(0.2 * attempt)
        failures.append({"film": film_iri, "title": title, "status": code, "url": url})
        return None

    ok = 0
    if targets:
        logger.info("Fetching Wikipedia summaries...")
        collected_batch = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(task, f): f for f in targets}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="wiki"):
                rec = fut.result()
                if rec:
                    results[rec["film"]] = rec["text"]
                    collected_batch.append(rec)
                    ok += 1
                    if len(collected_batch) >= 500:
                        append_cache(args.cache, collected_batch)
                        collected_batch.clear()
        if collected_batch:
            append_cache(args.cache, collected_batch)

    logger.info(f"Wikipedia phase: ok={ok}, fail={len(failures)}")

    # Log fallimenti (utile per diagnosi)
    if failures:
        os.makedirs(os.path.dirname(args.logfails) or ".", exist_ok=True)
        with open(args.logfails, "w", encoding="utf-8") as f:
            for r in failures:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Failures logged to {args.logfails} (showing first 5):")
        for r in failures[:5]:
            logger.info(f"  film={r['film']} status={r.get('status')} url={r.get('url')} title={r.get('title')}")

    # Output finale (solo successi)
    out_records = ({"film": f, "text": t} for f, t in results.items() if f in films_all and t)
    write_jsonl(args.out, out_records)
    have = sum(1 for _ in read_jsonl(args.out))
    logger.info(f"[OK] Wrote wiki intros to {args.out} (records={have}/{len(films_all)})")

if __name__ == "__main__":
    main()
