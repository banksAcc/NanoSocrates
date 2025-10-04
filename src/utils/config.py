"""
Caricamento semplice di YAML in dict.
"""
from __future__ import annotations
from typing import Any, Dict
import yaml, argparse

def _transform_processed_paths(node, prefix: str, toy_prefix: str):
    if isinstance(node, dict):
        return {k: _transform_processed_paths(v, prefix, toy_prefix) for k, v in node.items()}
    if isinstance(node, list):
        return [_transform_processed_paths(v, prefix, toy_prefix) for v in node]
    if isinstance(node, str):
        if node.startswith(toy_prefix):
            return node
        if node.startswith(prefix) and node.endswith(".jsonl"):
            return toy_prefix + node[len(prefix):]
    return node

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def add_common_overrides(ap: argparse.ArgumentParser):
    ap.add_argument("--cfg", required=True, help="path yaml (es. configs/train/baseline.yaml)")
    ap.add_argument("--override", nargs="*", default=[], help="chiave=valore (facoltative)")
    ap.add_argument("--toy", action="store_true", help="Reindirizza i path dei dataset verso data/processed/toy")

def apply_overrides(cfg: dict, kv_list):
    for kv in kv_list:
        k, v = kv.split("=", 1)
        # prova a castare numeri/bool
        if v.lower() in ("true", "false"): v = v.lower() == "true"
        else:
            try:
                if "." in v: v = float(v)
                else: v = int(v)
            except: pass
        # supporto chiavi annidate a punto
        cur, *rest = k.split(".")
        node = cfg
        while rest:
            if cur not in node: node[cur] = {}
            node = node[cur]; cur, *rest = rest
        node[cur] = v
    return cfg

def apply_toy_paths(cfg: dict, base_prefix: str = "data/processed", toy_subdir: str = "toy") -> dict:
    """Ritorna una copia del config con i path JSONL spostati su data/processed/toy."""
    if not cfg:
        return cfg
    prefix = base_prefix.rstrip("/") + "/"
    toy_prefix = prefix + toy_subdir.strip("/") + "/"
    transformed = _transform_processed_paths(cfg, prefix, toy_prefix)
    return transformed
