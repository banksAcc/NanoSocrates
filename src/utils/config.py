"""
Caricamento semplice di YAML in dict.
"""
from __future__ import annotations
from typing import Any, Dict
import yaml, argparse

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def add_common_overrides(ap: argparse.ArgumentParser):
    ap.add_argument("--cfg", required=True, help="path yaml (es. configs/train/baseline.yaml)")
    ap.add_argument("--override", nargs="*", default=[], help="chiave=valore (facoltative)")

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
