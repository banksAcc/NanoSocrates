"""
Caricamento semplice di YAML in dict.
"""
from __future__ import annotations
from typing import Any, Dict
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "rt", encoding="utf-8") as f:
        return yaml.safe_load(f)
