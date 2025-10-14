"""Caricamento semplice di YAML in dict e utility per i config."""
from __future__ import annotations

import argparse
from typing import Dict

import yaml

CHECKPOINT_PLACEHOLDER = "<<override-me>>"
"""Segnaposto usato nei config di valutazione quando manca un checkpoint reale."""

CHECKPOINT_ALIASES = {
    # Multitask T5 con mixing standard.
    "multitask_default": "checkpoints/multitask_default/best.pt",
    "mix": "checkpoints/multitask_default/best.pt",
    # Encoder-decoder vanilla con RoPE abilitato.
    "rope_on": "checkpoints/rope_on/best.pt",
    "rope": "checkpoints/rope_on/best.pt",
    # Alias storico per il preset vanilla senza RoPE.
    "baseline": "checkpoints/baseline/best.pt",
}
"""Mappa alias→path per i checkpoint salvati più comuni."""

def _transform_processed_paths(node, prefix: str, toy_prefix: str):
    """Recursively replace ``prefix`` with ``toy_prefix`` for JSONL dataset paths."""

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
    """Load a YAML configuration file returning the parsed Python object."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_checkpoint_reference(cfg: Dict[str, object]) -> Dict[str, object]:
    """Risolvi il checkpoint usando l'alias quando quello indicato è fittizio.

    Regole:
      - Se `checkpoint` è omesso o vuoto, usa `model_alias`.
      - Se `checkpoint` è un segnaposto (es. CHECKPOINT_PLACEHOLDER o "overfit.pt")
        oppure punta a un path inesistente, prova a risolvere tramite `model_alias`.
      - Altrimenti lascia invariato `checkpoint`.
    """

    if not isinstance(cfg, dict) or "checkpoint" not in cfg:
        return cfg

    checkpoint = cfg.get("checkpoint")
    alias = cfg.get("model_alias")

    # Normalizza
    if isinstance(checkpoint, str):
        ckpt_str = checkpoint.strip()
    else:
        ckpt_str = ""

    # Decide se il checkpoint è "fittizio" e quindi risolvibile via alias
    needs_resolution = False
    if not ckpt_str:
        needs_resolution = True
    elif ckpt_str == CHECKPOINT_PLACEHOLDER:
        needs_resolution = True
    elif ckpt_str.lower() == "overfit.pt":
        needs_resolution = True
    else:
        try:
            from pathlib import Path

            if not Path(ckpt_str).exists():
                needs_resolution = True
        except Exception:
            # In caso di path non valido, prova comunque con l'alias
            needs_resolution = True

    if not needs_resolution:
        return cfg

    if isinstance(alias, str) and alias.strip():
        key = alias.strip()
        resolved = CHECKPOINT_ALIASES.get(key)
        if resolved:
            cfg["checkpoint"] = resolved
            return cfg
        raise ValueError(
            f"Alias modello '{key}' non riconosciuto: passa --override checkpoint=... oppure "
            "aggiorna CHECKPOINT_ALIASES."
        )

    raise ValueError(
        "Config di valutazione senza checkpoint valido: usa --override checkpoint=... oppure definisci 'model_alias'."
    )

def add_common_overrides(ap: argparse.ArgumentParser):
    """Attach shared CLI arguments to training/evaluation sub-commands."""

    ap.add_argument("--cfg", required=True, help="path yaml (es. configs/train/multitask_default.yaml)")
    ap.add_argument("--override", nargs="*", default=[], help="chiave=valore (facoltative)")
    ap.add_argument("--toy", action="store_true", help="Reindirizza i path dei dataset verso data/processed/toy")


def apply_overrides(cfg: dict, kv_list):
    """Apply CLI ``key=value`` overrides to nested dictionaries in *cfg*."""

    for kv in kv_list:
        k, v = kv.split("=", 1)
        # Attempt to coerce booleans/numbers so configs remain strongly typed.
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                pass

        # Support dot-separated keys to override nested dictionaries.
        cur, *rest = k.split(".")
        node = cfg
        while rest:
            if cur not in node:
                node[cur] = {}
            node = node[cur]
            cur, *rest = rest
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
