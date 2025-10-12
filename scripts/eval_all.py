"""Compatibilità retroattiva per la vecchia CLI di valutazione con alias checkpoint."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional, Tuple

from src.utils.config import CHECKPOINT_ALIASES, load_yaml


def _pop_model_alias(argv: List[str]) -> Tuple[Optional[str], List[str]]:
    """Extract ``--model-alias`` from *argv* returning the alias and cleaned args."""
    alias: Optional[str] = None
    cleaned: List[str] = []
    skip_next = False
    for idx, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--model-alias":
            if idx + 1 < len(argv):
                alias = argv[idx + 1]
                skip_next = True
            continue
        if token.startswith("--model-alias="):
            alias = token.split("=", 1)[1]
            continue
        cleaned.append(token)
    return alias, cleaned


def _find_cfg_path(argv: List[str]) -> Optional[Path]:
    """Return the configuration path referenced in *argv* if present."""
    for idx, token in enumerate(argv):
        if token == "--cfg" and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        if token.startswith("--cfg="):
            return Path(token.split("=", 1)[1])
    return None


def _config_alias(cfg_path: Optional[Path]) -> Optional[str]:
    """Read ``model_alias`` from the YAML config if available."""
    if cfg_path is None or not cfg_path.exists():
        return None
    try:
        payload = load_yaml(str(cfg_path))
    except Exception:  # pragma: no cover - caricare YAML è best effort
        return None
    if isinstance(payload, dict):
        alias = payload.get("model_alias")
        if isinstance(alias, str) and alias.strip():
            return alias.strip()
    return None


def _contains_checkpoint_override(argv: List[str]) -> bool:
    """Check whether a ``checkpoint=`` override is already provided."""
    pending_override = False
    for token in argv:
        if token == "--override":
            pending_override = True
            continue
        if token.startswith("--"):
            pending_override = False
        if pending_override and token.startswith("checkpoint="):
            return True
        if token.startswith("checkpoint="):
            return True
    return False


def _inject_checkpoint_override(argv: List[str], checkpoint: str) -> List[str]:
    """Inject ``checkpoint=...`` into ``--override`` arguments when missing."""
    if not checkpoint:
        return argv
    updated = list(argv)
    for idx, token in enumerate(updated):
        if token == "--override":
            insert_at = idx + 1
            while insert_at < len(updated) and not updated[insert_at].startswith("--"):
                insert_at += 1
            updated.insert(insert_at, f"checkpoint={checkpoint}")
            return updated
    updated.extend(["--override", f"checkpoint={checkpoint}"])
    return updated


def _ensure_evaluate_command(argv: List[str]) -> List[str]:
    """Make sure the final command sequence starts with ``evaluate``."""
    if not argv or argv[0] != "evaluate":
        return ["evaluate", *argv]
    return argv


def main() -> None:
    """Reindirizza verso `python -m src.run evaluate` preservando gli argomenti."""
    forwarded = sys.argv[1:]
    alias, forwarded = _pop_model_alias(forwarded)
    cfg_path = _find_cfg_path(forwarded)
    alias = alias or _config_alias(cfg_path)

    if alias:
        alias = alias.strip()
        resolved = CHECKPOINT_ALIASES.get(alias)
        if resolved is None:
            print(
                f"[eval_all] Alias modello '{alias}' non riconosciuto: "
                "passa --override checkpoint=... manualmente.",
                file=sys.stderr,
            )
        elif not _contains_checkpoint_override(forwarded):
            forwarded = _inject_checkpoint_override(forwarded, resolved)

    forwarded = _ensure_evaluate_command(forwarded)

    sys.argv = [sys.argv[0], *forwarded]

    from src.run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
