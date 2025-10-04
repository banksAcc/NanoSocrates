"""Utility condivise per l'inizializzazione e il logging su Weights & Biases.

Questo modulo definisce le stesse primitive che venivano utilizzate dallo
script ``scripts/eval_all.py`` così da poterle riutilizzare anche dalla CLI.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Tuple


def _build_wandb_init_kwargs(
    wandb_cfg: Mapping[str, Any],
    run_config: Mapping[str, Any] | None,
    mode_value: str,
) -> MutableMapping[str, Any]:
    """Prepara gli argomenti da passare a ``wandb.init``.

    L'helper rimuove le chiavi impostate a ``None`` per evitare warning e
    supporta ``tags`` facoltativi. ``run_config`` viene allegato così da avere
    il config completo visibile nella run.
    """

    init_kwargs: MutableMapping[str, Any] = {
        "project": wandb_cfg.get("project"),
        "entity": wandb_cfg.get("entity"),
        "name": wandb_cfg.get("run_name"),
        "tags": wandb_cfg.get("tags"),
        "mode": mode_value,
    }
    if run_config is not None:
        init_kwargs["config"] = run_config

    # ``tags`` può essere ``None`` o una lista vuota: in entrambi i casi la
    # rimuoviamo per evitare che wandb alzi errori.
    if not init_kwargs.get("tags"):
        init_kwargs.pop("tags", None)

    # Filtra i valori ``None``.
    return {k: v for k, v in init_kwargs.items() if v is not None}


def _maybe_init_wandb(
    wandb_cfg: Mapping[str, Any] | None,
    run_config: Mapping[str, Any] | None = None,
) -> Tuple[Any, Any]:
    """Prova ad inizializzare una run Weights & Biases.

    Restituisce ``(run, module)`` se la run è stata creata con successo oppure
    ``(None, None)`` se il logging è disabilitato o non è disponibile la
    libreria. In caso di errori prova automaticamente un fallback in modalità
    ``offline`` mantenendo lo stesso payload di configurazione.
    """

    wandb_cfg = wandb_cfg or {}
    mode = str(wandb_cfg.get("mode", "disabled") or "disabled")
    if mode.lower() == "disabled":
        return None, None

    try:
        import wandb as wandb_module  # type: ignore
    except Exception as exc:  # pragma: no cover - messaggi diagnostici
        print(f"[wandb] library not available ({exc}); skipping logging.")
        return None, None

    init_kwargs = _build_wandb_init_kwargs(wandb_cfg, run_config, mode)

    try:
        run = wandb_module.init(**init_kwargs)
    except Exception as exc:  # pragma: no cover - dipende da wandb
        print(f"[wandb] init failed ({exc}); retrying in offline mode.")
        try:
            init_kwargs_offline = _build_wandb_init_kwargs(
                wandb_cfg, run_config, "offline"
            )
            run = wandb_module.init(**init_kwargs_offline)
        except Exception as offline_exc:  # pragma: no cover
            print(
                f"[wandb] offline fallback failed ({offline_exc}); disabling logging."
            )
            return None, None
    try:
        if run_config is not None:
            run.config.update(run_config, allow_val_change=True)
    except Exception as exc:  # pragma: no cover - solo logging
        print(f"[wandb] config update failed: {exc}")
    return run, wandb_module


def _flatten_for_logging(
    payload: Mapping[str, Any] | Iterable[Tuple[str, Any]],
    *,
    prefix: str | None = None,
    sep: str = "/",
) -> dict[str, float]:
    """Appiattisce dizionari arbitrariamente annidati per il logging W&B.

    Solo i valori numerici (int/float/bool) vengono mantenuti: gli altri tipi
    vengono scartati. Le liste/tuple vengono espanse numerando gli elementi.
    """

    flat: dict[str, float] = {}

    if isinstance(payload, Mapping):
        items = payload.items()
    else:
        items = payload

    for key, value in items:
        full_key = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_for_logging(value, prefix=full_key, sep=sep))
        elif isinstance(value, (list, tuple)):
            for idx, sub_value in enumerate(value):
                sub_key = f"{full_key}{sep}{idx}"
                if isinstance(sub_value, Mapping):
                    flat.update(
                        _flatten_for_logging(sub_value, prefix=sub_key, sep=sep)
                    )
                elif isinstance(sub_value, (int, float)):
                    flat[sub_key] = float(sub_value)
        elif isinstance(value, (int, float, bool)):
            flat[full_key] = float(value)
    return flat


__all__ = ["_maybe_init_wandb", "_flatten_for_logging"]
