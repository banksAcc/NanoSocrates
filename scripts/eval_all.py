"""Compatibilità retroattiva per la vecchia CLI di valutazione."""
from __future__ import annotations

import sys


def main() -> None:
    """Reindirizza verso `python -m src.run evaluate` preservando gli argomenti."""

    if len(sys.argv) < 2 or sys.argv[1] != "evaluate":
        sys.argv.insert(1, "evaluate")

    from src.run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
