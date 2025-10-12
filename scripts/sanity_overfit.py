"""Utility per verificare rapidamente se il modello riesce ad overfittare un batch."""

import argparse

from src.run import cmd_overfit
from src.utils.config import add_common_overrides


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser reused by :mod:`src.run` for the overfit command."""
    parser = argparse.ArgumentParser(
        description="Sanity check: forza l'overfit su un singolo batch usando src.run overfit",
    )
    add_common_overrides(parser)
    return parser


def main() -> None:
    """Parse arguments and delegate to :func:`src.run.cmd_overfit`."""
    parser = build_parser()
    args = parser.parse_args()
    cmd_overfit(args)


if __name__ == "__main__":
    main()
