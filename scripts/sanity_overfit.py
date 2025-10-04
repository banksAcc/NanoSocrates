"""Utility per verificare rapidamente se il modello riesce ad overfittare un batch."""

import argparse

from src.cli import cmd_overfit
from src.utils.config import add_common_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sanity check: forza l'overfit su un singolo batch usando src.cli overfit",
    )
    add_common_overrides(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd_overfit(args)


if __name__ == "__main__":
    main()
