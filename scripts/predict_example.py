"""Esempio minimale di inference da riga di comando."""
from __future__ import annotations

import argparse

from src.decoding.base import decode_to_text
from src.eval.evaluate import load_model_and_tokenizer

TASK_MARKERS = {
    "text2rdf": "<Text2RDF>",
    "rdf2text": "<RDF2Text>",
    "rdfcomp2": "<CONTINUERDF>",
    "rdfcomp1": "<MASK>",
}


def prepare_input(text: str, task: str | None) -> str:
    text = text.strip()
    if task:
        marker = TASK_MARKERS.get(task)
        if marker and marker not in text:
            if task == "rdfcomp1" and "<MASK>" not in text:
                text = f"{text} <MASK>".strip()
            else:
                text = f"{text} {marker}".strip()
    return text


def main():
    ap = argparse.ArgumentParser(description="Esegue una singola predizione")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", required=True, help="Input testuale o RDF linearizzato")
    ap.add_argument("--task", choices=list(TASK_MARKERS.keys()))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--model-override", nargs="*", default=[])
    args = ap.parse_args()

    overrides = {}
    if args.model_override:
        from src.utils.config import apply_overrides

        overrides = apply_overrides({}, args.model_override)

    model, tokenizer, device, _ = load_model_and_tokenizer(
        args.tokenizer,
        args.checkpoint,
        device=args.device,
        overrides=overrides,
    )

    prepared = prepare_input(args.input, args.task)
    output = decode_to_text(
        model,
        tokenizer,
        prepared,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    print(output)


if __name__ == "__main__":
    main()
