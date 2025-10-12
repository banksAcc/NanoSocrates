"""Quick helper to duplicate a single JSONL example into a tiny overfit set."""

import json

SRC = "data/processed/rdf2text.train.jsonl"
DST = "data/processed/_mini.train.jsonl"


def main() -> None:
    """Materialise 256 copies of the first training example for smoke tests."""

    with open(SRC, "r", encoding="utf-8") as fh:
        first = json.loads(next(fh))
    with open(DST, "w", encoding="utf-8") as out:
        for i in range(256):
            record = dict(first)
            record["id"] = f"mini-{i}"
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("Wrote", DST)


if __name__ == "__main__":
    main()