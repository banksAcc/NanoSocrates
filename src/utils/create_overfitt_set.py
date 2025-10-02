import json, itertools
src = "data/processed/rdf2text.train.jsonl"
dst = "data/processed/_mini.train.jsonl"
with open(src, "r", encoding="utf-8") as f:
    first = json.loads(next(f))
with open(dst, "w", encoding="utf-8") as g:
    for i in range(256):
        r = dict(first); r["id"] = f"mini-{i}"
        g.write(json.dumps(r, ensure_ascii=False) + "\n")
print("Wrote", dst)