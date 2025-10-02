from dataclasses import dataclass
from typing import List, Optional
import json, os, torch, re
from torch.utils.data import Dataset

@dataclass
class Example:
    id: str
    task: str
    input: str
    target: str
    film: Optional[str] = None

def _infer_task(path: str, input_text: str) -> str:
    # 1) prova a inferire dal nome file
    p = os.path.basename(path).lower()
    if "rdf2text" in p: return "rdf2text"
    if "text2rdf" in p: return "text2rdf"
    if "rcf1" in p or "completion1" in p: return "rcf1"
    if "rcf2" in p or "completion2" in p: return "rcf2"
    # 2) prova a inferire dai marker nel testo
    t = input_text or ""
    if "<RDF2Text>" in t: return "rdf2text"
    if "<Text2RDF>" in t: return "text2rdf"
    if "<CONTINUERDF>" in t: return "rcf2"
    if "<MASK>" in t: return "rcf1"
    # 3) fallback generico
    return "unknown"

class JsonlSeq2Seq(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 256):
        assert os.path.exists(path), f"Missing dataset: {path}"
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items: List[Example] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                # campi robusti
                ex_id   = str(r.get("id") or r.get("film") or i)
                ex_inp  = r.get("input", "")
                ex_tgt  = r.get("target", "")
                ex_task = r.get("task") or _infer_task(path, ex_inp)
                ex_film = r.get("film")
                self.items.append(Example(
                    id=ex_id, task=ex_task, input=ex_inp, target=ex_tgt, film=ex_film
                ))

        # EOT opzionale
        self.eot_id = self.tokenizer.token_to_id("<EOT>")
        self.pad_id = self.tokenizer.pad_id

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        x_ids = self.tokenizer.encode(ex.input)[: self.max_len - 1]
        y_ids = self.tokenizer.encode(ex.target)[: self.max_len - 1]
        if self.eot_id is not None:
            x_ids.append(self.eot_id)
            y_ids.append(self.eot_id)
        return {
            "input_ids": torch.tensor(x_ids, dtype=torch.long),
            "labels": torch.tensor(y_ids, dtype=torch.long)
        }

def pad_collate(batch, pad_id: int):
    def _pad(seqs):
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
        return out
    x = _pad([b["input_ids"] for b in batch])
    y = _pad([b["labels"] for b in batch])
    attn = (x != pad_id).long()
    return {"input_ids": x, "attention_mask": attn, "labels": y}
