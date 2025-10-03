from dataclasses import dataclass
from typing import List, Optional
import json, os, torch, re, random
from torch.utils.data import Dataset, ConcatDataset 

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
    if "rdfcomp1" in p or "completion1" in p: return "rdfcomp1"
    if "rdfcomp2" in p or "completion2" in p: return "rdfcomp2"
    # 2) prova a inferire dai marker nel testo
    # 2) prova a inferire dai marker nel testo
    t = input_text or ""
    if "<RDF2Text>" in t: return "rdf2text"
    if "<Text2RDF>" in t: return "text2rdf"
    if "<CONTINUERDF>" in t: return "rdfcomp2"
    if "<MASK>" in t: return "rdfcomp1"
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
    attn = (x != pad_id)            # <-- bool
    return {"input_ids": x, "attention_mask": attn, "labels": y}

#x la parte del multitask
class MultiTaskRandomDataset(Dataset):
    """
    Dataset stocastico: ad ogni __getitem__ ignora 'i' e
    pesca un task secondo i pesi, poi un esempio random in quel task.
    Così un 'epoch' è stocastico, ma è pratico e veloce.
    """
    def __init__(self, datasets_with_weights):
        """
        datasets_with_weights: List[ (JsonlSeq2Seq, weight:int) ]
        """
        self.ds = [d for d, w in datasets_with_weights]
        self.w  = [max(0.0, float(w)) for d, w in datasets_with_weights]
        sw = sum(self.w)
        assert sw > 0, "MultiTaskRandomDataset: somma pesi nulla."
        # cum prob
        acc, cum = [], 0.0
        for wi in self.w:
            cum += wi / sw
            acc.append(cum)
        self.cum = acc
        # lunghezza 'nominale': somma delle lunghezze (va bene per avere ~epoch size)
        self._len = sum(len(d) for d in self.ds)

    def __len__(self): return self._len

    def _pick_dataset(self):
        r = random.random()
        for i, c in enumerate(self.cum):
            if r <= c: return self.ds[i]
        return self.ds[-1]

    def __getitem__(self, i):
        d = self._pick_dataset()
        j = random.randrange(len(d))
        return d[j]

def build_multitask_train(dsets, weights):
    assert len(dsets) == len(weights)
    pairs = list(zip(dsets, weights))
    return MultiTaskRandomDataset(pairs)

def build_concat_val(dsets):
    # validazione: concatenazione di tutte le val per una loss media
    return ConcatDataset(dsets)
