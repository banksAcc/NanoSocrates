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
    _ENTITY_REGEX = re.compile(r"(dbr:|dbo:|dbc:|dbp:|http://|https://)\S+")

    def __init__(self, path: str, tokenizer, max_len: int = 256, *, enable_entity_spans: bool = False):
        assert os.path.exists(path), f"Missing dataset: {path}"
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.enable_entity_spans = bool(enable_entity_spans)
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
        self.sot_id = self.tokenizer.token_to_id("<SOT>")

    def __len__(self): return len(self.items)

    def _find_subsequence(self, sequence: List[int], pattern: List[int], *, start: int = 0) -> Optional[int]:
        if not pattern or not sequence:
            return None
        last = len(sequence) - len(pattern)
        if last < start:
            return None
        for idx in range(start, last + 1):
            if sequence[idx : idx + len(pattern)] == pattern:
                return idx
        return None

    def _compute_entity_spans(self, tokens: List[int], text: str) -> List[tuple[int, int]]:
        if not self.enable_entity_spans:
            return []
        matches = list(self._ENTITY_REGEX.finditer(text))
        if not matches:
            return []
        spans: List[tuple[int, int]] = []
        offset = 1 if tokens and self.sot_id is not None and tokens[0] == self.sot_id else 0
        cursor = 0
        for match in matches:
            entity = match.group(0)
            ent_ids = self.tokenizer.encode(entity)
            if not ent_ids:
                continue
            start = self._find_subsequence(tokens, ent_ids, start=cursor)
            if start is None:
                # riprova dall'inizio per essere tolleranti con entità ripetute
                start = self._find_subsequence(tokens, ent_ids)
                if start is None:
                    continue
            cursor = start + len(ent_ids)
            start_for_loss = start - offset
            if start_for_loss < 0:
                continue
            spans.append((start_for_loss, len(ent_ids)))
        return spans

    def __getitem__(self, idx):
        ex = self.items[idx]
        x_ids = self.tokenizer.encode(ex.input)[: self.max_len - 1]

        target_text = ex.target or ""
        if ex.task == "rdfcomp1" and target_text and not target_text.strip().startswith("<SOT>"):
            target_text = "<SOT> " + target_text.strip()

        y_tokens = self.tokenizer.encode(target_text)[: self.max_len - 1]
        spans = self._compute_entity_spans(y_tokens, target_text)
        y_ids = list(y_tokens)
        if self.eot_id is not None:
            x_ids.append(self.eot_id)
            y_ids.append(self.eot_id)
        item = {
            "input_ids": torch.tensor(x_ids, dtype=torch.long),
            "labels": torch.tensor(y_ids, dtype=torch.long),
        }
        mask_positions = torch.tensor([s for s, _ in spans], dtype=torch.long)
        mask_lengths = torch.tensor([l for _, l in spans], dtype=torch.long)
        item["mask_positions"] = mask_positions
        item["mask_lengths"] = mask_lengths
        return item

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
    mask_pos_list = [b.get("mask_positions", torch.empty(0, dtype=torch.long)) for b in batch]
    mask_len_list = [b.get("mask_lengths", torch.empty(0, dtype=torch.long)) for b in batch]

    max_spans = max((len(t) for t in mask_pos_list), default=0)
    if max_spans > 0:
        mask_positions = torch.full((len(batch), max_spans), -1, dtype=torch.long)
        mask_lengths = torch.zeros((len(batch), max_spans), dtype=torch.long)
        mask_valid = torch.zeros((len(batch), max_spans), dtype=torch.bool)
        for i, (pos, leng) in enumerate(zip(mask_pos_list, mask_len_list)):
            n = len(pos)
            if n == 0:
                continue
            mask_positions[i, : n] = pos
            mask_lengths[i, : n] = leng
            mask_valid[i, : n] = True
    else:
        mask_positions = torch.empty((len(batch), 0), dtype=torch.long)
        mask_lengths = torch.empty((len(batch), 0), dtype=torch.long)
        mask_valid = torch.empty((len(batch), 0), dtype=torch.bool)

    return {
        "input_ids": x,
        "attention_mask": attn,
        "labels": y,
        "mask_positions": mask_positions,
        "mask_lengths": mask_lengths,
        "mask_valid": mask_valid,
    }

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
