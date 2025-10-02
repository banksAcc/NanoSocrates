# src/cli.py
import argparse, os, torch, random, numpy as np
from src.utils.config import load_yaml, add_common_overrides, apply_overrides
from src.tokenizer.tokenizer_io import TokWrapper
from src.training.dataloaders import JsonlSeq2Seq, pad_collate, build_multitask_train, build_concat_val
from src.model.transformer import TinySeq2Seq
from src.training.loop import train_loop

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _coerce_cfg_types(cfg: dict):
    """Rende robusti i tipi numerici anche se arrivano come stringhe dal YAML/CLI."""
    def _as_float(k): 
        if k in cfg: cfg[k] = float(cfg[k])
    def _as_int(k):
        if k in cfg: cfg[k] = int(cfg[k])

    # float
    for k in ("lr", "weight_decay", "dropout"):
        if k in cfg: 
            try: _as_float(k)
            except: pass
    # int
    for k in ("warmup_steps", "max_steps", "batch_size",
              "d_model", "nhead", "enc_layers", "dec_layers",
              "ff_dim", "max_len", "eval_every", "seed", "num_workers"):
        if k in cfg:
            try: _as_int(k)
            except: pass
    return cfg

def cmd_train(args):
    # 1) carica config + override da CLI e sistema i tipi
    cfg = load_yaml(args.cfg)
    cfg = apply_overrides(cfg, args.override)   # es: --override lr=3e-4 batch_size=8
    cfg = _coerce_cfg_types(cfg)

    # 2) device selection (A) — usa "cuda" se richiesto e disponibile
    set_seed(cfg.get("seed", 42))
    want = cfg.get("device", "cuda")  # puoi mettere "cpu" o "cuda" in YAML
    device = "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[device] using: {device}")

    # 3) tokenizer
    tok = TokWrapper(cfg["tokenizer_file"])
    pad_id = tok.pad_id
    vocab_size = tok.vocab_size()

    # 4) datasets: single-task (train_file) oppure multi-task (datasets: [...])
    from torch.utils.data import DataLoader
    num_workers = int(cfg.get("num_workers", 4))
    pin = (device == "cuda")

    if "datasets" in cfg:
        # multi-task
        train_dsets, train_w = [], []
        val_dsets = []
        for d in cfg["datasets"]:
            train_dsets.append(JsonlSeq2Seq(d["train"], tokenizer=tok, max_len=cfg["max_len"]))
            train_w.append(int(d.get("weight", 1)))
            val_dsets.append(JsonlSeq2Seq(d["val"], tokenizer=tok, max_len=cfg["max_len"]))
        train_ds = build_multitask_train(train_dsets, train_w)
        val_ds   = build_concat_val(val_dsets)
    else:
        # single-task (retro-compatibile)
        train_ds = JsonlSeq2Seq(cfg["train_file"], tokenizer=tok, max_len=cfg["max_len"])
        val_ds   = JsonlSeq2Seq(cfg["val_file"],   tokenizer=tok, max_len=cfg["max_len"])

    from functools import partial
    collate = partial(pad_collate, pad_id=pad_id)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          collate_fn=collate, num_workers=num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                          collate_fn=collate, num_workers=num_workers, pin_memory=pin)
    
    # 5) modello
    model = TinySeq2Seq(
        vocab_size=vocab_size,
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_encoder_layers=cfg["enc_layers"], num_decoder_layers=cfg["dec_layers"],
        dim_feedforward=cfg["ff_dim"], dropout=cfg["dropout"],
        pad_id=pad_id, tie_embeddings=True
    ).to(device)

    # 6) training loop
    best = train_loop(model, train_dl, val_dl, cfg, device, pad_id)
    print(f"[train] best val loss: {best:.3f}")

def cmd_overfit(args):
    # usa la stessa pipeline ma con max_steps/batch_size ridotti via --override
    cmd_train(args)

def main():
    ap = argparse.ArgumentParser(prog="nanosocrates")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train")
    add_common_overrides(ap_train)   # --cfg ... --override k=v ...
    ap_train.set_defaults(func=cmd_train)

    ap_over = sub.add_parser("overfit")
    add_common_overrides(ap_over)
    ap_over.set_defaults(func=cmd_overfit)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
