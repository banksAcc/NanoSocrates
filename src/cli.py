import argparse, os, torch, random, numpy as np
from src.utils.config import load_yaml, add_common_overrides, apply_overrides
from src.tokenizer.tokenizer_io import TokWrapper
from src.training.dataloaders import JsonlSeq2Seq, pad_collate
from src.model.transformer import TinySeq2Seq
from src.training.loop import train_loop

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cmd_train(args):
    cfg = load_yaml(args.cfg); cfg = apply_overrides(cfg, args.override)

    # --- coercizione robusta tipi numerici ---
    def _as_float(k): 
        if k in cfg: cfg[k] = float(cfg[k])
    def _as_int(k):
        if k in cfg: cfg[k] = int(cfg[k])

    # float
    for k in ("lr", "weight_decay", "dropout"):
        _as_float(k)
    # int
    for k in ("warmup_steps", "max_steps", "batch_size",
              "d_model", "nhead", "enc_layers", "dec_layers",
              "ff_dim", "max_len", "eval_every", "seed"):
        _as_int(k)
    # -----------------------------------------

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = TokWrapper(cfg["tokenizer_file"])
    pad_id = tok.pad_id
    vocab_size = tok.vocab_size()

    train_ds = JsonlSeq2Seq(cfg["train_file"], tokenizer=tok, max_len=cfg["max_len"])
    val_ds   = JsonlSeq2Seq(cfg["val_file"],   tokenizer=tok, max_len=cfg["max_len"])

    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          collate_fn=lambda b: pad_collate(b, pad_id))
    val_dl   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                          collate_fn=lambda b: pad_collate(b, pad_id))

    model = TinySeq2Seq(
        vocab_size=vocab_size,
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_encoder_layers=cfg["enc_layers"], num_decoder_layers=cfg["dec_layers"],
        dim_feedforward=cfg["ff_dim"], dropout=cfg["dropout"],
        pad_id=pad_id, tie_embeddings=True
    ).to(device)

    best = train_loop(model, train_dl, val_dl, cfg, device, pad_id)
    print(f"[train] best val loss: {best:.3f}")

def cmd_overfit(args):
    # usa gli stessi campi di train; riduci max_steps e batch_size via --override
    cmd_train(args)

def main():
    ap = argparse.ArgumentParser(prog="nanosocrates")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train")
    add_common_overrides(ap_train)
    ap_train.set_defaults(func=cmd_train)

    ap_over = sub.add_parser("overfit")
    add_common_overrides(ap_over)
    ap_over.set_defaults(func=cmd_overfit)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
