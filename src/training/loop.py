import os, torch
from tqdm import tqdm
from .scheduler import cosine_with_warmup

@torch.no_grad()
def evaluate(model, dataloader, device, pad_id):
    model.eval()
    tot, n = 0.0, 0
    for batch in dataloader:
        inp = batch["input_ids"].to(device, non_blocking=True)
        att = batch["attention_mask"].to(device, non_blocking=True)
        lab = batch["labels"].to(device, non_blocking=True)
        out = model(inp, att, lab)
        tot += out["loss"].item(); n += 1
    return tot / max(1, n)

def train_loop(model, train_dl, val_dl, cfg, device, pad_id):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # ✅ nuove API (evita i FutureWarning)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    autocast = (lambda: torch.amp.autocast(device_type="cuda")) if device == "cuda" else (lambda: nullcontext())

    # opzionale: migliora throughput con input variabili
    torch.backends.cudnn.benchmark = True if device == "cuda" else False

    os.makedirs(cfg["save_dir"], exist_ok=True)
    step, best_val = 0, float("inf")
    pbar = tqdm(total=cfg["max_steps"], desc="train")
    model.train()

    while step < cfg["max_steps"]:
        for batch in train_dl:
            step += 1
            lr_scale = cosine_with_warmup(step, cfg["warmup_steps"], cfg["max_steps"])
            for pg in opt.param_groups: pg["lr"] = cfg["lr"] * lr_scale

            inp = batch["input_ids"].to(device, non_blocking=True)
            att = batch["attention_mask"].to(device, non_blocking=True)
            lab = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    out = model(inp, att, lab)
                    loss = out["loss"]
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                out = model(inp, att, lab)
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{pg['lr']:.2e}"})
            pbar.update(1)

            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                val = evaluate(model, val_dl, device, pad_id)
                torch.save({"model": model.state_dict(), "config": cfg},
                           os.path.join(cfg["save_dir"], f"step{step}_valloss{val:.3f}.pt"))
                if val < best_val:
                    best_val = val
                    torch.save({"model": model.state_dict(), "config": cfg},
                               os.path.join(cfg["save_dir"], "best.pt"))
                model.train()
            if step >= cfg["max_steps"]: break
    pbar.close()
    return best_val

# util per CPU
from contextlib import nullcontext
