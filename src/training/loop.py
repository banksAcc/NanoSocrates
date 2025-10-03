import math
import os
import torch
from contextlib import nullcontext
from inspect import signature

from tqdm import tqdm

from .scheduler import cosine_with_warmup


def _supports_kwarg(fn, name):
    try:
        return name in signature(fn).parameters
    except (TypeError, ValueError):  # signature may fail on some builtins
        return False


@torch.no_grad()
def evaluate(model, dataloader, device, pad_id):
    model.eval()
    tot, n = 0.0, 0
    for batch in dataloader:
        inp = batch["input_ids"].to(device, non_blocking=True)
        att = batch["attention_mask"].to(device, non_blocking=True)
        lab = batch["labels"].to(device, non_blocking=True)
        out = model(inp, att, labels=lab)
        tot += out["loss"].item()
        n += 1
    return tot / max(1, n)


def train_loop(
    model,
    train_dl,
    val_dl,
    cfg,
    device,
    pad_id,
    steps_per_epoch,
    *,
    max_train_steps=0,
):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # ✅ nuove API (evita i FutureWarning)
    if device == "cuda":
        grad_scaler_ctor = torch.amp.GradScaler
        if _supports_kwarg(grad_scaler_ctor.__init__, "device_type"):
            scaler = grad_scaler_ctor(device_type="cuda")
        else:
            try:
                scaler = grad_scaler_ctor()
            except TypeError:
                cuda_amp = getattr(torch.cuda, "amp", None)
                if cuda_amp is not None and hasattr(cuda_amp, "GradScaler"):
                    scaler = cuda_amp.GradScaler()
                else:
                    raise

        autocast_fn = getattr(torch.amp, "autocast", torch.autocast)
        try:
            autocast_ctx = autocast_fn(device_type=device)
        except TypeError:
            autocast_ctx = autocast_fn(device)
    else:
        scaler = None
        autocast_ctx = nullcontext()

    # opzionale: migliora throughput con input variabili
    torch.backends.cudnn.benchmark = True if device == "cuda" else False

    os.makedirs(cfg["save_dir"], exist_ok=True)

    steps_per_epoch = max(1, int(steps_per_epoch))
    num_epochs = int(cfg["num_epochs"])
    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    warmup_steps = int(cfg.get("warmup_steps", 0))

    max_train_steps = int(max_train_steps or 0)
    total_optimizer_steps = max(1, num_epochs * optimizer_steps_per_epoch)
    if max_train_steps > 0:
        total_optimizer_steps = min(total_optimizer_steps, max_train_steps)

    global_step = 0
    best_val = float("inf")
    best_epoch = 0
    best_step = 0

    stop_training = False

    for epoch in range(num_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        running_loss = 0.0
        epoch_bar = tqdm(
            enumerate(train_dl, start=1),
            total=steps_per_epoch,
            desc=f"epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )

        for batch_idx, batch in epoch_bar:
            inp = batch["input_ids"].to(device, non_blocking=True)
            att = batch["attention_mask"].to(device, non_blocking=True)
            lab = batch["labels"].to(device, non_blocking=True)

            with autocast_ctx:
                out = model(inp, att, labels=lab)
                loss = out["loss"]

            loss_to_backward = loss / grad_accum
            if scaler is not None:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            running_loss += loss.item()

            perform_step = (batch_idx % grad_accum == 0) or (batch_idx == steps_per_epoch)
            if perform_step:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                global_step += 1
                lr_scale = cosine_with_warmup(
                    global_step,
                    warmup_steps,
                    total_optimizer_steps,
                )
                for pg in opt.param_groups:
                    pg["lr"] = cfg["lr"] * lr_scale

                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

                if max_train_steps and global_step >= max_train_steps:
                    stop_training = True

            current_lr = opt.param_groups[0]["lr"]
            avg_loss = running_loss / batch_idx
            epoch_bar.set_postfix({
                "loss": f"{avg_loss:.3f}",
                "lr": f"{current_lr:.2e}",
                "step": global_step,
            })

            if batch_idx >= steps_per_epoch or stop_training:
                break

        epoch_bar.close()

        val = evaluate(model, val_dl, device, pad_id)
        ckpt = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch + 1,
            "global_step": global_step,
            "val_loss": val,
            "optimizer": opt.state_dict(),
        }
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, os.path.join(cfg["save_dir"], f"epoch{epoch + 1:03d}.pt"))
        if val < best_val:
            best_val = val
            best_epoch = epoch + 1
            best_step = global_step
            torch.save(ckpt, os.path.join(cfg["save_dir"], "best.pt"))

        model.train()
        tqdm.write(
            f"[epoch {epoch + 1}] val loss: {val:.3f} | best: {best_val:.3f}"
            f" @ step {best_step}"
        )

        if stop_training:
            break

    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "global_step": global_step,
    }
