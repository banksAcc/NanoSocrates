import math
import os
from collections import defaultdict
from contextlib import nullcontext
from inspect import signature

import torch
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
    metric_totals = defaultdict(float)
    metric_counts = defaultdict(int)
    for batch in dataloader:
        inp = batch["input_ids"].to(device, non_blocking=True)
        att = batch["attention_mask"].to(device, non_blocking=True)
        lab = batch["labels"].to(device, non_blocking=True)
        extra = {}
        if "mask_positions" in batch:
            extra["mask_positions"] = batch["mask_positions"].to(device, non_blocking=True)
        if "mask_lengths" in batch:
            extra["mask_lengths"] = batch["mask_lengths"].to(device, non_blocking=True)
        out = model(inp, att, labels=lab, **extra)
        loss = out["loss"].item()
        tot += loss
        n += 1
        metrics = out.get("metrics") if isinstance(out, dict) else None
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_totals[name] += float(value)
                    metric_counts[name] += 1
    avg_loss = tot / max(1, n)
    if metric_totals:
        averaged = {
            name: metric_totals[name] / max(1, metric_counts[name])
            for name in metric_totals
        }
        averaged["loss"] = avg_loss
        return averaged
    return avg_loss


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
    wandb_run=None,
    wandb_module=None,
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

    early_cfg = cfg.get("early_stopping") or {}
    try:
        patience = int(early_cfg.get("patience", 0))
    except (TypeError, ValueError):
        patience = 0
    try:
        min_delta = float(early_cfg.get("min_delta", 0.0))
    except (TypeError, ValueError):
        min_delta = 0.0
    min_delta = max(0.0, min_delta)
    patience = max(0, patience)
    epochs_since_improvement = 0

    stop_training = False
    overfit_one_batch = bool(cfg.get("overfit_one_batch", False))

    wandb_cfg = cfg.get("wandb") or {}
    watch_model = bool(wandb_cfg.get("watch", False))
    wandb_logger = getattr(wandb_run, "log", None) if wandb_run is not None else None

    if watch_model and wandb_module is not None and wandb_run is not None:
        try:
            wandb_module.watch(model)
        except Exception as exc:
            tqdm.write(f"[wandb] watch() failed: {exc}")

    try:
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
                extra = {}
                if "mask_positions" in batch:
                    extra["mask_positions"] = batch["mask_positions"].to(device, non_blocking=True)
                if "mask_lengths" in batch:
                    extra["mask_lengths"] = batch["mask_lengths"].to(device, non_blocking=True)

                with autocast_ctx:
                    out = model(inp, att, labels=lab, **extra)
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

                if perform_step and wandb_logger is not None:
                    log_payload = {
                        "train/loss": loss.item(),
                        "train/loss_avg": avg_loss,
                        "train/epoch": epoch + batch_idx / max(1, steps_per_epoch),
                        "lr": current_lr,
                        "global_step": global_step,
                    }
                    metrics = out.get("metrics") if isinstance(out, dict) else None
                    if isinstance(metrics, dict):
                        for name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                log_payload[f"train/{name}"] = float(value)
                    try:
                        wandb_logger(log_payload, step=global_step)
                    except Exception as exc:
                        tqdm.write(f"[wandb] log() failed: {exc}")

                if overfit_one_batch:
                    stop_training = True

                if batch_idx >= steps_per_epoch or stop_training:
                    break

            epoch_bar.close()

            val = evaluate(model, val_dl, device, pad_id)
            if isinstance(val, dict):
                val_loss = float(val.get("loss", 0.0))
                extra_val_metrics = {
                    name: value for name, value in val.items() if name != "loss"
                }
            else:
                val_loss = float(val)
                extra_val_metrics = {}
            ckpt = {
                "model": model.state_dict(),
                "config": cfg,
                "epoch": epoch + 1,
                "global_step": global_step,
                "val_loss": val_loss,
                "optimizer": opt.state_dict(),
            }
            if scaler is not None:
                ckpt["scaler"] = scaler.state_dict()
            torch.save(ckpt, os.path.join(cfg["save_dir"], f"epoch{epoch + 1:03d}.pt"))
            improved = val_loss < (best_val - min_delta)
            if improved:
                best_val = val_loss
                best_epoch = epoch + 1
                best_step = global_step
                epochs_since_improvement = 0
                torch.save(ckpt, os.path.join(cfg["save_dir"], "best.pt"))
            else:
                if not math.isinf(best_val):
                    epochs_since_improvement += 1

            if wandb_logger is not None:
                log_payload = {
                    "val/loss": val_loss,
                    "val/best_loss": best_val,
                    "val/best_epoch": best_epoch,
                    "early_stopping/epochs_since_improvement": epochs_since_improvement,
                    "early_stopping/patience": patience,
                    "early_stopping/triggered": bool(
                        patience > 0 and epochs_since_improvement >= patience
                    ),
                }
                for name, value in extra_val_metrics.items():
                    if isinstance(value, (int, float)):
                        log_payload[f"val/{name}"] = float(value)
                try:
                    wandb_logger(log_payload, step=global_step)
                except Exception as exc:
                    tqdm.write(f"[wandb] log() failed: {exc}")

            model.train()
            status = (
                f"[epoch {epoch + 1}] val loss: {val_loss:.3f} | best: {best_val:.3f}"
                f" @ step {best_step}"
            )
            status += f" | no_improve={epochs_since_improvement}"
            if patience > 0:
                status += f"/{patience}"
            tqdm.write(status)

            if patience > 0 and epochs_since_improvement >= patience:
                stop_training = True
                tqdm.write(
                    "[early-stopping] pazienza esaurita: training interrotto per mancato"
                    " miglioramento."
                )

            if stop_training:
                break

        return {
            "best_val": best_val,
            "best_epoch": best_epoch,
            "best_step": best_step,
            "global_step": global_step,
        }
    finally:
        if wandb_run is not None and wandb_module is not None:
            try:
                wandb_module.finish()
            except Exception as exc:
                tqdm.write(f"[wandb] finish() failed: {exc}")
