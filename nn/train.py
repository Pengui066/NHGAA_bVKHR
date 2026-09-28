"""Training loop for the spatial reconstruction network Psi.

Usage:
    python nn/train.py --data dumps/prepared_full --out runs/psi_v1 --steps 50000
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import HairGBufferDataset, ValDataset
from losses import HairGBufferLoss
from models import SpatialNet, count_parameters


def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    sums, count = {}, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prediction = model(batch["input"])
                _, terms = criterion(prediction, batch)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + value.item()
            count += 1
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dumps/prepared_full")
    parser.add_argument("--out", default="runs/psi_v1")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--micro-batch", type=int, default=2,
                        help="micro batch per forward; VRAM peak scales with this")
    parser.add_argument("--accum", type=int, default=4,
                        help="gradient accumulation steps (effective batch = micro x accum)")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--resume", action="store_true",
                        help="continue from checkpoint.pt in --out")
    options = parser.parse_args()

    torch.manual_seed(options.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(options.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SpatialNet(base_channels=32).to(device)
    print(f"SpatialNet parameters: {count_parameters(model) / 1e6:.2f} M")

    train_set = HairGBufferDataset(options.data, split="train",
                                   patch_size=options.patch_size, preload=True)
    val_set = ValDataset(options.data, split="train")
    print(f"train samples: {len(train_set)}, val frames: {len(val_set)}")

    # num_workers=0 everywhere: worker IPC exhausts the Windows page file
    # on this machine, and the RAM-preloaded dataset needs no workers.
    train_loader = DataLoader(train_set, batch_size=options.micro_batch * options.accum,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    criterion = HairGBufferLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.lr,
                                  weight_decay=options.weight_decay)

    def lr_scale(step: int) -> float:
        if step < options.warmup:
            return step / max(options.warmup, 1)
        progress = (step - options.warmup) / max(options.steps - options.warmup, 1)
        return 0.5 * (1.0 + __import__("math").cos(3.14159 * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    start_step = 0
    if options.resume and (out_dir / "checkpoint.pt").exists():
        state = torch.load(out_dir / "checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            print("optimizer state restored")
        else:
            print("checkpoint has no optimizer state - Adam moments restart fresh")
        start_step = state["step"]
        for _ in range(start_step):
            scheduler.step()
        print(f"resumed from step {start_step}")

    csv_path = out_dir / "log.csv"
    with open(csv_path, "w", newline="") as handle:
        csv.writer(handle).writerow(
            ["step", "lr", "seconds", *["train_" + k for k in
             ["loss", "coverage", "tangent_l1", "tangent_angular", "mask"]],
             *["val_" + k for k in ["loss", "coverage", "tangent_l1", "tangent_angular", "mask"]]])

    best_val = float("inf")
    running = {}
    model.train()
    step = start_step
    oom_skips = 0
    started = time.time()

    data_iterator = iter(train_loader)
    while step < options.steps:
        step += 1
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            batch = next(data_iterator)

        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # Gradient accumulation over micro-batches: the VRAM peak (and
        # therefore the contention with the Windows desktop session) scales
        # with the micro batch, while the effective batch stays constant.
        optimizer.zero_grad(set_to_none=True)
        step_failed = False

        for m in range(options.accum):
            micro = {k: (v[m * options.micro_batch:(m + 1) * options.micro_batch]
                         if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    prediction = model(micro["input"])
                    loss, terms = criterion(prediction, micro)

                (loss / options.accum).backward()
                for key, value in terms.items():
                    running[key] = running.get(key, 0.0) + value.item() / options.accum
            except torch.OutOfMemoryError:
                # The GPU is shared with the Windows desktop: browser and
                # launcher VRAM spikes can transiently fail even small
                # allocations. Drop this step, free the cache, carry on.
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                oom_skips += 1
                step_failed = True
                print(f"step {step}: CUDA OOM - step skipped (total {oom_skips})",
                      flush=True)
                break
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                oom_skips += 1
                step_failed = True
                print(f"step {step}: raw CUDA OOM - step skipped (total {oom_skips})",
                      flush=True)
                break

        if not step_failed:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if step % options.log_every == 0:
            averages = {key: value / options.log_every for key, value in running.items()}
            running = {}
            elapsed = time.time() - started

            line = (f"step {step:6d}/{options.steps} | lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"loss {averages['loss']:.4f} (cov {averages['coverage']:.4f} "
                    f"tan {averages['tangent_l1']:.4f} ang {averages['tangent_angular']:.4f} "
                    f"msk {averages['mask']:.4f}) | {elapsed:.0f}s")
            print(line, flush=True)

            val_terms = {f"val_{k}": "" for k in averages}
            if step % options.val_every == 0:
                val_averages = evaluate(model, val_loader, criterion, device)
                if val_averages:
                    line += " | VAL " + " ".join(f"{k} {v:.4f}" for k, v in val_averages.items())
                    print(line, flush=True)
                    if val_averages["loss"] < best_val:
                        best_val = val_averages["loss"]
                        torch.save({"model": model.state_dict(), "step": step,
                                    "val": val_averages}, out_dir / "best.pt")
                    val_terms = {f"val_{k}": v for k, v in val_averages.items()}

            with open(csv_path, "a", newline="") as handle:
                csv.writer(handle).writerow(
                    [step, f"{scheduler.get_last_lr()[0]:.2e}", f"{elapsed:.0f}",
                     *[f"{averages[k]:.5f}" for k in
                       ["loss", "coverage", "tangent_l1", "tangent_angular", "mask"]],
                     *[val_terms.get(f"val_{k}", "") for k in
                       ["loss", "coverage", "tangent_l1", "tangent_angular", "mask"]]])

        if step % options.checkpoint_every == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step}, out_dir / "checkpoint.pt")

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": options.steps}, out_dir / "final.pt")
    print(f"training done, best val loss {best_val:.4f}, saved to {out_dir}")


if __name__ == "__main__":
    main()
