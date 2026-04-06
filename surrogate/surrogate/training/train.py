"""Training loop for the FNO surrogate model.

Usage:
    cd surrogate
    uv run python -m surrogate.training.train [--epochs 150] [--batch-size 4]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from surrogate.config import CHECKPOINT_DIR, PRESET_IDS, RUNS_DIR, ModelConfig, TrainConfig
from surrogate.data.dataset import WindFieldDataset
from surrogate.model.fno3d import FNO3d
from surrogate.training.losses import surrogate_loss


def _create_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    common = dict(
        preset_ids=list(PRESET_IDS),
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        data_dir=cfg.data_dir,
    )
    train_ds = WindFieldDataset(split="train", **common)
    val_ds = WindFieldDataset(split="val", **common)
    test_ds = WindFieldDataset(split="test", **common)

    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def _train_one_epoch(
    model: FNO3d,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int = 0,
) -> dict[str, float]:
    """Train for one epoch, return averaged loss components."""
    model.train()
    accum = {"l_data": 0.0, "l_div": 0.0, "total": 0.0}
    n_batches = 0

    bar = tqdm(loader, desc=f"E{epoch:>4} train", unit="batch", leave=False, dynamic_ncols=True)

    for x, target, occ_mask in bar:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        occ_mask = occ_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            pred = model(x, occupancy=occ_mask)
            loss, components = surrogate_loss(
                pred, target, occ_mask,
                lambda_div=cfg.lambda_div,
                pedestrian_weight=cfg.pedestrian_weight,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for k in accum:
            accum[k] += components[k]
        n_batches += 1

        bar.set_postfix(loss=f"{components['total']:.4f}")

    return {k: v / max(n_batches, 1) for k, v in accum.items()}


@torch.no_grad()
def _validate(
    model: FNO3d,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int = 0,
) -> dict[str, float]:
    """Validate and return averaged loss components."""
    model.eval()
    accum = {"l_data": 0.0, "l_div": 0.0, "total": 0.0}
    n_batches = 0

    bar = tqdm(loader, desc=f"E{epoch:>4}   val", unit="batch", leave=False, dynamic_ncols=True)

    for x, target, occ_mask in bar:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        occ_mask = occ_mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            pred = model(x, occupancy=occ_mask)
            _, components = surrogate_loss(
                pred, target, occ_mask,
                lambda_div=cfg.lambda_div,
                pedestrian_weight=cfg.pedestrian_weight,
            )

        for k in accum:
            accum[k] += components[k]
        n_batches += 1

        bar.set_postfix(loss=f"{components['total']:.4f}")

    return {k: v / max(n_batches, 1) for k, v in accum.items()}


def train(cfg: TrainConfig | None = None, model_cfg: ModelConfig | None = None):
    """Main training function."""
    if cfg is None:
        cfg = TrainConfig()
    if model_cfg is None:
        model_cfg = ModelConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = _create_dataloaders(cfg)
    if len(train_loader.dataset) == 0:
        print("ERROR: No training data found. Run data generation first:")
        print("  uv run python -m surrogate.data.generate")
        return

    # Model
    model = FNO3d(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB float32)")

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return epoch / max(cfg.warmup_epochs, 1)
        progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    # Resume from checkpoint if one exists
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    resume_path = cfg.checkpoint_dir / "latest.pt"
    if resume_path.exists():
        print(f"Resuming from {resume_path} ...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_metrics"]["total"]
        print(f"  Resumed at epoch {start_epoch}, best val loss so far: {best_val_loss:.6f}")
    else:
        print("No checkpoint found — starting from scratch.")

    # Logging
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(cfg.runs_dir / f"run_{int(time.time())}"))

    print(f"\nTraining for {cfg.epochs} epochs (patience={cfg.patience})...")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'L_data':>10} {'L_div':>10} {'LR':>10} {'Time':>8}")
    print("-" * 70)

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        train_metrics = _train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch=epoch)
        val_metrics = _validate(model, val_loader, device, cfg, epoch=epoch)
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Log
        writer.add_scalar("loss/train", train_metrics["total"], epoch)
        writer.add_scalar("loss/val", val_metrics["total"], epoch)
        writer.add_scalar("loss/train_data", train_metrics["l_data"], epoch)
        writer.add_scalar("loss/train_div", train_metrics["l_div"], epoch)
        writer.add_scalar("loss/val_data", val_metrics["l_data"], epoch)
        writer.add_scalar("loss/val_div", val_metrics["l_div"], epoch)
        writer.add_scalar("lr", lr, epoch)

        is_best = val_metrics["total"] < best_val_loss
        best_marker = ""
        if is_best:
            best_val_loss = val_metrics["total"]
            patience_counter = 0
            best_marker = " ★"
        else:
            patience_counter += 1

        print(
            f"{epoch:>6d} {train_metrics['total']:>10.6f} {val_metrics['total']:>10.6f} "
            f"{val_metrics['l_data']:>10.6f} {val_metrics['l_div']:>10.6f} "
            f"{lr:>10.2e} {dt:>7.1f}s{best_marker}"
        )

        # Checkpointing
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "model_config": model_cfg.__dict__,
            "train_config": {k: v for k, v in cfg.__dict__.items() if not isinstance(v, Path)},
        }

        torch.save(checkpoint, cfg.checkpoint_dir / "latest.pt")

        if is_best:
            torch.save(checkpoint, cfg.checkpoint_dir / "best.pt")

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, cfg.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

        if patience_counter >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.patience} epochs)")
            break

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train FNO surrogate model")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32, help="FNO hidden width")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of Fourier layers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing (uses more VRAM, faster if VRAM allows)")
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_amp=not args.no_amp,
        patience=args.patience,
    )
    model_cfg = ModelConfig(width=args.width, n_layers=args.n_layers, use_checkpoint=not args.no_checkpoint)

    train(train_cfg, model_cfg)


if __name__ == "__main__":
    main()
