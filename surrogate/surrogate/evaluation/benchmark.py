"""Benchmark: compare surrogate predictions against LBM ground truth.

Measures accuracy metrics and inference latency for both the FNO surrogate
and the CUDA LBM solver.

Usage:
    cd surrogate
    uv run python -m surrogate.evaluation.benchmark [--preset chicago] [--n-samples 20]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from surrogate.config import CHECKPOINT_DIR, DEFAULT_ROUGHNESS, PRESET_IDS
from surrogate.evaluation.metrics import (
    comfort_mae,
    lawson_agreement,
    max_pointwise_error,
    mean_abs_divergence,
    mse,
    relative_l2,
)


def _run_surrogate(predictor, preset_id: str, direction: float, speed: float):
    """Run surrogate prediction and return (vel, comfort, domain, elapsed_ms)."""
    t0 = time.perf_counter()
    vel, comfort, domain = predictor.predict(preset_id, direction, speed)
    elapsed = (time.perf_counter() - t0) * 1000
    return vel, comfort, domain, elapsed


def _run_lbm(preset_id: str, direction: float, speed: float):
    """Run LBM simulation and return (vel, comfort, domain, elapsed_ms)."""
    from lbm.run_cuda import run_cuda_simulation

    t0 = time.perf_counter()
    vel, comfort, domain = run_cuda_simulation(
        preset_id=preset_id,
        direction=direction,
        speed=speed,
        roughness=DEFAULT_ROUGHNESS,
        resolution=8.0,
        num_steps=500,
        snapshot_interval=0,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return vel, comfort, domain, elapsed


def benchmark(
    preset_ids: list[str] | None = None,
    n_samples: int = 20,
    checkpoint_path: str | None = None,
    run_lbm: bool = True,
    output_json: str | None = None,
):
    """Run benchmark comparison.

    Args:
        preset_ids: List of presets to test (default: all).
        n_samples: Number of random parameter combinations to test per preset.
        checkpoint_path: Path to model checkpoint (default: best.pt).
        run_lbm: Whether to also run LBM for latency comparison (slow).
        output_json: Optional path to save raw results as JSON.
    """
    from surrogate.inference.predictor import WindSurrogate

    if preset_ids is None:
        preset_ids = list(PRESET_IDS)

    cp = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_DIR / "best.pt"
    predictor = WindSurrogate(checkpoint_path=cp)

    rng = np.random.default_rng(42)
    results = []

    for pid in preset_ids:
        print(f"\n{'='*60}")
        print(f"Benchmarking preset: {pid}")
        print(f"{'='*60}")

        # Generate random test parameters
        directions = rng.uniform(0, 360, n_samples)
        speeds = rng.uniform(1, 15, n_samples)

        surr_latencies = []
        lbm_latencies = []
        metrics_accum = {
            "rel_l2": [], "mse": [], "max_err": [],
            "comfort_mae": [], "lawson_agree": [],
            "div_surr": [], "div_lbm": [],
        }

        for i in range(n_samples):
            d, s = float(directions[i]), float(speeds[i])
            print(f"\n  Sample {i+1}/{n_samples}: dir={d:.0f}, speed={s:.1f}, z0={DEFAULT_ROUGHNESS:.3f}")

            # Surrogate
            vel_surr, _, _, surr_ms = _run_surrogate(predictor, pid, d, s)
            surr_latencies.append(surr_ms)
            print(f"    Surrogate: {surr_ms:.1f} ms")

            if run_lbm:
                try:
                    vel_lbm, _, _, lbm_ms = _run_lbm(pid, d, s)
                    lbm_latencies.append(lbm_ms)
                    print(f"    LBM:       {lbm_ms:.1f} ms")

                    # Compute metrics (match shapes — LBM might differ slightly)
                    min_shape = tuple(min(a, b) for a, b in zip(vel_surr.shape[:3], vel_lbm.shape[:3]))
                    vs = vel_surr[:min_shape[0], :min_shape[1], :min_shape[2]]
                    vl = vel_lbm[:min_shape[0], :min_shape[1], :min_shape[2]]

                    rl2 = relative_l2(vs, vl)
                    m = mse(vs, vl)
                    me = max_pointwise_error(vs, vl)
                    cm = comfort_mae(vs, vl)
                    la = lawson_agreement(vs, vl)
                    ds = mean_abs_divergence(vs)
                    dl = mean_abs_divergence(vl)

                    metrics_accum["rel_l2"].append(rl2)
                    metrics_accum["mse"].append(m)
                    metrics_accum["max_err"].append(me)
                    metrics_accum["comfort_mae"].append(cm)
                    metrics_accum["lawson_agree"].append(la)
                    metrics_accum["div_surr"].append(ds)
                    metrics_accum["div_lbm"].append(dl)

                    print(f"    Rel L2: {rl2:.4f}, MSE: {m:.6f}, Comfort MAE: {cm:.3f} m/s, Lawson: {la:.1%}")
                except Exception as e:
                    print(f"    LBM failed: {e}")

        # Summary for this preset
        print(f"\n--- Summary: {pid} ---")
        print(f"  Surrogate latency: {np.mean(surr_latencies):.1f} +/- {np.std(surr_latencies):.1f} ms")
        if lbm_latencies:
            print(f"  LBM latency:       {np.mean(lbm_latencies):.0f} +/- {np.std(lbm_latencies):.0f} ms")
            print(f"  Speedup:           {np.mean(lbm_latencies) / np.mean(surr_latencies):.0f}x")

        if metrics_accum["rel_l2"]:
            print(f"  Relative L2:       {np.mean(metrics_accum['rel_l2']):.4f} +/- {np.std(metrics_accum['rel_l2']):.4f}")
            print(f"  MSE:               {np.mean(metrics_accum['mse']):.6f}")
            print(f"  Max error:         {np.mean(metrics_accum['max_err']):.3f} m/s")
            print(f"  Comfort MAE:       {np.mean(metrics_accum['comfort_mae']):.3f} m/s")
            print(f"  Lawson agreement:  {np.mean(metrics_accum['lawson_agree']):.1%}")
            print(f"  Divergence (surr): {np.mean(metrics_accum['div_surr']):.6f}")
            print(f"  Divergence (LBM):  {np.mean(metrics_accum['div_lbm']):.6f}")

        results.append({
            "preset": pid,
            "n_samples": n_samples,
            "surrogate_latency_ms": {"mean": np.mean(surr_latencies), "std": np.std(surr_latencies)},
            "lbm_latency_ms": {"mean": np.mean(lbm_latencies), "std": np.std(lbm_latencies)} if lbm_latencies else None,
            "metrics": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in metrics_accum.items() if v},
        })

    if output_json:
        Path(output_json).write_text(json.dumps(results, indent=2, default=float))
        print(f"\nRaw results saved to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark FNO surrogate vs LBM")
    parser.add_argument("--preset", type=str, default=None, help="Single preset (default: all)")
    parser.add_argument("--n-samples", type=int, default=20, help="Test samples per preset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--no-lbm", action="store_true", help="Skip LBM runs (surrogate-only latency)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    presets = [args.preset] if args.preset else None
    benchmark(
        preset_ids=presets,
        n_samples=args.n_samples,
        checkpoint_path=args.checkpoint,
        run_lbm=not args.no_lbm,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
