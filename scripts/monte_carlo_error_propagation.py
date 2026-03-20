"""
Monte Carlo Error Propagation Analysis for 3-Stage Photonic Softmax Pipeline.

Stages:
  1. Exponentiation (MRR cascade AEF) — multiplicative error epsilon_aef
  2. Summation (PD current summing) — multiplicative error epsilon_sum
  3. Normalization (electronic division) — multiplicative error epsilon_norm

Usage:
    python monte_carlo_error_propagation.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
N_TRIALS = 100_000
K_DEFAULT = 64  # attention dimension
K_SWEEP = [8, 16, 32, 64, 128]

# Stage error standard deviations (baseline)
SIGMA_AEF = 0.02    # 2%  — AEF exponentiation
SIGMA_SUM = 0.005   # 0.5% — shot noise in PD summation
SIGMA_NORM = 0.02   # 2%  — electronic normalization

# Sensitivity sweep range
SIGMA_SWEEP = np.linspace(0.005, 0.05, 10)

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DPI = 300
np.random.seed(42)


# ──────────────────────────────────────────────
# Core simulation
# ──────────────────────────────────────────────
def ideal_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    x_shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x_shifted)
    return e / e.sum(axis=-1, keepdims=True)


def noisy_softmax(x: np.ndarray, sigma_aef: float, sigma_sum: float,
                  sigma_norm: float) -> np.ndarray:
    """Simulate the 3-stage photonic pipeline with multiplicative noise."""
    x_shifted = x - x.max(axis=-1, keepdims=True)

    # Stage 1: AEF exponentiation with multiplicative error
    eps_aef = np.random.normal(0, sigma_aef, size=x_shifted.shape)
    e_noisy = np.exp(x_shifted) * (1.0 + eps_aef)
    e_noisy = np.maximum(e_noisy, 0.0)  # physical: current cannot be negative

    # Stage 2: Summation with multiplicative shot-noise error
    S_ideal = e_noisy.sum(axis=-1, keepdims=True)
    eps_sum = np.random.normal(0, sigma_sum, size=S_ideal.shape)
    S_noisy = S_ideal * (1.0 + eps_sum)

    # Stage 3: Normalization with multiplicative electronic error
    eps_norm = np.random.normal(0, sigma_norm, size=x_shifted.shape)
    p_noisy = (e_noisy / S_noisy) * (1.0 + eps_norm)

    return p_noisy


def compute_errors(x: np.ndarray, sigma_aef: float, sigma_sum: float,
                   sigma_norm: float) -> np.ndarray:
    """Return per-trial max relative error."""
    s_ideal = ideal_softmax(x)
    p_noisy = noisy_softmax(x, sigma_aef, sigma_sum, sigma_norm)

    # Relative error: max over components, per trial
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.abs(p_noisy - s_ideal) / np.maximum(s_ideal, 1e-30)
    max_rel_err = rel_err.max(axis=-1)  # shape (N_TRIALS,)
    return max_rel_err


# ──────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────
def run_baseline(K: int = K_DEFAULT):
    """Run baseline Monte Carlo with default parameters."""
    x = np.random.randn(N_TRIALS, K) * 3.0  # logits ~ N(0,3)
    errors = compute_errors(x, SIGMA_AEF, SIGMA_SUM, SIGMA_NORM)
    return errors


def run_sensitivity(K: int = K_DEFAULT):
    """Sweep each stage's error independently; return 3 x len(SIGMA_SWEEP) array."""
    results = {}
    base_sigmas = [SIGMA_AEF, SIGMA_SUM, SIGMA_NORM]
    stage_names = ["AEF (Stage 1)", "Summation (Stage 2)", "Normalization (Stage 3)"]
    x = np.random.randn(N_TRIALS, K) * 3.0

    for i, name in enumerate(stage_names):
        medians = []
        p95s = []
        for sigma in SIGMA_SWEEP:
            sigmas = list(base_sigmas)
            sigmas[i] = sigma
            errs = compute_errors(x, *sigmas)
            medians.append(float(np.median(errs)))
            p95s.append(float(np.percentile(errs, 95)))
        results[name] = {"sigma": SIGMA_SWEEP.tolist(), "median": medians, "p95": p95s}
    return results


def run_K_sweep():
    """Sweep sequence length K."""
    results = {}
    for K in K_SWEEP:
        x = np.random.randn(N_TRIALS, K) * 3.0
        errs = compute_errors(x, SIGMA_AEF, SIGMA_SUM, SIGMA_NORM)
        results[K] = {
            "mean": float(np.mean(errs)),
            "std": float(np.std(errs)),
            "p50": float(np.median(errs)),
            "p95": float(np.percentile(errs, 95)),
            "p99": float(np.percentile(errs, 99)),
        }
    return results


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_error_distribution(errors: np.ndarray, save_path: str):
    """Histogram of total error with percentile markers."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(errors, bins=200, density=True, color="#4A90D9", alpha=0.85,
            edgecolor="none", label="MC samples")

    # Percentile lines
    p50 = np.median(errors)
    p95 = np.percentile(errors, 95)
    p99 = np.percentile(errors, 99)
    first_order = SIGMA_AEF + SIGMA_SUM + SIGMA_NORM  # additive prediction

    ymax = ax.get_ylim()[1]
    ax.axvline(p50, color="#E67E22", ls="--", lw=1.5,
               label=f"Median = {p50:.3f}")
    ax.axvline(p95, color="#E74C3C", ls="--", lw=1.5,
               label=f"95th pct = {p95:.3f}")
    ax.axvline(p99, color="#8E44AD", ls="--", lw=1.5,
               label=f"99th pct = {p99:.3f}")
    ax.axvline(first_order, color="#2ECC71", ls="-.", lw=2.0,
               label=f"1st-order additive = {first_order:.3f}")

    ax.set_xlabel("Max Relative Error (per trial)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(f"Monte Carlo Error Distribution (K={K_DEFAULT}, N={N_TRIALS:,})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, min(errors.max(), 0.3))
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_sensitivity(sensitivity: dict, save_path: str):
    """Heatmap / line plot showing per-stage sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    for idx, (name, data) in enumerate(sensitivity.items()):
        sigma_pct = [s * 100 for s in data["sigma"]]
        axes[0].plot(sigma_pct, [m * 100 for m in data["median"]],
                     "-o", color=colors[idx], markersize=4, label=name)
        axes[1].plot(sigma_pct, [p * 100 for p in data["p95"]],
                     "-s", color=colors[idx], markersize=4, label=name)

    for ax, title in zip(axes, ["Median Error", "95th Percentile Error"]):
        ax.set_xlabel("Stage Error Std Dev (%)", fontsize=11)
        ax.set_ylabel("Max Relative Error (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle("Sensitivity Analysis: Per-Stage Error Contribution",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_error_vs_K(K_results: dict, save_path: str):
    """Error statistics vs sequence length K."""
    Ks = sorted(K_results.keys())
    means = [K_results[k]["mean"] * 100 for k in Ks]
    p95s = [K_results[k]["p95"] * 100 for k in Ks]
    p99s = [K_results[k]["p99"] * 100 for k in Ks]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ks, means, "-o", color="#3498DB", lw=2, markersize=6, label="Mean")
    ax.fill_between(Ks,
                    [K_results[k]["mean"] * 100 - K_results[k]["std"] * 100 for k in Ks],
                    [K_results[k]["mean"] * 100 + K_results[k]["std"] * 100 for k in Ks],
                    alpha=0.2, color="#3498DB")
    ax.plot(Ks, p95s, "-s", color="#E74C3C", lw=2, markersize=6, label="95th pct")
    ax.plot(Ks, p99s, "-^", color="#8E44AD", lw=2, markersize=6, label="99th pct")

    first_order = (SIGMA_AEF + SIGMA_SUM + SIGMA_NORM) * 100
    ax.axhline(first_order, color="#2ECC71", ls="-.", lw=2,
               label=f"1st-order additive ({first_order:.1f}%)")

    ax.set_xlabel("Sequence Length K", fontsize=12)
    ax.set_ylabel("Max Relative Error (%)", fontsize=12)
    ax.set_title("Error vs Sequence Length", fontsize=13, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 60)
    print("Monte Carlo Error Propagation - Photonic Softmax Pipeline")
    print("=" * 60)

    # 1. Baseline
    print(f"\n[1/3] Baseline simulation (K={K_DEFAULT}, N={N_TRIALS:,}) ...")
    errors = run_baseline()
    stats = {
        "K": K_DEFAULT,
        "N_trials": N_TRIALS,
        "sigma_aef": SIGMA_AEF,
        "sigma_sum": SIGMA_SUM,
        "sigma_norm": SIGMA_NORM,
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "first_order_additive": SIGMA_AEF + SIGMA_SUM + SIGMA_NORM,
    }
    print(f"  Mean error:          {stats['mean']:.4f}  ({stats['mean']*100:.2f}%)")
    print(f"  Std dev:             {stats['std']:.4f}")
    print(f"  Median:              {stats['median']:.4f}  ({stats['median']*100:.2f}%)")
    print(f"  95th percentile:     {stats['p95']:.4f}  ({stats['p95']*100:.2f}%)")
    print(f"  99th percentile:     {stats['p99']:.4f}  ({stats['p99']*100:.2f}%)")
    print(f"  1st-order additive:  {stats['first_order_additive']:.4f}  "
          f"({stats['first_order_additive']*100:.1f}%)")

    plot_error_distribution(
        errors, os.path.join(FIGURE_DIR, "fig_mc_error_distribution.png"))

    # 2. Sensitivity
    print(f"\n[2/3] Sensitivity sweep ...")
    sensitivity = run_sensitivity()
    plot_sensitivity(
        sensitivity, os.path.join(FIGURE_DIR, "fig_mc_error_sensitivity.png"))

    # 3. K sweep
    print(f"\n[3/3] Sequence-length sweep (K={K_SWEEP}) ...")
    K_results = run_K_sweep()
    for k in K_SWEEP:
        r = K_results[k]
        print(f"  K={k:>3d}: mean={r['mean']*100:.2f}%  "
              f"p95={r['p95']*100:.2f}%  p99={r['p99']*100:.2f}%")
    plot_error_vs_K(
        K_results, os.path.join(FIGURE_DIR, "fig_mc_error_vs_K.png"))

    # Save results
    all_results = {
        "baseline": stats,
        "sensitivity": sensitivity,
        "K_sweep": {str(k): v for k, v in K_results.items()},
    }
    result_path = os.path.join(RESULT_DIR, "monte_carlo_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {result_path}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
