r"""
Noise-Injected BERT Evaluation for Photonic Softmax Validation

Evaluates pretrained DistilBERT on SST-2 sentiment classification
with multiplicative noise injected into the softmax (simulating
photonic exponential approximation error).

Usage:
    C:\anaconda3\python.exe scripts\noise_injected_bert.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(r"C:\Users\연구실\photonic-softmax-sim")
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Noisy softmax (simulates photonic exp error) ────────────────────
def noisy_softmax(logits, noise_level):
    """Softmax with multiplicative Gaussian noise on exp values.

    Models photonic analog computing error where the exponential
    function has a relative error of ~noise_level.
    """
    max_val = logits.max(dim=-1, keepdim=True).values
    exp_vals = torch.exp(logits - max_val)
    if noise_level > 0:
        noise = 1.0 + noise_level * torch.randn_like(exp_vals)
        exp_vals = exp_vals * noise
        exp_vals = exp_vals.clamp(min=1e-10)
    return exp_vals / exp_vals.sum(dim=-1, keepdim=True)


# ── Global softmax hook ─────────────────────────────────────────────
_current_noise_level = 0.0
_original_softmax = F.softmax

def _hooked_softmax(input, dim=None, _stacklevel=3, dtype=None):
    """Drop-in replacement for F.softmax that injects multiplicative noise."""
    if _current_noise_level > 0:
        max_val = input.max(dim=dim, keepdim=True).values
        exp_vals = torch.exp(input - max_val)
        noise = 1.0 + _current_noise_level * torch.randn_like(exp_vals)
        exp_vals = (exp_vals * noise).clamp(min=1e-10)
        if dtype is not None:
            exp_vals = exp_vals.to(dtype)
        return exp_vals / exp_vals.sum(dim=dim, keepdim=True)
    return _original_softmax(input, dim=dim, dtype=dtype)

def enable_noisy_softmax():
    """Globally replace F.softmax with noisy version."""
    F.softmax = _hooked_softmax
    # Also patch torch.softmax which some code paths use
    torch.softmax = _hooked_softmax

def disable_noisy_softmax():
    """Restore original F.softmax."""
    F.softmax = _original_softmax
    torch.softmax = _original_softmax


# ── KL divergence analysis (synthetic) ──────────────────────────────
def compute_kl_divergence_sweep(noise_levels, n_samples=5000, seq_len=64):
    """Compute KL divergence between clean and noisy softmax
    on synthetic attention score distributions."""
    results = {}
    torch.manual_seed(42)

    for nl in noise_levels:
        kl_vals = []
        for _ in range(n_samples):
            # Synthetic attention scores ~ N(0, 1) like typical QK^T / sqrt(d)
            logits = torch.randn(1, 8, seq_len, seq_len, device=DEVICE)

            with torch.no_grad():
                clean = F.softmax(logits, dim=-1)
                noisy = noisy_softmax(logits, nl)

                # KL(clean || noisy) per head, averaged
                kl = F.kl_div(noisy.log(), clean, reduction='none').sum(dim=-1).mean()
                kl_vals.append(kl.item())

        results[nl] = {
            'mean': float(np.mean(kl_vals)),
            'std': float(np.std(kl_vals)),
            'median': float(np.median(kl_vals)),
            'p95': float(np.percentile(kl_vals, 95)),
        }
        print(f"  noise={nl:5.1%}  KL={results[nl]['mean']:.6f} ± {results[nl]['std']:.6f}")

    return results


# ── BERT evaluation ─────────────────────────────────────────────────
def evaluate_bert_with_noise(noise_levels, n_eval_samples=1000, n_trials=3):
    """Evaluate DistilBERT-SST2 accuracy across noise levels."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset

    print("\n── Loading model and data ──")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    # Enable global softmax hook
    enable_noisy_softmax()

    # Load SST-2 validation set
    dataset = load_dataset("glue", "sst2", split="validation")
    if n_eval_samples and n_eval_samples < len(dataset):
        dataset = dataset.select(range(n_eval_samples))

    n_total = len(dataset)
    print(f"  Model: {model_name}")
    print(f"  Eval samples: {n_total}")

    # Tokenize
    texts = dataset["sentence"]
    labels = torch.tensor(dataset["label"], device=DEVICE)

    # Batch evaluation
    batch_size = 64
    results = {}

    for nl in noise_levels:
        global _current_noise_level
        _current_noise_level = nl

        trial_accs = []
        trial_f1s = []

        for trial in range(n_trials):
            all_preds = []

            for i in range(0, n_total, batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    outputs = model(**inputs)
                    preds = outputs.logits.argmax(dim=-1)
                    all_preds.append(preds)

            all_preds = torch.cat(all_preds)
            correct = (all_preds == labels).float()
            acc = correct.mean().item()

            # F1 score
            tp = ((all_preds == 1) & (labels == 1)).sum().float()
            fp = ((all_preds == 1) & (labels == 0)).sum().float()
            fn = ((all_preds == 0) & (labels == 1)).sum().float()
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            trial_accs.append(acc)
            trial_f1s.append(f1.item())

        results[nl] = {
            'accuracy_mean': float(np.mean(trial_accs)),
            'accuracy_std': float(np.std(trial_accs)),
            'f1_mean': float(np.mean(trial_f1s)),
            'f1_std': float(np.std(trial_f1s)),
        }
        print(f"  noise={nl:5.1%}  acc={results[nl]['accuracy_mean']:.4f} ± {results[nl]['accuracy_std']:.4f}"
              f"  F1={results[nl]['f1_mean']:.4f} ± {results[nl]['f1_std']:.4f}")

    _current_noise_level = 0.0
    disable_noisy_softmax()
    return results


# ── Plotting ────────────────────────────────────────────────────────
def plot_accuracy_vs_noise(results, save_path):
    """Publication-quality accuracy & F1 vs noise level plot."""
    noise_pct = sorted(results.keys())
    acc_mean = [results[n]['accuracy_mean'] * 100 for n in noise_pct]
    acc_std = [results[n]['accuracy_std'] * 100 for n in noise_pct]
    f1_mean = [results[n]['f1_mean'] * 100 for n in noise_pct]
    f1_std = [results[n]['f1_std'] * 100 for n in noise_pct]
    x = [n * 100 for n in noise_pct]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(x, acc_mean, yerr=acc_std, fmt='o-', color='#2171B5',
                capsize=4, capthick=1.5, linewidth=2, markersize=7,
                label='Accuracy', zorder=3)
    ax.errorbar(x, f1_mean, yerr=f1_std, fmt='s--', color='#CB181D',
                capsize=4, capthick=1.5, linewidth=2, markersize=7,
                label='F1 Score', zorder=3)

    # Highlight the 2-5% noise region (our photonic device range)
    ax.axvspan(2, 5, alpha=0.12, color='green', zorder=0,
               label='Photonic device range (2–5%)')

    ax.set_xlabel('Softmax Noise Level (%)', fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('DistilBERT SST-2 Robustness to Softmax Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.set_xlim(-0.5, max(x) + 0.5)
    ax.set_ylim(max(0, min(acc_mean + f1_mean) - 8), 100.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_kl_divergence(kl_results, save_path):
    """Publication-quality KL divergence vs noise level plot."""
    noise_pct = sorted(kl_results.keys())
    kl_mean = [kl_results[n]['mean'] for n in noise_pct]
    kl_std = [kl_results[n]['std'] for n in noise_pct]
    kl_p95 = [kl_results[n]['p95'] for n in noise_pct]
    x = [n * 100 for n in noise_pct]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(x, kl_mean, yerr=kl_std, fmt='o-', color='#2171B5',
                capsize=4, capthick=1.5, linewidth=2, markersize=7,
                label='Mean KL divergence', zorder=3)
    ax.plot(x, kl_p95, 's--', color='#CB181D', linewidth=2, markersize=7,
            label='95th percentile', zorder=3)

    # Highlight the 2-5% noise region
    ax.axvspan(2, 5, alpha=0.12, color='green', zorder=0,
               label='Photonic device range (2–5%)')

    ax.set_xlabel('Softmax Noise Level (%)', fontsize=13)
    ax.set_ylabel('KL Divergence (nats)', fontsize=13)
    ax.set_title('Attention Distribution Distortion from Softmax Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(-0.5, max(x) + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ────────────────────────────────────────────────────────────
def main():
    noise_levels = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

    print("=" * 60)
    print("Photonic Softmax Noise: BERT Robustness Evaluation")
    print("=" * 60)

    # Part 1: KL divergence analysis (fast, synthetic)
    print("\n── KL Divergence Analysis (synthetic attention scores) ──")
    kl_results = compute_kl_divergence_sweep(noise_levels)

    # Part 2: BERT accuracy evaluation
    print("\n── DistilBERT SST-2 Evaluation ──")
    t0 = time.time()
    bert_results = evaluate_bert_with_noise(noise_levels, n_eval_samples=872, n_trials=3)
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    # Part 3: Generate figures
    print("\n── Generating Figures ──")
    plot_accuracy_vs_noise(bert_results, FIG_DIR / "fig_bert_accuracy_vs_noise.png")
    plot_kl_divergence(kl_results, FIG_DIR / "fig_attention_kl_divergence.png")

    # Part 4: Save results
    all_results = {
        'model': 'distilbert-base-uncased-finetuned-sst-2-english',
        'dataset': 'SST-2 (GLUE) validation',
        'n_eval_samples': 872,
        'n_trials': 3,
        'noise_levels': noise_levels,
        'bert_results': {str(k): v for k, v in bert_results.items()},
        'kl_divergence': {str(k): v for k, v in kl_results.items()},
        'device': str(DEVICE),
    }
    out_path = RES_DIR / "bert_noise_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # Summary
    print("\n── Summary ──")
    baseline_acc = bert_results[0.0]['accuracy_mean'] * 100
    for nl in [0.02, 0.05, 0.10, 0.20]:
        acc = bert_results[nl]['accuracy_mean'] * 100
        drop = baseline_acc - acc
        print(f"  {nl:5.0%} noise: {acc:.1f}% accuracy (Δ = {drop:+.1f}%)")


if __name__ == "__main__":
    main()
