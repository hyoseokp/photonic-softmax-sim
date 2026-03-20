"""
GPT-scale energy extrapolation for photonic softmax accelerator.

Computes MAC counts, photonic vs GPU energy, and component counts
for Transformer models from BERT-base to Llama-2 7B scale.

Generates publication-quality figures and LaTeX table.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json

# ── Output directories ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
RES_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ── Model definitions ───────────────────────────────────────────────
MODELS = [
    {"name": "BERT-base",     "d_model": 768,  "L": 12, "h": 12, "d_k": 64,  "d_ff": 3072,  "K": 128},
    {"name": "GPT-2",         "d_model": 768,  "L": 12, "h": 12, "d_k": 64,  "d_ff": 3072,  "K": 1024},
    {"name": "GPT-2 Medium",  "d_model": 1024, "L": 24, "h": 16, "d_k": 64,  "d_ff": 4096,  "K": 1024},
    {"name": "GPT-2 Large",   "d_model": 1280, "L": 36, "h": 20, "d_k": 64,  "d_ff": 5120,  "K": 1024},
    {"name": "GPT-3 (approx)","d_model": 4096, "L": 32, "h": 32, "d_k": 128, "d_ff": 16384, "K": 2048},
    {"name": "Llama-2 7B",    "d_model": 4096, "L": 32, "h": 32, "d_k": 128, "d_ff": 11008, "K": 4096},
]

# ── Energy parameters ───────────────────────────────────────────────
E_MAC_PHOTONIC_FJ = [10, 50]   # fJ/MAC range
E_ADC_PJ = 0.5                 # pJ per conversion
E_DAC_PJ = 0.5                 # pJ per conversion

E_GPU_H100_PJ = 0.71           # pJ/MAC
E_GPU_B200_PJ = 0.44           # pJ/MAC

N_RINGS_PER_CASCADE = 12       # MRR rings per exponential cascade


def compute_macs_per_token(m):
    """Compute total MACs per token for one forward pass."""
    d, L, h, d_k, d_ff, K = m["d_model"], m["L"], m["h"], m["d_k"], m["d_ff"], m["K"]

    # Per-layer MAC breakdown
    qkv_proj = 3 * d * d_k * h          # = 3 * d^2
    attn_score = K * d_k * h             # Q @ K^T
    attn_value = K * d_k * h             # attn @ V
    out_proj = d * d                     # = d^2
    ffn = 2 * d * d_ff                   # two linear layers

    per_layer = qkv_proj + attn_score + attn_value + out_proj + ffn
    total = per_layer * L

    return {
        "qkv_proj": qkv_proj,
        "attn_score": attn_score,
        "attn_value": attn_value,
        "out_proj": out_proj,
        "ffn": ffn,
        "per_layer": per_layer,
        "total": total,
    }


def compute_energy(m, macs):
    """Compute photonic and GPU energy per token."""
    d, L, h, d_k, d_ff, K = m["d_model"], m["L"], m["h"], m["d_k"], m["d_ff"], m["K"]
    total_macs = macs["total"]

    # Photonic: optical core + bridge (ADC/DAC)
    e_opt_low = total_macs * E_MAC_PHOTONIC_FJ[0] * 1e-15   # Joules
    e_opt_high = total_macs * E_MAC_PHOTONIC_FJ[1] * 1e-15

    # Bridge: ADC+DAC at every layer boundary for attention inputs/outputs
    # 2 conversions (DAC in, ADC out) × L layers × K tokens × d_model channels
    n_conversions = 2 * L * K * d
    e_bridge = n_conversions * (E_ADC_PJ + E_DAC_PJ) * 0.5 * 1e-12  # each side 0.5 pJ

    # Actually: E_bridge = 2 * L * (K * d_model * E_adc + K * d_model * E_dac)
    e_bridge = 2 * L * K * d * (E_ADC_PJ + E_DAC_PJ) * 1e-12  # Joules

    e_photonic_low = e_opt_low + e_bridge
    e_photonic_high = e_opt_high + e_bridge

    # GPU energy
    e_h100 = total_macs * E_GPU_H100_PJ * 1e-12
    e_b200 = total_macs * E_GPU_B200_PJ * 1e-12

    # Advantage ratios (GPU / photonic)
    adv_h100_low = e_h100 / e_photonic_low if e_photonic_low > 0 else 0
    adv_h100_high = e_h100 / e_photonic_high if e_photonic_high > 0 else 0
    adv_b200_low = e_b200 / e_photonic_low if e_photonic_low > 0 else 0
    adv_b200_high = e_b200 / e_photonic_high if e_photonic_high > 0 else 0

    return {
        "e_opt_low_J": e_opt_low,
        "e_opt_high_J": e_opt_high,
        "e_bridge_J": e_bridge,
        "e_photonic_low_J": e_photonic_low,
        "e_photonic_high_J": e_photonic_high,
        "e_h100_J": e_h100,
        "e_b200_J": e_b200,
        "adv_h100_low": adv_h100_low,
        "adv_h100_high": adv_h100_high,
        "adv_b200_low": adv_b200_low,
        "adv_b200_high": adv_b200_high,
    }


def compute_components(m):
    """Compute photonic component counts."""
    d, L, h, d_k, K = m["d_model"], m["L"], m["h"], m["d_k"], m["K"]

    mrr_softmax = h * K * N_RINGS_PER_CASCADE   # MRR cascades for softmax
    mrr_weight_bank = d_k * d                    # MRR weight bank per head (for MVM)
    mrr_total = (mrr_softmax + mrr_weight_bank) * L  # total across layers (simplified)

    return {
        "mrr_softmax_per_layer": mrr_softmax,
        "mrr_weight_bank_per_head": mrr_weight_bank,
        "mrr_total": mrr_total,
        "mrr_softmax_total": mrr_softmax * L,
    }


def to_uJ(joules):
    """Convert Joules to microJoules."""
    return joules * 1e6


# ── Main computation ─────────────────────────────────────────────────
results = []
for m in MODELS:
    macs = compute_macs_per_token(m)
    energy = compute_energy(m, macs)
    components = compute_components(m)
    results.append({
        "model": m,
        "macs": macs,
        "energy": energy,
        "components": components,
    })

# Print summary table
print(f"{'Model':<16} {'MACs/token':>14} {'Photonic(10fJ)':>14} {'Photonic(50fJ)':>14} "
      f"{'H100':>12} {'B200':>12} {'Adv(H100)':>12} {'MRR_soft':>12}")
print("=" * 120)
for r in results:
    m = r["model"]
    macs = r["macs"]["total"]
    e = r["energy"]
    c = r["components"]
    print(f"{m['name']:<16} {macs:>14,.0f} "
          f"{to_uJ(e['e_photonic_low_J']):>12.2f}uJ "
          f"{to_uJ(e['e_photonic_high_J']):>12.2f}uJ "
          f"{to_uJ(e['e_h100_J']):>10.2f}uJ "
          f"{to_uJ(e['e_b200_J']):>10.2f}uJ "
          f"{e['adv_h100_low']:>10.1f}x "
          f"{c['mrr_softmax_total']:>12,}")


# ── Figure 1: Energy scaling bar chart ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

names = [r["model"]["name"] for r in results]
x = np.arange(len(names))
width = 0.2

# Use geometric mean of photonic range for the bar
e_phot_low = [to_uJ(r["energy"]["e_photonic_low_J"]) for r in results]
e_phot_high = [to_uJ(r["energy"]["e_photonic_high_J"]) for r in results]
e_phot_mid = [np.sqrt(lo * hi) for lo, hi in zip(e_phot_low, e_phot_high)]
e_phot_err_lo = [mid - lo for mid, lo in zip(e_phot_mid, e_phot_low)]
e_phot_err_hi = [hi - mid for mid, hi in zip(e_phot_mid, e_phot_high)]

e_h100 = [to_uJ(r["energy"]["e_h100_J"]) for r in results]
e_b200 = [to_uJ(r["energy"]["e_b200_J"]) for r in results]

bars1 = ax.bar(x - width, e_phot_mid, width, label="Photonic (10–50 fJ/MAC)",
               color="#2196F3", edgecolor="black", linewidth=0.5,
               yerr=[e_phot_err_lo, e_phot_err_hi], capsize=3, error_kw={"linewidth": 1})
bars2 = ax.bar(x, e_h100, width, label="H100 (0.71 pJ/MAC)",
               color="#FF5722", edgecolor="black", linewidth=0.5)
bars3 = ax.bar(x + width, e_b200, width, label="B200 (0.44 pJ/MAC)",
               color="#FFC107", edgecolor="black", linewidth=0.5)

ax.set_yscale("log")
ax.set_ylabel("Energy per Token (µJ)", fontsize=13)
ax.set_xlabel("Model", fontsize=13)
ax.set_title("Photonic vs. GPU Energy per Token", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", alpha=0.3, which="both")
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_energy_scaling.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig_energy_scaling.pdf"), bbox_inches="tight")
print(f"\nSaved fig_energy_scaling.png/pdf")
plt.close()


# ── Figure 2: Advantage ratio vs model size ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

total_params_proxy = [r["macs"]["total"] for r in results]  # MACs as size proxy

adv_h100_lo = [r["energy"]["adv_h100_high"] for r in results]  # high fJ → low advantage
adv_h100_hi = [r["energy"]["adv_h100_low"] for r in results]   # low fJ → high advantage
adv_b200_lo = [r["energy"]["adv_b200_high"] for r in results]
adv_b200_hi = [r["energy"]["adv_b200_low"] for r in results]

ax.fill_between(range(len(names)), adv_h100_lo, adv_h100_hi, alpha=0.25, color="#FF5722")
ax.plot(range(len(names)), [(lo+hi)/2 for lo, hi in zip(adv_h100_lo, adv_h100_hi)],
        "o-", color="#FF5722", linewidth=2, markersize=8, label="vs. H100")

ax.fill_between(range(len(names)), adv_b200_lo, adv_b200_hi, alpha=0.25, color="#FFC107")
ax.plot(range(len(names)), [(lo+hi)/2 for lo, hi in zip(adv_b200_lo, adv_b200_hi)],
        "s-", color="#E65100", linewidth=2, markersize=8, label="vs. B200")

ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, label="Break-even")

ax.set_ylabel("Energy Advantage Ratio (GPU / Photonic)", fontsize=13)
ax.set_xlabel("Model", fontsize=13)
ax.set_title("Photonic Energy Advantage vs. Model Scale", fontsize=15, fontweight="bold")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_advantage_vs_model_size.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig_advantage_vs_model_size.pdf"), bbox_inches="tight")
print(f"Saved fig_advantage_vs_model_size.png/pdf")
plt.close()


# ── Figure 3: Component count scaling ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

mrr_soft = [r["components"]["mrr_softmax_total"] for r in results]
mrr_wb = [r["components"]["mrr_weight_bank_per_head"] * r["model"]["L"] for r in results]

x = np.arange(len(names))
width = 0.3

ax.bar(x - width/2, mrr_soft, width, label="MRR (Softmax cascades)",
       color="#4CAF50", edgecolor="black", linewidth=0.5)
ax.bar(x + width/2, mrr_wb, width, label="MRR (Weight bank per layer stack)",
       color="#9C27B0", edgecolor="black", linewidth=0.5)

ax.set_yscale("log")
ax.set_ylabel("Component Count", fontsize=13)
ax.set_xlabel("Model", fontsize=13)
ax.set_title("Photonic Component Count vs. Model Scale", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, which="both")
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_component_count_scaling.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig_component_count_scaling.pdf"), bbox_inches="tight")
print(f"Saved fig_component_count_scaling.png/pdf")
plt.close()


# ── LaTeX table ──────────────────────────────────────────────────────
tex_lines = []
tex_lines.append(r"\begin{table*}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Energy-per-token comparison: photonic softmax accelerator vs.\ GPU baselines across Transformer model scales.}")
tex_lines.append(r"\label{tab:energy_scaling}")
tex_lines.append(r"\small")
tex_lines.append(r"\begin{tabular}{l r r r r r r r}")
tex_lines.append(r"\hline")
tex_lines.append(r"\textbf{Model} & \textbf{$d_\text{model}$} & \textbf{MACs/token} & "
                 r"\textbf{Photonic (10\,fJ)} & \textbf{Photonic (50\,fJ)} & "
                 r"\textbf{H100} & \textbf{B200} & \textbf{Adv.\ (H100)} \\")
tex_lines.append(r" & & & (\si{\micro\joule}) & (\si{\micro\joule}) & "
                 r"(\si{\micro\joule}) & (\si{\micro\joule}) & \\")
tex_lines.append(r"\hline")

for r in results:
    m = r["model"]
    macs = r["macs"]["total"]
    e = r["energy"]
    name_tex = m["name"].replace("(approx)", r"{\scriptsize(approx)}")

    # Format MACs
    if macs >= 1e9:
        mac_str = f"{macs/1e9:.1f}G"
    elif macs >= 1e6:
        mac_str = f"{macs/1e6:.0f}M"
    else:
        mac_str = f"{macs/1e3:.0f}K"

    adv_lo = r["energy"]["adv_h100_high"]  # conservative (50fJ)
    adv_hi = r["energy"]["adv_h100_low"]   # optimistic (10fJ)

    tex_lines.append(
        f"  {name_tex} & {m['d_model']} & {mac_str} & "
        f"{to_uJ(e['e_photonic_low_J']):.2f} & "
        f"{to_uJ(e['e_photonic_high_J']):.2f} & "
        f"{to_uJ(e['e_h100_J']):.2f} & "
        f"{to_uJ(e['e_b200_J']):.2f} & "
        f"{adv_lo:.1f}--{adv_hi:.1f}$\\times$ \\\\"
    )

tex_lines.append(r"\hline")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table*}")

tex_content = "\n".join(tex_lines)

with open(os.path.join(RES_DIR, "energy_scaling_table.tex"), "w") as f:
    f.write(tex_content)
print(f"\nSaved energy_scaling_table.tex")

# ── Save JSON results ───────────────────────────────────────────────
json_results = []
for r in results:
    jr = {
        "model": r["model"]["name"],
        "d_model": r["model"]["d_model"],
        "L": r["model"]["L"],
        "h": r["model"]["h"],
        "d_k": r["model"]["d_k"],
        "d_ff": r["model"]["d_ff"],
        "K": r["model"]["K"],
        "total_macs": r["macs"]["total"],
        "energy_photonic_10fJ_uJ": to_uJ(r["energy"]["e_photonic_low_J"]),
        "energy_photonic_50fJ_uJ": to_uJ(r["energy"]["e_photonic_high_J"]),
        "energy_h100_uJ": to_uJ(r["energy"]["e_h100_J"]),
        "energy_b200_uJ": to_uJ(r["energy"]["e_b200_J"]),
        "advantage_h100_range": [r["energy"]["adv_h100_high"], r["energy"]["adv_h100_low"]],
        "advantage_b200_range": [r["energy"]["adv_b200_high"], r["energy"]["adv_b200_low"]],
        "mrr_softmax_total": r["components"]["mrr_softmax_total"],
        "mrr_weight_bank_per_head": r["components"]["mrr_weight_bank_per_head"],
    }
    json_results.append(jr)

with open(os.path.join(RES_DIR, "energy_scaling_results.json"), "w") as f:
    json.dump(json_results, f, indent=2)
print(f"Saved energy_scaling_results.json")

print("\nDone.")
