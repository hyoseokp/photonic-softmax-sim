"""
Ring Uniformity Perturbation Analysis for MRR Cascade AEF.

Analyzes how fabrication-induced resonance scatter (non-identical ring
detunings) degrades the exponential approximation quality of a cascaded
microring resonator approximate exponential function (AEF).

Reference: Park & Park, "From Microring Cascades to Optical Transformers"
Transfer function: T(I) = C * prod_{k=1}^{N} 1/(1 + (a_k + b*I)^2)
Ideal case: all a_k = a (identical rings)

Physical mapping:
  sigma_physical (nm) = sigma / b_V * FWHM
  where b_V = 0.146 V^-1, FWHM = 0.124 nm (124 pm)
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
import json
import os

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Physical parameters ──────────────────────────────────────────────
N = 12          # cascade depth
L = 8.0         # interval length
b_V = 0.146     # V^-1, EO sensitivity (FDTD Q_L=12500)
FWHM_nm = 0.124 # nm (124 pm linewidth)

N_MC = 1000     # Monte Carlo trials per sigma
N_SIGMA = 21    # number of sigma steps (including 0)
N_GRID = 2001   # evaluation grid points on [0, L]

SIGMA_MAX = 0.5

# Output paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
RES_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


# ── Transfer function ────────────────────────────────────────────────
def log_transfer(I_arr, a_arr, b, ln_C):
    """Log of cascade transfer: ln C - sum_k ln(1 + (a_k + b*I)^2)."""
    # a_arr shape: (N,) or (N_trials, N)
    # I_arr shape: (N_grid,)
    if a_arr.ndim == 1:
        # (N, N_grid)
        terms = np.log1p((a_arr[:, None] + b * I_arr[None, :]) ** 2)
        return ln_C - terms.sum(axis=0)
    else:
        # a_arr: (N_trials, N), output: (N_trials, N_grid)
        # (N_trials, N, N_grid)
        terms = np.log1p((a_arr[:, :, None] + b * I_arr[None, None, :]) ** 2)
        return ln_C - terms.sum(axis=1)


def minimax_optimal_lnC(log_y, I_arr, L):
    """Compute minimax-optimal ln C given log_y = ln(transfer without C)."""
    # g(I) = log_y(I) - (I - L)  (without ln C)
    # minimax optimal: ln C* = -(max g + min g) / 2
    target = I_arr - L  # target is I - L
    if log_y.ndim == 1:
        g = log_y - target
        ln_C_star = -(g.max() + g.min()) / 2
    else:
        g = log_y - target[None, :]
        ln_C_star = -(g.max(axis=1) + g.min(axis=1)) / 2
    return ln_C_star


def max_log_error(I_arr, a, b, L, N_rings):
    """Compute E_inf = max |r(I)| for identical detuning."""
    a_arr = np.full(N_rings, a)
    log_y_no_C = log_transfer(I_arr, a_arr, b, 0.0)
    ln_C = minimax_optimal_lnC(log_y_no_C, I_arr, L)
    r = ln_C + log_y_no_C - (I_arr - L)
    return np.max(np.abs(r))


def max_rel_error(I_arr, a, b, L, N_rings):
    """Compute max |exp(r(I)) - 1| = max relative error."""
    a_arr = np.full(N_rings, a)
    log_y_no_C = log_transfer(I_arr, a_arr, b, 0.0)
    ln_C = minimax_optimal_lnC(log_y_no_C, I_arr, L)
    r = ln_C + log_y_no_C - (I_arr - L)
    return np.max(np.abs(np.exp(r) - 1.0))


# ── Step 1: Fit ideal cascade (identical detuning) ────────────────────
def fit_ideal_cascade(N_rings, L, I_arr):
    """Fit (a, b) for identical-detuning cascade using minimax on log-error."""
    # Initialize from flank design: b = 1/N, a = -1 - b*L/2
    b_init = 1.0 / N_rings
    a_init = -1.0 - b_init * L / 2.0

    def objective(params):
        a, b = params
        if b <= 0:
            return 1e10
        return max_log_error(I_arr, a, b, L, N_rings)

    result = minimize(objective, [a_init, b_init], method='Nelder-Mead',
                      options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 50000})
    a_opt, b_opt = result.x
    E_inf = result.fun

    # Compute optimal C
    a_arr = np.full(N_rings, a_opt)
    log_y_no_C = log_transfer(I_arr, a_arr, b_opt, 0.0)
    ln_C_opt = minimax_optimal_lnC(log_y_no_C, I_arr, L)

    rel_err = max_rel_error(I_arr, a_opt, b_opt, L, N_rings)
    return a_opt, b_opt, ln_C_opt, E_inf, rel_err


# ── Step 2: Monte Carlo perturbation analysis ─────────────────────────
def run_perturbation_mc(a_opt, b_opt, L, N_rings, I_arr, sigma_values, n_mc, rng):
    """Run Monte Carlo for each sigma, return statistics."""
    results = {
        'sigma': sigma_values.tolist(),
        'eps_max_mean': [],
        'eps_max_median': [],
        'eps_max_p25': [],
        'eps_max_p75': [],
        'eps_max_p95': [],
        'eps_max_p05': [],
    }

    for sigma in sigma_values:
        if sigma == 0:
            # Deterministic
            rel_err = max_rel_error(I_arr, a_opt, b_opt, L, N_rings)
            results['eps_max_mean'].append(float(rel_err))
            results['eps_max_median'].append(float(rel_err))
            results['eps_max_p25'].append(float(rel_err))
            results['eps_max_p75'].append(float(rel_err))
            results['eps_max_p95'].append(float(rel_err))
            results['eps_max_p05'].append(float(rel_err))
            continue

        # Generate perturbed detunings: a_k = a_opt + delta_k
        delta = rng.normal(0, sigma, size=(n_mc, N_rings))
        a_perturbed = a_opt + delta  # (n_mc, N_rings)

        # Compute log transfer for all trials at once
        log_y_no_C = log_transfer(I_arr, a_perturbed, b_opt, 0.0)  # (n_mc, N_grid)

        # Minimax optimal C for each trial
        ln_C_arr = minimax_optimal_lnC(log_y_no_C, I_arr, L)  # (n_mc,)

        # Residual r(I) = ln C + log_y - (I - L)
        target = I_arr - L  # (N_grid,)
        r = ln_C_arr[:, None] + log_y_no_C - target[None, :]  # (n_mc, N_grid)

        # Max relative error per trial
        eps_max = np.max(np.abs(np.exp(r) - 1.0), axis=1)  # (n_mc,)

        results['eps_max_mean'].append(float(np.mean(eps_max)))
        results['eps_max_median'].append(float(np.median(eps_max)))
        results['eps_max_p25'].append(float(np.percentile(eps_max, 25)))
        results['eps_max_p75'].append(float(np.percentile(eps_max, 75)))
        results['eps_max_p95'].append(float(np.percentile(eps_max, 95)))
        results['eps_max_p05'].append(float(np.percentile(eps_max, 5)))

    return results


# ── Step 3: Plotting ──────────────────────────────────────────────────
def plot_main_figure(results, b_opt, fig_path):
    """Plot sigma vs eps_max with confidence bands and physical axis."""
    sigma = np.array(results['sigma'])
    mean = np.array(results['eps_max_mean']) * 100  # to percent
    median = np.array(results['eps_max_median']) * 100
    p25 = np.array(results['eps_max_p25']) * 100
    p75 = np.array(results['eps_max_p75']) * 100
    p95 = np.array(results['eps_max_p95']) * 100
    p05 = np.array(results['eps_max_p05']) * 100

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Confidence bands
    ax1.fill_between(sigma, p05, p95, alpha=0.15, color='C0', label='5th--95th percentile')
    ax1.fill_between(sigma, p25, p75, alpha=0.3, color='C0', label='25th--75th percentile')
    ax1.plot(sigma, median, '-', color='C0', linewidth=2, label='Median')
    ax1.plot(sigma, mean, '--', color='C3', linewidth=1.5, label='Mean')

    ax1.set_xlabel(r'Detuning scatter $\sigma$ (normalized)')
    ax1.set_ylabel(r'Max relative error $\varepsilon_{\max}$ (%)')
    ax1.set_xlim(0, SIGMA_MAX)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Ring uniformity perturbation analysis ($N={N}$, $L={L:.0f}$)')

    # Secondary x-axis: physical resonance scatter in pm
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    # sigma_physical (nm) = sigma * FWHM / (b_V * b_opt * ... )
    # Actually: sigma is in normalized detuning units.
    # The detuning parameter a = (lambda_0 - lambda_res) / (FWHM/2)
    # So delta_a = delta_lambda / (FWHM/2)
    # => delta_lambda = delta_a * FWHM/2
    # sigma_physical (nm) = sigma * FWHM/2
    sigma_to_pm = FWHM_nm * 1000 / 2  # convert sigma to pm (FWHM/2 in pm)
    tick_sigmas = ax1.get_xticks()
    tick_sigmas = tick_sigmas[(tick_sigmas >= 0) & (tick_sigmas <= SIGMA_MAX)]
    ax2.set_xticks(tick_sigmas)
    ax2.set_xticklabels([f'{s * sigma_to_pm:.0f}' for s in tick_sigmas])
    ax2.set_xlabel(r'Resonance scatter $\sigma_\lambda$ (pm)')

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def plot_example_curves(a_opt, b_opt, ln_C_opt, L, N_rings, I_arr, fig_path, rng):
    """Plot example cascade transmissions at sigma=0, 0.1, 0.3."""
    sigma_examples = [0.0, 0.1, 0.3]
    n_examples = 5  # number of random realizations to show per sigma

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios': [2, 1]})

    # Target
    target = np.exp(I_arr - L)

    colors = ['C0', 'C1', 'C2']

    for i, sigma in enumerate(sigma_examples):
        if sigma == 0:
            a_arr = np.full(N_rings, a_opt)
            log_y = log_transfer(I_arr, a_arr, b_opt, ln_C_opt)
            y = np.exp(log_y)
            axes[0].plot(I_arr, y, '-', color=colors[i], linewidth=2,
                         label=f'$\\sigma={sigma}$ (ideal)', zorder=5)
            rel_err = (y / target - 1) * 100
            axes[1].plot(I_arr, rel_err, '-', color=colors[i], linewidth=2, zorder=5)
        else:
            for j in range(n_examples):
                delta = rng.normal(0, sigma, size=N_rings)
                a_perturbed = a_opt + delta
                log_y_no_C = log_transfer(I_arr, a_perturbed, b_opt, 0.0)
                ln_C_trial = minimax_optimal_lnC(log_y_no_C, I_arr, L)
                log_y = ln_C_trial + log_y_no_C
                y = np.exp(log_y)
                label = f'$\\sigma={sigma}$' if j == 0 else None
                axes[0].plot(I_arr, y, '-', color=colors[i], alpha=0.4,
                             linewidth=1, label=label)
                rel_err = (y / target - 1) * 100
                axes[1].plot(I_arr, rel_err, '-', color=colors[i], alpha=0.4, linewidth=1)

    # Target curve
    axes[0].plot(I_arr, target, 'k--', linewidth=1.5, label='$e^{I-L}$ (target)', zorder=10)

    axes[0].set_yscale('log')
    axes[0].set_ylabel('Transmission $\\tilde{y}(I)$')
    axes[0].set_xlabel('Control signal $I$')
    axes[0].legend(loc='lower right', framealpha=0.9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Cascade transmission examples ($N={N}$, $L={L:.0f}$)')
    axes[0].set_xlim(0, L)

    axes[1].set_ylabel('Relative error (%)')
    axes[1].set_xlabel('Control signal $I$')
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, L)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(42)
    I_arr = np.linspace(0, L, N_GRID)

    print(f"Fitting ideal cascade: N={N}, L={L}")
    a_opt, b_opt, ln_C_opt, E_inf, rel_err_ideal = fit_ideal_cascade(N, L, I_arr)
    print(f"  a = {a_opt:.6f}")
    print(f"  b = {b_opt:.6f}")
    print(f"  ln C = {ln_C_opt:.6f}")
    print(f"  E_inf (log) = {E_inf:.6f}")
    print(f"  Max rel error = {rel_err_ideal*100:.3f}%")

    # Sigma sweep
    sigma_values = np.linspace(0, SIGMA_MAX, N_SIGMA)
    print(f"\nRunning Monte Carlo ({N_MC} trials per sigma, {N_SIGMA} sigma values)...")
    results = run_perturbation_mc(a_opt, b_opt, L, N, I_arr, sigma_values, N_MC, rng)

    # Add metadata
    results['parameters'] = {
        'N': N,
        'L': L,
        'a_opt': float(a_opt),
        'b_opt': float(b_opt),
        'ln_C_opt': float(ln_C_opt),
        'E_inf_log': float(E_inf),
        'rel_err_ideal_pct': float(rel_err_ideal * 100),
        'b_V': b_V,
        'FWHM_nm': FWHM_nm,
        'N_MC': N_MC,
        'N_GRID': N_GRID,
    }

    # Save results
    res_path = os.path.join(RES_DIR, "ring_uniformity_results.json")
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {res_path}")

    # Plot main figure
    fig1_path = os.path.join(FIG_DIR, "fig_ring_uniformity.png")
    plot_main_figure(results, b_opt, fig1_path)

    # Plot example curves
    fig2_path = os.path.join(FIG_DIR, "fig_ring_uniformity_examples.png")
    plot_example_curves(a_opt, b_opt, ln_C_opt, L, N, I_arr, fig2_path, rng)

    print("\nDone.")


if __name__ == '__main__':
    main()
