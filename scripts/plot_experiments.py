#!/usr/bin/env python
"""
Generate all plots from the experiment results JSON.

Usage:
    python plot_experiments.py                          # default
    python plot_experiments.py --results results/all_experiments.json
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_multiseed_comparison(results, out_dir):
    """Bar chart: Y0 with error bars for continuous vs jump vs FD."""
    cont = results.get("continuous_multiseed", {})
    jump = results.get("jump_multiseed", {})

    fd_v0 = 0.433  # FD ground truth (T=1 corrected)

    labels = ["FD ground truth", "Jump model\n(Option B)", "Diffusion surrogate\n(Option A)"]
    means = [fd_v0, jump.get("y0_mean", 0), cont.get("y0_mean", 0)]
    stds = [0, jump.get("y0_std", 0), cont.get("y0_std", 0)]
    colors = ["#2d2d2d", "#2196F3", "#FF9800"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.6)
    ax.set_ylabel("$Y_0$ (value at $q=0$)", fontsize=12)
    ax.set_title("Method Comparison (multi-seed)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Annotate values
    for bar, m, s in zip(bars, means, stds):
        if s > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
                    f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{m:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multiseed_comparison.png"), dpi=150)
    plt.close()
    print("Saved multiseed_comparison.png")


def plot_ablation(results, out_dir):
    """Type 1 vs Type 3 comparison."""
    t1 = results.get("ablation_type1", {})
    t3 = results.get("ablation_type3", {})

    if not t1 or not t3:
        print("Skipping ablation plot (no data)")
        return

    labels = ["Type 1\n(no mean-field)", "Type 3\n(fictitious play)"]
    means = [t1["y0_mean"], t3["y0_mean"]]
    stds = [t1["y0_std"], t3["y0_std"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=["#9E9E9E", "#4CAF50"], edgecolor="white", width=0.5)
    ax.set_ylabel("$Y_0$", fontsize=12)
    ax.set_title("Mean-Field Ablation", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.003,
                f"{m:.4f}±{s:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_meanfield.png"), dpi=150)
    plt.close()
    print("Saved ablation_meanfield.png")


def plot_stress_multiseed(results, out_dir):
    """Stress test: Y0 and z_max across penalty types with error bars."""
    penalties = ["quadratic", "cubic", "exponential"]
    data = [results.get(f"stress_{p}", {}) for p in penalties]
    if not all(data):
        print("Skipping stress plot (incomplete data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Y0
    y0_means = [d["y0_mean"] for d in data]
    y0_stds = [d["y0_std"] for d in data]
    axes[0].bar(penalties, y0_means, yerr=y0_stds, capsize=6,
                color=["#4CAF50", "#FF9800", "#F44336"], edgecolor="white", width=0.5)
    axes[0].set_ylabel("$Y_0$", fontsize=12)
    axes[0].set_title("Value Function by Penalty Type", fontsize=12)
    axes[0].grid(axis="y", alpha=0.3)

    # z_max
    z_means = [d["z_max_mean"] for d in data]
    z_stds = [d["z_max_std"] for d in data]
    axes[1].bar(penalties, z_means, yerr=z_stds, capsize=6,
                color=["#4CAF50", "#FF9800", "#F44336"], edgecolor="white", width=0.5)
    axes[1].set_ylabel("max $|Z_t|$", fontsize=12)
    axes[1].set_title("Gradient Magnitude by Penalty Type", fontsize=12)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stress_multiseed.png"), dpi=150)
    plt.close()
    print("Saved stress_multiseed.png")


def plot_breaking_point(results, out_dir):
    """Breaking point: gamma and phi sweeps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Gamma sweep
    bp_gamma = results.get("breaking_point_gamma", [])
    if bp_gamma:
        gammas = [r["gamma"] for r in bp_gamma]
        losses = [r["loss"] for r in bp_gamma]
        z_maxs = [r["z_max"] for r in bp_gamma]
        converged = [r["converged"] for r in bp_gamma]

        ax = axes[0]
        colors = ["#4CAF50" if c else "#F44336" for c in converged]
        ax.semilogy(gammas, losses, "o-", color="#2196F3", linewidth=1.5, markersize=6)
        for g, l, c in zip(gammas, losses, colors):
            ax.plot(g, l, "o", color=c, markersize=8, zorder=5)
        ax.set_xlabel("$\\gamma$ (exponential penalty)", fontsize=11)
        ax.set_ylabel("Final loss (log)", fontsize=11)
        ax.set_title("Breaking Point: $\\gamma$ Sweep", fontsize=12)
        ax.grid(True, alpha=0.3)
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
                                  markersize=8, label='Converged'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
                                  markersize=8, label='Diverged')]
        ax.legend(handles=legend_elements, fontsize=9)

    # Phi sweep
    bp_phi = results.get("breaking_point_phi", [])
    if bp_phi:
        phis = [r["phi"] for r in bp_phi]
        losses = [r["loss"] for r in bp_phi]
        converged = [r["converged"] for r in bp_phi]

        ax = axes[1]
        colors = ["#4CAF50" if c else "#F44336" for c in converged]
        ax.semilogy(phis, losses, "o-", color="#FF9800", linewidth=1.5, markersize=6)
        for p, l, c in zip(phis, losses, colors):
            ax.plot(p, l, "o", color=c, markersize=8, zorder=5)
        ax.set_xlabel("$\\phi$ (penalty coefficient)", fontsize=11)
        ax.set_ylabel("Final loss (log)", fontsize=11)
        ax.set_title("Breaking Point: $\\phi$ Sweep", fontsize=12)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "breaking_point.png"), dpi=150)
    plt.close()
    print("Saved breaking_point.png")


def plot_sensitivity(results, out_dir):
    """Parameter sensitivity: alpha, lambda, N_T."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Alpha
    alpha_data = results.get("sensitivity_alpha", [])
    if alpha_data:
        alphas = [r["alpha"] for r in alpha_data]
        y0s = [r["y0"] for r in alpha_data]
        axes[0].plot(alphas, y0s, "o-", color="#2196F3", linewidth=1.5, markersize=7)
        axes[0].set_xlabel("$\\alpha$ (execution decay)", fontsize=11)
        axes[0].set_ylabel("$Y_0$", fontsize=11)
        axes[0].set_title("Sensitivity to $\\alpha$", fontsize=12)
        axes[0].grid(True, alpha=0.3)

    # Lambda
    lam_data = results.get("sensitivity_lambda", [])
    if lam_data:
        lams = [r["lambda"] for r in lam_data]
        y0s = [r["y0"] for r in lam_data]
        axes[1].plot(lams, y0s, "o-", color="#FF9800", linewidth=1.5, markersize=7)
        axes[1].set_xlabel("$\\lambda$ (arrival rate)", fontsize=11)
        axes[1].set_ylabel("$Y_0$", fontsize=11)
        axes[1].set_title("Sensitivity to $\\lambda$", fontsize=12)
        axes[1].grid(True, alpha=0.3)

    # N_T
    nt_data = results.get("sensitivity_NT", [])
    if nt_data:
        nts = [r["N_T"] for r in nt_data]
        y0s = [r["y0"] for r in nt_data]
        axes[2].plot(nts, y0s, "o-", color="#4CAF50", linewidth=1.5, markersize=7)
        axes[2].set_xlabel("$N_T$ (time steps)", fontsize=11)
        axes[2].set_ylabel("$Y_0$", fontsize=11)
        axes[2].set_title("Sensitivity to $N_T$", fontsize=12)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sensitivity.png"), dpi=150)
    plt.close()
    print("Saved sensitivity.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/all_experiments.json")
    parser.add_argument("--out_dir", default="plots/experiments")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        print("Run: python run_all_experiments.py")
        return

    results = load_results(args.results)
    print(f"Loaded results from {args.results}")
    if "metadata" in results:
        m = results["metadata"]
        print(f"  Seeds: {m.get('n_seeds')}, Iterations: {m.get('n_iters')}, "
              f"Time: {m.get('elapsed_seconds', 0)/60:.1f} min")

    plot_multiseed_comparison(results, args.out_dir)
    plot_ablation(results, args.out_dir)
    plot_stress_multiseed(results, args.out_dir)
    plot_breaking_point(results, args.out_dir)
    plot_sensitivity(results, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
