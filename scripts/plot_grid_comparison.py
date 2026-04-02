#!/usr/bin/env python
"""
Full-grid comparison: FD ground truth vs Deep BSDE.

Compares V(q), delta_a(q), delta_b(q), spread(q), and errors
across the ENTIRE inventory grid, not just q=0.

Usage:
    python plot_grid_comparison.py --weights logs/lob_corrected_model.pt
    python plot_grid_comparison.py --weights logs/lob_corrected_model.pt --fd_horizon finite
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from registry import EQUATION_REGISTRY
import equations
from solver import ContXiongLOBModel
from finite_difference_finite_horizon import solve_finite_horizon_1d


def evaluate_deep_bsde(model, bsde, q_range):
    """Evaluate the trained Deep BSDE at t=0 across inventory levels."""
    model.eval()
    sigma_q = bsde._sigma_q_equilibrium()

    y0 = model.y_init.item()
    r = bsde.discount_rate
    T = bsde.total_time
    finite_factor = (1 - np.exp(-r * T)) / r

    results = {"q": [], "V": [], "delta_a": [], "delta_b": [], "spread": [], "z_q": []}

    for q in q_range:
        x = torch.tensor([[bsde.s_init, q]], dtype=torch.float64)
        if len(model.subnet) > 0:
            with torch.no_grad():
                z = model.subnet[0](x) / bsde.dim
            z_q = z[0, 1].item()
        else:
            z_q = 0.0

        p = z_q / sigma_q  # approximate dV/dq
        d_a = 1.0 / bsde.alpha + p
        d_b = 1.0 / bsde.alpha - p

        # Value estimate via HJB relation
        f_a = np.exp(-bsde.alpha * d_a) * bsde.lambda_a
        f_b = np.exp(-bsde.alpha * d_b) * bsde.lambda_b
        profits = f_a * d_a + f_b * d_b
        psi = bsde.phi * q ** 2
        V_q = (profits - psi) * finite_factor + (-psi) * np.exp(-r * T)

        results["q"].append(q)
        results["V"].append(V_q)
        results["delta_a"].append(d_a)
        results["delta_b"].append(d_b)
        results["spread"].append(d_a + d_b)
        results["z_q"].append(z_q)

    return {k: np.array(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lob_d2.json")
    parser.add_argument("--weights", default="logs/lob_corrected_model.pt")
    parser.add_argument("--out_dir", default="plots/grid_comparison")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_default_dtype(torch.float64)

    # Load Deep BSDE model
    config = Config.from_json(args.config)
    bsde = EQUATION_REGISTRY["contxiong_lob"](config.eqn)
    model = ContXiongLOBModel(config, bsde)
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"Loaded weights: Y0={ckpt['y0']:.4f}")
    else:
        print(f"Warning: {args.weights} not found")

    # Solve finite-horizon FD (matched problem)
    print("Solving finite-horizon FD...")
    t_fd, q_fd, V_fd, da_fd, db_fd = solve_finite_horizon_1d(
        lambda_a=bsde.lambda_a, lambda_b=bsde.lambda_b, alpha=bsde.alpha,
        phi=bsde.phi, r=bsde.discount_rate, T=bsde.total_time,
        H=10, Delta=1.0, N_t=200,
    )

    # Evaluate Deep BSDE on the same q grid
    q_continuous = np.linspace(q_fd[0], q_fd[-1], 200)
    db_results = evaluate_deep_bsde(model, bsde, q_continuous)

    # Also evaluate at FD grid points for error computation
    db_at_fd = evaluate_deep_bsde(model, bsde, q_fd)

    # === Plot 1: Value function comparison ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(q_fd, V_fd[0, :], "k-", linewidth=2.5, label="FD (finite horizon)")
    axes[0].plot(q_continuous, db_results["V"], "b--", linewidth=1.5, label="Deep BSDE surrogate")
    axes[0].set_xlabel("Inventory $q$", fontsize=11)
    axes[0].set_ylabel("$V(0, q)$", fontsize=11)
    axes[0].set_title("Value Function", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Error
    V_error = db_at_fd["V"] - V_fd[0, :]
    axes[1].bar(q_fd, V_error, width=0.8, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Inventory $q$", fontsize=11)
    axes[1].set_ylabel("$V_{\\mathrm{BSDE}} - V_{\\mathrm{FD}}$", fontsize=11)
    axes[1].set_title("Value Error Across Grid", fontsize=12)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].axhline(y=0, color="k", linewidth=0.5)

    # Relative error
    V_rel = np.abs(V_error) / np.maximum(np.abs(V_fd[0, :]), 1e-8) * 100
    axes[2].bar(q_fd, V_rel, width=0.8, color="coral", edgecolor="white")
    axes[2].set_xlabel("Inventory $q$", fontsize=11)
    axes[2].set_ylabel("Relative error (%)", fontsize=11)
    axes[2].set_title("Relative Value Error", fontsize=12)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "value_grid_comparison.png"), dpi=150)
    plt.close()
    print("Saved value_grid_comparison.png")

    # === Plot 2: Spread comparison ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    fd_spread = da_fd[0, :] + db_fd[0, :]
    axes[0].plot(q_fd, fd_spread, "k-", linewidth=2.5, label="FD spread (discrete)")
    axes[0].plot(q_continuous, db_results["spread"], "b--", linewidth=1.5,
                 label="Surrogate spread (= $2/\\alpha$)")
    axes[0].set_xlabel("Inventory $q$", fontsize=11)
    axes[0].set_ylabel("Total spread", fontsize=11)
    axes[0].set_title("Spread: FD vs Surrogate", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Individual quotes
    axes[1].plot(q_fd, da_fd[0, :], "k-", linewidth=2, label="FD $\\delta^a$")
    axes[1].plot(q_fd, db_fd[0, :], "k--", linewidth=2, label="FD $\\delta^b$")
    axes[1].plot(q_continuous, db_results["delta_a"], "b-", linewidth=1.2, alpha=0.7,
                 label="BSDE $\\delta^a$")
    axes[1].plot(q_continuous, db_results["delta_b"], "b--", linewidth=1.2, alpha=0.7,
                 label="BSDE $\\delta^b$")
    axes[1].set_xlabel("Inventory $q$", fontsize=11)
    axes[1].set_ylabel("Quote", fontsize=11)
    axes[1].set_title("Individual Quotes", fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "spread_grid_comparison.png"), dpi=150)
    plt.close()
    print("Saved spread_grid_comparison.png")

    # Print summary
    print(f"\n=== Grid Comparison Summary ===")
    print(f"Max absolute V error: {np.max(np.abs(V_error)):.6f}")
    print(f"Mean absolute V error: {np.mean(np.abs(V_error)):.6f}")
    print(f"Max relative V error: {np.max(V_rel):.2f}%")
    print(f"Mean relative V error: {np.mean(V_rel):.2f}%")
    print(f"FD spread range: [{fd_spread.min():.4f}, {fd_spread.max():.4f}]")
    print(f"BSDE spread (constant): {db_results['spread'][0]:.4f}")


if __name__ == "__main__":
    main()
