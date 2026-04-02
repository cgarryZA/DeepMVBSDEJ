#!/usr/bin/env python
"""
Forward simulation of the learned market-making policy.

Simulates N paths under the trained Deep BSDE quotes and reports:
- Realized P&L (spread capture minus inventory penalty)
- Inventory statistics (mean, std, max)
- Quote statistics (mean spread, skew)
- Sharpe ratio of the P&L
- Comparison against FD-optimal quotes

Usage:
    python simulate_policy.py --weights logs/lob_corrected_model.pt
    python simulate_policy.py --weights logs/lob_corrected_model.pt --n_paths 10000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
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


def simulate_under_policy(model, bsde, n_paths=5000, use_fd_policy=False, fd_data=None):
    """Simulate market-making under the learned (or FD) policy.

    Returns dict with P&L, inventory, and quote trajectories.
    """
    model.eval()
    sigma_q_eq = bsde._sigma_q_equilibrium()
    dt = bsde.delta_t
    T = bsde.total_time
    N_T = bsde.num_time_interval

    # State trajectories
    S = np.zeros((n_paths, N_T + 1))
    q = np.zeros((n_paths, N_T + 1))
    S[:, 0] = bsde.s_init
    q[:, 0] = 0.0

    # P&L tracking
    pnl = np.zeros((n_paths, N_T + 1))
    spread_earned = np.zeros(n_paths)
    penalty_paid = np.zeros(n_paths)

    # Quote trajectories
    delta_a_traj = np.zeros((n_paths, N_T))
    delta_b_traj = np.zeros((n_paths, N_T))

    # FD policy lookup
    if use_fd_policy and fd_data is not None:
        fd_t, fd_q, fd_V, fd_da, fd_db = fd_data
        fd_q_list = fd_q.tolist()

    for n in range(N_T):
        dW_S = np.random.normal(0, np.sqrt(dt), n_paths)
        dW_q = np.random.normal(0, np.sqrt(dt), n_paths)

        if use_fd_policy and fd_data is not None:
            # Look up FD quotes by nearest inventory
            t_idx = min(n, fd_da.shape[0] - 1)
            for i in range(n_paths):
                q_idx = np.argmin(np.abs(fd_q - q[i, n]))
                delta_a_traj[i, n] = fd_da[t_idx, q_idx]
                delta_b_traj[i, n] = fd_db[t_idx, q_idx]
        else:
            # Deep BSDE policy
            x_batch = torch.tensor(
                np.stack([S[:, n], q[:, n]], axis=1), dtype=torch.float64
            )
            subnet_idx = min(n, len(model.subnet) - 1)
            with torch.no_grad():
                z = model.subnet[subnet_idx](x_batch) / bsde.dim
            z_q = z[:, 1].numpy()
            p = z_q / sigma_q_eq
            delta_a_traj[:, n] = 1.0 / bsde.alpha + p
            delta_b_traj[:, n] = 1.0 / bsde.alpha - p

        # Clamp quotes
        delta_a_traj[:, n] = np.clip(delta_a_traj[:, n], 0.001, 10.0)
        delta_b_traj[:, n] = np.clip(delta_b_traj[:, n], 0.001, 10.0)

        # Execution (Poisson)
        rate_a = bsde.lambda_a * np.exp(-bsde.alpha * delta_a_traj[:, n])
        rate_b = bsde.lambda_b * np.exp(-bsde.alpha * delta_b_traj[:, n])
        n_ask = np.random.poisson(rate_a * dt)
        n_bid = np.random.poisson(rate_b * dt)

        # P&L from executions
        spread_earned += delta_a_traj[:, n] * n_ask + delta_b_traj[:, n] * n_bid

        # Inventory update (discrete jumps)
        q[:, n + 1] = q[:, n] + (n_bid - n_ask)
        q[:, n + 1] = np.clip(q[:, n + 1], -10, 10)

        # Price update
        S[:, n + 1] = S[:, n] + bsde.sigma_s * dW_S

        # Running penalty
        penalty_paid += bsde.phi * q[:, n] ** 2 * dt

        # Mark-to-market P&L from inventory
        mtm = q[:, n] * bsde.sigma_s * dW_S

        # Cumulative P&L
        pnl[:, n + 1] = pnl[:, n] + (
            delta_a_traj[:, n] * n_ask + delta_b_traj[:, n] * n_bid
            + mtm
            - bsde.phi * q[:, n] ** 2 * dt
        ) * np.exp(-bsde.discount_rate * n * dt)

    # Terminal penalty
    terminal_penalty = bsde.phi * q[:, -1] ** 2
    final_pnl = pnl[:, -1] - terminal_penalty * np.exp(-bsde.discount_rate * T)

    return {
        "pnl": pnl,
        "final_pnl": final_pnl,
        "spread_earned": spread_earned,
        "penalty_paid": penalty_paid,
        "terminal_penalty": terminal_penalty,
        "q": q,
        "S": S,
        "delta_a": delta_a_traj,
        "delta_b": delta_b_traj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lob_d2.json")
    parser.add_argument("--weights", default="logs/lob_corrected_model.pt")
    parser.add_argument("--n_paths", type=int, default=5000)
    parser.add_argument("--out_dir", default="plots/simulation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    np.random.seed(42)

    config = Config.from_json(args.config)
    bsde = EQUATION_REGISTRY["contxiong_lob"](config.eqn)
    model = ContXiongLOBModel(config, bsde)

    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded weights: Y0={ckpt['y0']:.4f}")

    # Solve FD for comparison policy
    print("Solving FD for comparison...")
    fd_data = solve_finite_horizon_1d(
        lambda_a=bsde.lambda_a, lambda_b=bsde.lambda_b, alpha=bsde.alpha,
        phi=bsde.phi, r=bsde.discount_rate, T=bsde.total_time,
    )

    print(f"\nSimulating {args.n_paths} paths under Deep BSDE policy...")
    bsde_sim = simulate_under_policy(model, bsde, args.n_paths)

    print(f"Simulating {args.n_paths} paths under FD policy...")
    np.random.seed(42)  # same seed for fair comparison
    fd_sim = simulate_under_policy(model, bsde, args.n_paths, use_fd_policy=True, fd_data=fd_data)

    # === Summary statistics ===
    print(f"\n{'='*60}")
    print(f"{'Metric':<30} {'BSDE Policy':>15} {'FD Policy':>15}")
    print(f"{'='*60}")
    for label, bsde_val, fd_val in [
        ("Mean final P&L", np.mean(bsde_sim["final_pnl"]), np.mean(fd_sim["final_pnl"])),
        ("Std final P&L", np.std(bsde_sim["final_pnl"]), np.std(fd_sim["final_pnl"])),
        ("Sharpe ratio", np.mean(bsde_sim["final_pnl"]) / max(np.std(bsde_sim["final_pnl"]), 1e-8),
                         np.mean(fd_sim["final_pnl"]) / max(np.std(fd_sim["final_pnl"]), 1e-8)),
        ("Mean |q_T|", np.mean(np.abs(bsde_sim["q"][:, -1])), np.mean(np.abs(fd_sim["q"][:, -1]))),
        ("Mean spread", np.mean(bsde_sim["delta_a"] + bsde_sim["delta_b"]),
                        np.mean(fd_sim["delta_a"] + fd_sim["delta_b"])),
    ]:
        print(f"{label:<30} {bsde_val:>15.4f} {fd_val:>15.4f}")

    # === Plots ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # P&L distribution
    axes[0, 0].hist(bsde_sim["final_pnl"], bins=50, alpha=0.6, color="blue",
                    label="BSDE", density=True, edgecolor="white")
    axes[0, 0].hist(fd_sim["final_pnl"], bins=50, alpha=0.6, color="black",
                    label="FD", density=True, edgecolor="white")
    axes[0, 0].set_xlabel("Final P&L")
    axes[0, 0].set_title("P&L Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Inventory paths (sample)
    for i in range(min(20, args.n_paths)):
        axes[0, 1].plot(bsde_sim["q"][i, :], alpha=0.3, linewidth=0.5, color="blue")
    axes[0, 1].axhline(y=0, color="k", alpha=0.3)
    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].set_ylabel("Inventory $q$")
    axes[0, 1].set_title("Inventory Paths (BSDE policy)")
    axes[0, 1].grid(True, alpha=0.3)

    # Cumulative P&L (mean + quantiles)
    mean_pnl = np.mean(bsde_sim["pnl"], axis=0)
    q25 = np.percentile(bsde_sim["pnl"], 25, axis=0)
    q75 = np.percentile(bsde_sim["pnl"], 75, axis=0)
    steps = np.arange(len(mean_pnl))
    axes[1, 0].plot(steps, mean_pnl, "b-", linewidth=1.5, label="BSDE mean")
    axes[1, 0].fill_between(steps, q25, q75, alpha=0.2, color="blue")
    fd_mean = np.mean(fd_sim["pnl"], axis=0)
    axes[1, 0].plot(steps, fd_mean, "k--", linewidth=1.5, label="FD mean")
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("Cumulative P&L")
    axes[1, 0].set_title("Cumulative P&L (mean ± IQR)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Terminal inventory distribution
    axes[1, 1].hist(bsde_sim["q"][:, -1], bins=30, alpha=0.6, color="blue",
                    label="BSDE", density=True, edgecolor="white")
    axes[1, 1].hist(fd_sim["q"][:, -1], bins=30, alpha=0.6, color="black",
                    label="FD", density=True, edgecolor="white")
    axes[1, 1].set_xlabel("Terminal inventory $q_T$")
    axes[1, 1].set_title("Terminal Inventory Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "policy_simulation.png"), dpi=150)
    plt.close()
    print(f"\nSaved {args.out_dir}/policy_simulation.png")


if __name__ == "__main__":
    main()
