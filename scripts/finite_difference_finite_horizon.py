#!/usr/bin/env python
"""
Finite-difference solution of the FINITE-HORIZON discrete-inventory HJB.

Solves EXACTLY the same problem as the Deep BSDE solver:
- Same T, same r, same terminal condition g(q_T) = -psi(q_T)
- Same discrete inventory grid q ∈ {-H, ..., H}
- Backward in time from t=T to t=0

This provides an apples-to-apples ground truth for V(t, q).

The time-dependent HJB is:
  dV/dt + rV(t,q) + psi(q) = lambda^a * max_da [f(da)(da*D + V(t,q-D) - V(t,q))]
                             + lambda^b * max_db [f(db)(db*D + V(t,q+D) - V(t,q))]

with terminal condition V(T, q) = -psi(q).

Also solves the 2D version V(t, S, q) when price is economically active
(mark-to-market: running P&L includes q * dS contribution).

Usage:
    python finite_difference_finite_horizon.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json


def solve_finite_horizon_1d(
    lambda_a=1.0, lambda_b=1.0, alpha=1.5,
    phi=0.01, r=0.1, T=1.0, H=10, Delta=1.0,
    N_t=200, penalty_type="quadratic", gamma=1.0,
):
    """Backward solve of the finite-horizon discrete-inventory HJB.

    Returns:
        t_grid: time grid [0, ..., T]
        q_grid: inventory grid [-H, ..., H]
        V: value function V[t_idx, q_idx]
        delta_a: optimal ask quotes [t_idx, q_idx]
        delta_b: optimal bid quotes [t_idx, q_idx]
    """
    dt = T / N_t
    q_grid = np.arange(-H, H + Delta, Delta)
    t_grid = np.linspace(0, T, N_t + 1)
    n_q = len(q_grid)

    def psi(q):
        if penalty_type == "quadratic":
            return phi * q ** 2
        elif penalty_type == "cubic":
            return phi * q ** 2 + phi * np.abs(q) ** 3 / 3
        elif penalty_type == "exponential":
            return phi * (np.exp(gamma * np.abs(q)) - 1)
        return phi * q ** 2

    # V[t_idx, q_idx]
    V = np.zeros((N_t + 1, n_q))
    delta_a = np.zeros((N_t + 1, n_q))
    delta_b = np.zeros((N_t + 1, n_q))

    # Terminal condition: V(T, q) = -psi(q)
    for j, q in enumerate(q_grid):
        V[N_t, j] = -psi(q)

    # Backward solve
    for n in range(N_t - 1, -1, -1):
        for j, q in enumerate(q_grid):
            # Neighboring values
            if j > 0:
                V_down = V[n + 1, j - 1]
            else:
                V_down = -psi(q - Delta)  # boundary

            if j < n_q - 1:
                V_up = V[n + 1, j + 1]
            else:
                V_up = -psi(q + Delta)  # boundary

            V_here = V[n + 1, j]

            # Optimal quotes (same FOC as stationary case)
            jump_down = V_down - V_here
            jump_up = V_up - V_here

            da = 1.0 / alpha - jump_down / Delta
            db = 1.0 / alpha - jump_up / Delta
            da = max(da, 0.001)
            db = max(db, 0.001)

            # Execution rates
            rate_a = lambda_a * np.exp(-alpha * da)
            rate_b = lambda_b * np.exp(-alpha * db)

            # Profits
            profit_a = rate_a * (da * Delta + jump_down)
            profit_b = rate_b * (db * Delta + jump_up)

            # Explicit Euler backward:
            # V(t, q) = V(t+dt, q) + dt * [profits - psi - r*V(t+dt, q)]
            V[n, j] = V_here + dt * (profit_a + profit_b - psi(q) - r * V_here)

            delta_a[n, j] = da
            delta_b[n, j] = db

    return t_grid, q_grid, V, delta_a, delta_b


def solve_finite_horizon_2d(
    sigma=0.3, lambda_a=1.0, lambda_b=1.0, alpha=1.5,
    phi=0.01, r=0.1, T=1.0, H=5, Delta=1.0,
    N_t=100, N_s=50,
    S_lo=95.0, S_hi=105.0,
):
    """Backward solve of the 2D finite-horizon HJB with active price.

    When price is active (mark-to-market), the value function depends on
    both S and q. The running P&L includes q * sigma^2 / 2 * d(S^2)/dS
    from holding inventory, and the HJB has a sigma^2/2 * d^2V/dS^2 term.

    Returns:
        t_grid, s_grid, q_grid, V[t, s, q]
    """
    dt = T / N_t
    q_grid = np.arange(-H, H + Delta, Delta)
    s_grid = np.linspace(S_lo, S_hi, N_s)
    t_grid = np.linspace(0, T, N_t + 1)
    n_q = len(q_grid)
    n_s = len(s_grid)
    ds = s_grid[1] - s_grid[0]

    def psi(q):
        return phi * q ** 2

    # V[t, s, q]
    V = np.zeros((N_t + 1, n_s, n_q))

    # Terminal: V(T, S, q) = -psi(q)  (no S dependence at terminal)
    for k in range(n_s):
        for j, q in enumerate(q_grid):
            V[N_t, k, j] = -psi(q)

    # Backward solve
    for n in range(N_t - 1, -1, -1):
        for k in range(n_s):
            for j, q in enumerate(q_grid):
                V_here = V[n + 1, k, j]

                # Inventory neighbors
                V_down = V[n + 1, k, j - 1] if j > 0 else -psi(q - Delta)
                V_up = V[n + 1, k, j + 1] if j < n_q - 1 else -psi(q + Delta)

                # Price second derivative (central difference)
                if 0 < k < n_s - 1:
                    V_ss = (V[n + 1, k + 1, j] - 2 * V_here + V[n + 1, k - 1, j]) / ds ** 2
                else:
                    V_ss = 0.0  # boundary

                # Optimal quotes
                jump_down = V_down - V_here
                jump_up = V_up - V_here
                da = max(1.0 / alpha - jump_down / Delta, 0.001)
                db = max(1.0 / alpha - jump_up / Delta, 0.001)

                rate_a = lambda_a * np.exp(-alpha * da)
                rate_b = lambda_b * np.exp(-alpha * db)
                profit_a = rate_a * (da * Delta + jump_down)
                profit_b = rate_b * (db * Delta + jump_up)

                # Mark-to-market: inventory holding cost/gain from price moves
                # This enters through sigma^2/2 * d^2V/dS^2 in the PDE
                price_term = 0.5 * sigma ** 2 * V_ss

                V[n, k, j] = V_here + dt * (
                    profit_a + profit_b - psi(q) - r * V_here + price_term
                )

    return t_grid, s_grid, q_grid, V


def main():
    out_dir = "plots/fd_finite_horizon"
    os.makedirs(out_dir, exist_ok=True)

    print("=== 1D Finite-Horizon FD (same T=1, same g(X_T) = -psi(q_T)) ===\n")
    t, q, V, da, db = solve_finite_horizon_1d()

    mid_q = len(q) // 2  # q=0
    print(f"V(0, q=0) = {V[0, mid_q]:.6f}")
    print(f"V(T, q=0) = {V[-1, mid_q]:.6f} (should be -psi(0) = 0)")
    print(f"Spread at (t=0, q=0) = {da[0, mid_q] + db[0, mid_q]:.6f}")

    # Save V(0, q) for comparison
    results = {
        "V_0_q": V[0, :].tolist(),
        "q_grid": q.tolist(),
        "V_0_at_0": float(V[0, mid_q]),
        "spread_0_at_0": float(da[0, mid_q] + db[0, mid_q]),
        "da_0": da[0, :].tolist(),
        "db_0": db[0, :].tolist(),
    }
    with open(os.path.join(out_dir, "fd_finite_horizon_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot V(t=0, q)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(q, V[0, :], "k-", linewidth=2.5, label="FD finite-horizon $V(0, q)$")
    axes[0].set_xlabel("Inventory $q$", fontsize=11)
    axes[0].set_ylabel("$V(0, q)$", fontsize=11)
    axes[0].set_title("Finite-Horizon Value Function (ground truth)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(q, da[0, :] + db[0, :], "k-", linewidth=2.5, label="FD spread")
    axes[1].axhline(y=2.0 / 1.5, color="b", linestyle=":", alpha=0.5, label="$2/\\alpha$ (continuous)")
    axes[1].set_xlabel("Inventory $q$", fontsize=11)
    axes[1].set_ylabel("Spread $\\delta^a + \\delta^b$", fontsize=11)
    axes[1].set_title("Optimal Spread at $t=0$", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(1.2, 1.8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fd_finite_horizon_1d.png"), dpi=150)
    plt.close()
    print(f"Saved {out_dir}/fd_finite_horizon_1d.png")

    # Plot V(t, q=0) over time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, V[:, mid_q], "k-", linewidth=2)
    ax.set_xlabel("Time $t$", fontsize=11)
    ax.set_ylabel("$V(t, q=0)$", fontsize=11)
    ax.set_title("Value at $q=0$ Over Time", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fd_value_over_time.png"), dpi=150)
    plt.close()
    print(f"Saved {out_dir}/fd_value_over_time.png")

    # Heatmap V(t, q)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(V.T, aspect="auto", origin="lower",
                   extent=[t[0], t[-1], q[0], q[-1]], cmap="coolwarm")
    plt.colorbar(im, ax=ax, label="$V(t, q)$")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Inventory $q$")
    ax.set_title("FD Value Function $V(t, q)$ (finite horizon)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fd_value_heatmap.png"), dpi=150)
    plt.close()
    print(f"Saved {out_dir}/fd_value_heatmap.png")

    print(f"\n2D solve with active price:")
    t2, s2, q2, V2 = solve_finite_horizon_2d(N_t=50, N_s=30, H=5)
    mid_s = len(s2) // 2
    mid_q2 = len(q2) // 2
    print(f"V(0, S=100, q=0) = {V2[0, mid_s, mid_q2]:.6f}")
    print(f"V depends on S: V(0, S_lo, 0)={V2[0, 0, mid_q2]:.4f}, "
          f"V(0, S_hi, 0)={V2[0, -1, mid_q2]:.4f}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
