#!/usr/bin/env python
"""
Finite-difference ground truth for the 1D stationary market-making HJB.

Solves the discrete inventory HJB (Cont-Xiong eq 28) on a grid
q ∈ {-H, ..., H} by policy iteration. Provides ground truth V(q)
and optimal quotes for validating the Deep BSDE solver.

Usage:
    python finite_difference_baseline.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def solve_hjb_policy_iteration(
    lambda_a=1.0, lambda_b=1.0, alpha=1.5,
    phi=0.01, r=0.1, H=10, Delta=1.0,
    penalty_type="quadratic", gamma=1.0,
    tol=1e-12, max_iter=200,
):
    """Solve the stationary discrete-inventory HJB by policy iteration.

    Policy iteration alternates:
    1. Policy evaluation: given quotes delta(q), solve the LINEAR system for V
    2. Policy improvement: update quotes from the new V

    This is much more stable than value iteration for this problem.
    """
    q_grid = np.arange(-H, H + Delta, Delta)
    n = len(q_grid)
    q_to_idx = {q: i for i, q in enumerate(q_grid)}

    def psi(q):
        if penalty_type == "quadratic":
            return phi * q ** 2
        elif penalty_type == "cubic":
            return phi * q ** 2 + phi * np.abs(q) ** 3 / 3
        elif penalty_type == "exponential":
            return phi * (np.exp(gamma * np.abs(q)) - 1)
        return phi * q ** 2

    # Initial policy: everyone quotes 1/alpha (equilibrium)
    delta_a = np.ones(n) / alpha
    delta_b = np.ones(n) / alpha
    V = np.zeros(n)

    for iteration in range(max_iter):
        # --- Policy evaluation: solve linear system for V given quotes ---
        # HJB with fixed quotes:
        # rV(q) + psi(q) = rate_a(q) * [delta_a(q)*Delta + V(q-Delta) - V(q)]
        #                 + rate_b(q) * [delta_b(q)*Delta + V(q+Delta) - V(q)]
        #
        # Rearrange: [r + rate_a(q) + rate_b(q)] V(q) - rate_a(q) V(q-Delta) - rate_b(q) V(q+Delta)
        #            = rate_a(q)*delta_a(q)*Delta + rate_b(q)*delta_b(q)*Delta - psi(q)

        A = np.zeros((n, n))
        b = np.zeros(n)

        for i, q in enumerate(q_grid):
            rate_a_i = lambda_a * np.exp(-alpha * delta_a[i])
            rate_b_i = lambda_b * np.exp(-alpha * delta_b[i])

            # Diagonal: r + rate_a + rate_b
            A[i, i] = r + rate_a_i + rate_b_i

            # Off-diagonal: -rate_a * V(q - Delta)
            if i > 0:
                A[i, i - 1] = -rate_a_i
            else:
                # Boundary: V(q - Delta) = -psi(q - Delta)
                b[i] += rate_a_i * (-psi(q - Delta))

            # Off-diagonal: -rate_b * V(q + Delta)
            if i < n - 1:
                A[i, i + 1] = -rate_b_i
            else:
                b[i] += rate_b_i * (-psi(q + Delta))

            # RHS: spread profits - penalty
            b[i] += rate_a_i * delta_a[i] * Delta + rate_b_i * delta_b[i] * Delta - psi(q)

        # Solve linear system
        V_new = np.linalg.solve(A, b)

        # --- Policy improvement: update quotes from new V ---
        delta_a_new = np.zeros(n)
        delta_b_new = np.zeros(n)

        for i, q in enumerate(q_grid):
            # V(q - Delta)
            if i > 0:
                V_down = V_new[i - 1]
            else:
                V_down = -psi(q - Delta)

            # V(q + Delta)
            if i < n - 1:
                V_up = V_new[i + 1]
            else:
                V_up = -psi(q + Delta)

            # FOC: delta_a* = 1/alpha + (V(q-Delta) - V(q)) / Delta
            # Wait: let's rederive. Maximise rate_a * (delta_a*Delta + V(q-Delta) - V(q))
            # = exp(-alpha*delta_a) * (delta_a*Delta + jump_down)
            # FOC: -alpha * exp(-alpha*delta_a) * (delta_a*Delta + jump_down)
            #      + exp(-alpha*delta_a) * Delta = 0
            # => Delta = alpha * (delta_a*Delta + jump_down)
            # => delta_a = 1/alpha - jump_down / Delta  where jump_down = V(q-Delta) - V(q)
            # Wait: Delta / alpha = delta_a * Delta + jump_down
            # => delta_a = 1/alpha - jump_down / Delta

            jump_down = V_down - V_new[i]
            jump_up = V_up - V_new[i]

            # Ask: sell inventory. FOC gives delta = 1/alpha - jump_down/Delta
            # where jump_down = V(q-Delta) - V(q) < 0 for concave V near q=0
            # so delta_a > 1/alpha when holding positive inventory. Correct.
            delta_a_new[i] = 1.0 / alpha - jump_down / Delta

            # Bid: buy inventory. FOC gives delta = 1/alpha - jump_up/Delta
            # where jump_up = V(q+Delta) - V(q) < 0 for concave V at q > 0
            # so delta_b > 1/alpha when already holding positive inventory (wider bid = less buying)
            delta_b_new[i] = 1.0 / alpha - jump_up / Delta

            # Clamp to positive
            delta_a_new[i] = max(delta_a_new[i], 0.001)
            delta_b_new[i] = max(delta_b_new[i], 0.001)

        # Check convergence
        v_err = np.max(np.abs(V_new - V))
        p_err = np.max(np.abs(delta_a_new - delta_a)) + np.max(np.abs(delta_b_new - delta_b))

        V = V_new.copy()
        delta_a = delta_a_new.copy()
        delta_b = delta_b_new.copy()

        if v_err < tol and p_err < tol:
            print(f"  Converged in {iteration + 1} iterations (v_err={v_err:.2e}, p_err={p_err:.2e})")
            break

    if iteration == max_iter - 1:
        print(f"  Warning: did not converge (v_err={v_err:.2e}, p_err={p_err:.2e})")

    return q_grid, V, delta_a, delta_b


def main():
    out_dir = "plots/baseline"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for ptype in ["quadratic", "cubic", "exponential"]:
        kwargs = {"penalty_type": ptype}
        if ptype == "exponential":
            kwargs["gamma"] = 1.0
        print(f"\n--- {ptype} penalty ---")
        q, V, da, db = solve_hjb_policy_iteration(**kwargs)
        results[ptype] = (q, V, da, db)
        mid = len(V) // 2
        print(f"  V(0) = {V[mid]:.6f}")
        print(f"  spread at q=0: {da[mid] + db[mid]:.6f}")
        print(f"  delta_a(0) = {da[mid]:.6f}, delta_b(0) = {db[mid]:.6f}")

    # Plot value functions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ptype in zip(axes, ["quadratic", "cubic", "exponential"]):
        q, V, da, db = results[ptype]
        ax.plot(q, V, "b-", linewidth=2)
        ax.set_xlabel("Inventory $q$")
        ax.set_ylabel("$V(q)$")
        ax.set_title(f"{ptype.capitalize()} penalty (FD ground truth)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fd_value_functions.png"), dpi=150)
    plt.close()
    print(f"\nSaved {out_dir}/fd_value_functions.png")

    # Plot optimal quotes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ptype in zip(axes, ["quadratic", "cubic", "exponential"]):
        q, V, da, db = results[ptype]
        ax.plot(q, da, "g-", linewidth=1.5, label="$\\delta^a$ (ask)")
        ax.plot(q, db, "r-", linewidth=1.5, label="$\\delta^b$ (bid)")
        ax.plot(q, da + db, "b--", linewidth=1, alpha=0.7, label="Total spread")
        ax.axhline(y=2.0 / 1.5, color="gray", linestyle=":", alpha=0.5, label="$2/\\alpha$")
        ax.set_xlabel("Inventory $q$")
        ax.set_ylabel("Quote")
        ax.set_title(f"{ptype.capitalize()} penalty")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fd_optimal_quotes.png"), dpi=150)
    plt.close()
    print(f"Saved {out_dir}/fd_optimal_quotes.png")

    # Print comparison table
    print("\n=== Ground Truth Values (Finite Difference, Infinite Horizon) ===")
    print(f"{'Penalty':<15} {'V(0)':<12} {'spread(0)':<12} {'delta_a(0)':<12} {'delta_b(0)':<12}")
    print("-" * 63)
    for ptype in ["quadratic", "cubic", "exponential"]:
        q, V, da, db = results[ptype]
        mid = len(V) // 2
        print(f"{ptype:<15} {V[mid]:<12.6f} {da[mid]+db[mid]:<12.6f} {da[mid]:<12.6f} {db[mid]:<12.6f}")

    # Finite-horizon correction for comparison with Deep BSDE
    T, r_val = 1.0, 0.1
    finite_factor = (1.0 - np.exp(-r_val * T)) / r_val / (1.0 / r_val)
    print(f"\nFinite-horizon correction factor (T={T}, r={r_val}): {finite_factor:.4f}")
    print(f"Approximate V_T(0) for quadratic: {results['quadratic'][1][len(results['quadratic'][1])//2] * finite_factor:.6f}")


if __name__ == "__main__":
    main()
