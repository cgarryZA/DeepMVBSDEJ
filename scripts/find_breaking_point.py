#!/usr/bin/env python
"""
Binary search for the solver's breaking point.

Finds the exact parameter threshold where max|Z_t| explodes,
indicating loss of Lipschitz continuity in the backward driver.

Sweeps a parameter (phi, gamma, or sigma_s) and reports:
  - The critical value where the solver destabilises
  - max|Z_t| at each tested value
  - Whether the loss converged or diverged

Usage:
    python find_breaking_point.py --param phi --lo 0.001 --hi 5.0
    python find_breaking_point.py --param gamma --lo 0.1 --hi 5.0 --penalty exponential
    python find_breaking_point.py --param sigma_s --lo 0.1 --hi 3.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import logging
import numpy as np
import torch

from config import Config
from registry import EQUATION_REGISTRY
import equations
from solver import ContXiongLOBSolver


def test_parameter(config_path, param_name, param_value, penalty_type="quadratic",
                   num_iters=500, device="cpu"):
    """Run a short training with given parameter value. Returns (converged, z_max, loss, y0)."""
    config = Config.from_json(config_path)
    config.net.opt_config1.num_iterations = num_iters
    config.net.opt_config1.freq_update_drift = 9999999  # no MF updates for speed
    config.net.logging_frequency = num_iters  # only log at end
    config.net.valid_size = 512
    config.net.batch_size = 128
    config.net.verbose = False
    config.eqn.type = 1  # no competition — isolate the penalty effect
    config.eqn.penalty_type = penalty_type

    # Set the parameter under test
    setattr(config.eqn, param_name, param_value)

    bsde = EQUATION_REGISTRY["contxiong_lob"](config.eqn)
    dev = torch.device(device)
    solver = ContXiongLOBSolver(config, bsde, device=dev)

    try:
        result = solver.train()
        loss = result["final_loss"]
        y0 = result["y0"]
        z_max = result["history"][-1, 3] if result["history"].shape[1] > 3 else 0.0

        # Convergence criteria: loss < 1.0 and z_max < threshold
        converged = loss < 1.0 and z_max < 50.0 and not np.isnan(loss)
        return converged, z_max, loss, y0
    except Exception as e:
        logging.warning("  Exception at %s=%.4f: %s" % (param_name, param_value, e))
        return False, float("inf"), float("inf"), 0.0


def binary_search(config_path, param_name, lo, hi, penalty_type, num_iters, device,
                  tol=0.01, max_steps=15):
    """Binary search for the critical parameter value."""
    print(f"\n{'='*60}")
    print(f"Breaking Point Search: {param_name}")
    print(f"Range: [{lo}, {hi}], penalty: {penalty_type}")
    print(f"{'='*60}\n")

    results = []

    # First test the endpoints
    for val in [lo, hi]:
        ok, z_max, loss, y0 = test_parameter(
            config_path, param_name, val, penalty_type, num_iters, device
        )
        status = "STABLE" if ok else "UNSTABLE"
        print(f"  {param_name}={val:.4f}:  {status}  max|Z|={z_max:.4f}  loss={loss:.4e}")
        results.append((val, ok, z_max, loss))

    if results[0][1] == results[1][1]:
        if results[0][1]:
            print(f"\nBoth endpoints stable — increase hi beyond {hi}")
        else:
            print(f"\nBoth endpoints unstable — decrease lo below {lo}")
        return results

    # Binary search
    for step in range(max_steps):
        mid = (lo + hi) / 2
        if hi - lo < tol:
            break

        ok, z_max, loss, y0 = test_parameter(
            config_path, param_name, mid, penalty_type, num_iters, device
        )
        status = "STABLE" if ok else "UNSTABLE"
        print(f"  {param_name}={mid:.4f}:  {status}  max|Z|={z_max:.4f}  loss={loss:.4e}")
        results.append((mid, ok, z_max, loss))

        if ok:
            lo = mid
        else:
            hi = mid

    critical = (lo + hi) / 2
    print(f"\n{'='*60}")
    print(f"CRITICAL VALUE: {param_name} ≈ {critical:.4f}")
    print(f"Stable for {param_name} < {lo:.4f}")
    print(f"Unstable for {param_name} > {hi:.4f}")
    print(f"{'='*60}\n")

    # Print full sweep table
    results.sort(key=lambda r: r[0])
    print(f"{'Value':>10} {'Status':>10} {'max|Z|':>10} {'Loss':>12}")
    print("-" * 46)
    for val, ok, z_max, loss in results:
        status = "STABLE" if ok else "UNSTABLE"
        print(f"{val:10.4f} {status:>10} {z_max:10.4f} {loss:12.4e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Find solver breaking point")
    parser.add_argument("--config", default="configs/lob_d2.json")
    parser.add_argument("--param", default="phi", help="Parameter to sweep: phi, gamma, sigma_s")
    parser.add_argument("--lo", type=float, default=0.001)
    parser.add_argument("--hi", type=float, default=5.0)
    parser.add_argument("--penalty", default="quadratic", help="quadratic, cubic, exponential")
    parser.add_argument("--iters", type=int, default=500, help="Iterations per test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tol", type=float, default=0.01)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    torch.set_default_dtype(torch.float64)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    binary_search(
        args.config, args.param, args.lo, args.hi,
        args.penalty, args.iters, device, args.tol,
    )


if __name__ == "__main__":
    main()
