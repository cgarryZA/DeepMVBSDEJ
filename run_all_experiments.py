#!/usr/bin/env python
"""
Complete experiment suite for the preprint.

Runs all experiments needed for the paper:
1. Multi-seed continuous surrogate (5 seeds)
2. Multi-seed jump model (5 seeds)
3. Ablation: Type 1 vs Type 3 (no MF vs MF)
4. Stress tests across penalties (quadratic/cubic/exponential) with multi-seed
5. Breaking point finder (exponential gamma sweep)
6. Parameter sensitivity (alpha, lambda, N_T)

Usage:
    python run_all_experiments.py              # full suite
    python run_all_experiments.py --quick      # 500 iter, 2 seeds
    python run_all_experiments.py --device cpu  # force CPU
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from registry import EQUATION_REGISTRY
import equations
from solver import ContXiongLOBSolver, ContXiongLOBJumpSolver


def run_single(config_path, overrides, device, label=""):
    """Run a single experiment and return the result dict."""
    config = Config.from_json(config_path)
    for key, val in overrides.items():
        parts = key.split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)

    eqn_name = config.eqn.eqn_name
    bsde = EQUATION_REGISTRY[eqn_name](config.eqn)

    if eqn_name == "contxiong_lob":
        solver = ContXiongLOBSolver(config, bsde, device=device)
    elif eqn_name == "contxiong_lob_jump":
        solver = ContXiongLOBJumpSolver(config, bsde, device=device)
    else:
        raise ValueError(f"Unknown equation: {eqn_name}")

    config.net.verbose = False
    result = solver.train()
    y0 = result["y0"]
    loss = result["final_loss"]
    z_max = result["history"][-1, 3] if result["history"].shape[1] > 3 else 0.0
    print(f"  {label}: Y0={y0:.4f}, loss={loss:.4e}, z_max={z_max:.4f}")
    return result


def run_multi_seed(config_path, overrides, device, n_seeds, label=""):
    """Run experiment with multiple seeds, return stats."""
    y0s, losses, z_maxs = [], [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 42 + 7)
        np.random.seed(seed * 42 + 7)
        r = run_single(config_path, overrides, device, label=f"{label} seed={seed}")
        y0s.append(r["y0"])
        losses.append(r["final_loss"])
        if r["history"].shape[1] > 3:
            z_maxs.append(r["history"][-1, 3])
    return {
        "y0_mean": np.mean(y0s), "y0_std": np.std(y0s),
        "loss_mean": np.mean(losses), "loss_std": np.std(losses),
        "z_max_mean": np.mean(z_maxs) if z_maxs else 0,
        "z_max_std": np.std(z_maxs) if z_maxs else 0,
        "y0s": y0s, "losses": losses,
    }


def save_results(results, path):
    """Save results dict to JSON."""
    # Convert numpy types to native Python
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    n_iters = 500 if args.quick else 3000
    n_iters_long = 1000 if args.quick else 5000
    n_seeds = 2 if args.quick else 5
    base_continuous = "configs/lob_d2.json"
    base_jump = "configs/lob_d2_jump.json"

    all_results = {}
    start = time.time()

    # ================================================================
    # 1. Multi-seed continuous surrogate
    # ================================================================
    print(f"\n{'='*60}")
    print(f"1. Multi-seed continuous surrogate ({n_seeds} seeds, {n_iters_long} iter)")
    print(f"{'='*60}")
    r = run_multi_seed(base_continuous, {
        "net.opt_config1.num_iterations": n_iters_long,
        "net.logging_frequency": n_iters_long,
    }, device, n_seeds, "continuous")
    all_results["continuous_multiseed"] = r
    print(f"  => Y0 = {r['y0_mean']:.4f} ± {r['y0_std']:.4f}")

    # ================================================================
    # 2. Multi-seed jump model
    # ================================================================
    print(f"\n{'='*60}")
    print(f"2. Multi-seed jump model ({n_seeds} seeds, {n_iters_long} iter)")
    print(f"{'='*60}")
    r = run_multi_seed(base_jump, {
        "net.opt_config1.num_iterations": n_iters_long,
        "net.logging_frequency": n_iters_long,
        "net.batch_size": 512,
    }, device, n_seeds, "jump")
    all_results["jump_multiseed"] = r
    print(f"  => Y0 = {r['y0_mean']:.4f} ± {r['y0_std']:.4f}")

    # ================================================================
    # 3. Ablation: Type 1 (no MF) vs Type 3 (MF)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"3. Ablation: Type 1 vs Type 3")
    print(f"{'='*60}")
    for mf_type in [1, 3]:
        label = f"type{mf_type}"
        freq = 9999999 if mf_type == 1 else 100
        r = run_multi_seed(base_continuous, {
            "eqn.type": mf_type,
            "net.opt_config1.num_iterations": n_iters,
            "net.opt_config1.freq_update_drift": freq,
            "net.logging_frequency": n_iters,
        }, device, n_seeds, label)
        all_results[f"ablation_{label}"] = r
        print(f"  => {label}: Y0 = {r['y0_mean']:.4f} ± {r['y0_std']:.4f}")

    # ================================================================
    # 4. Stress tests with multi-seed
    # ================================================================
    print(f"\n{'='*60}")
    print(f"4. Stress tests ({n_seeds} seeds each)")
    print(f"{'='*60}")
    for ptype in ["quadratic", "cubic", "exponential"]:
        overrides = {
            "eqn.penalty_type": ptype,
            "eqn.type": 1,
            "net.opt_config1.num_iterations": n_iters,
            "net.opt_config1.freq_update_drift": 9999999,
            "net.logging_frequency": n_iters,
        }
        if ptype == "exponential":
            overrides["eqn.gamma"] = 1.0
        r = run_multi_seed(base_continuous, overrides, device, n_seeds, ptype)
        all_results[f"stress_{ptype}"] = r
        print(f"  => {ptype}: Y0 = {r['y0_mean']:.4f} ± {r['y0_std']:.4f}, "
              f"z_max = {r['z_max_mean']:.4f} ± {r['z_max_std']:.4f}")

    # ================================================================
    # 5. Breaking point: exponential gamma sweep
    # ================================================================
    print(f"\n{'='*60}")
    print(f"5. Breaking point: gamma sweep")
    print(f"{'='*60}")
    bp_results = []
    for gamma in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        r = run_single(base_continuous, {
            "eqn.penalty_type": "exponential",
            "eqn.gamma": gamma,
            "eqn.type": 1,
            "net.opt_config1.num_iterations": n_iters,
            "net.opt_config1.freq_update_drift": 9999999,
            "net.logging_frequency": n_iters,
        }, device, label=f"gamma={gamma}")
        z_max = r["history"][-1, 3] if r["history"].shape[1] > 3 else 0
        converged = not (np.isnan(r["final_loss"]) or r["final_loss"] > 10.0)
        bp_results.append({
            "gamma": gamma, "y0": r["y0"], "loss": r["final_loss"],
            "z_max": float(z_max), "converged": converged,
        })
    all_results["breaking_point_gamma"] = bp_results

    # Breaking point: phi sweep
    print(f"\n  Phi sweep:")
    bp_phi = []
    for phi in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        r = run_single(base_continuous, {
            "eqn.phi": phi,
            "eqn.type": 1,
            "net.opt_config1.num_iterations": n_iters,
            "net.opt_config1.freq_update_drift": 9999999,
            "net.logging_frequency": n_iters,
        }, device, label=f"phi={phi}")
        z_max = r["history"][-1, 3] if r["history"].shape[1] > 3 else 0
        converged = not (np.isnan(r["final_loss"]) or r["final_loss"] > 10.0)
        bp_phi.append({
            "phi": phi, "y0": r["y0"], "loss": r["final_loss"],
            "z_max": float(z_max), "converged": converged,
        })
    all_results["breaking_point_phi"] = bp_phi

    # ================================================================
    # 6. Parameter sensitivity
    # ================================================================
    print(f"\n{'='*60}")
    print(f"6. Parameter sensitivity")
    print(f"{'='*60}")

    # Alpha sensitivity
    print("  Alpha sweep:")
    alpha_results = []
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        r = run_single(base_continuous, {
            "eqn.alpha": alpha,
            "eqn.type": 1,
            "net.opt_config1.num_iterations": n_iters,
            "net.opt_config1.freq_update_drift": 9999999,
            "net.logging_frequency": n_iters,
        }, device, label=f"alpha={alpha}")
        alpha_results.append({
            "alpha": alpha, "y0": r["y0"], "loss": r["final_loss"],
            "expected_spread": 2.0 / alpha,
        })
    all_results["sensitivity_alpha"] = alpha_results

    # Lambda sensitivity
    print("  Lambda sweep:")
    lambda_results = []
    for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
        r = run_single(base_continuous, {
            "eqn.lambda_a": lam, "eqn.lambda_b": lam,
            "eqn.type": 1,
            "net.opt_config1.num_iterations": n_iters,
            "net.opt_config1.freq_update_drift": 9999999,
            "net.logging_frequency": n_iters,
        }, device, label=f"lambda={lam}")
        lambda_results.append({
            "lambda": lam, "y0": r["y0"], "loss": r["final_loss"],
        })
    all_results["sensitivity_lambda"] = lambda_results

    # N_T sensitivity
    print("  N_T sweep:")
    nt_results = []
    for nt in [10, 25, 50, 100, 200]:
        r = run_single(base_continuous, {
            "eqn.num_time_interval": nt,
            "eqn.type": 1,
            "net.opt_config1.num_iterations": n_iters,
            "net.opt_config1.freq_update_drift": 9999999,
            "net.logging_frequency": n_iters,
        }, device, label=f"N_T={nt}")
        nt_results.append({
            "N_T": nt, "y0": r["y0"], "loss": r["final_loss"],
        })
    all_results["sensitivity_NT"] = nt_results

    # ================================================================
    # Save all results
    # ================================================================
    elapsed = time.time() - start
    all_results["metadata"] = {
        "device": str(device),
        "n_seeds": n_seeds,
        "n_iters": n_iters,
        "n_iters_long": n_iters_long,
        "elapsed_seconds": elapsed,
        "quick": args.quick,
    }
    save_results(all_results, os.path.join(args.out_dir, "all_experiments.json"))

    print(f"\n{'='*60}")
    print(f"All experiments complete in {elapsed/60:.1f} minutes")
    print(f"Results saved to {args.out_dir}/all_experiments.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
