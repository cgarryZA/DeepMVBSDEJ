#!/usr/bin/env python
"""Build the arXiv preprint PDF.

1. Copies plots from the solver's output directory into report/preprint/figures/
2. Compiles main.tex with pdflatex (two passes for references)

Usage:
    python report/preprint/build.py
    python report/preprint/build.py --plots-dir plots/final
"""

import argparse
import os
import shutil
import subprocess
import sys


FIGURES = [
    "convergence.png",
    "spread_heatmap.png",
    "value_function.png",
    "sample_paths.png",
    "inventory_distribution.png",
    "quoting_strategy.png",
    "value_surface_3d.png",
    "z_gradient_surface_3d.png",
    "z_max_evolution.png",
]

# Additional figures from other directories
EXTRA_FIGURES = [
    ("plots/comparison/fd_vs_deepbsde.png", "fd_vs_deepbsde.png"),
    ("plots/baseline/fd_value_functions.png", "fd_value_functions.png"),
    ("plots/baseline/fd_optimal_quotes.png", "fd_optimal_quotes.png"),
    ("plots/experiments_full/multiseed_comparison.png", "multiseed_comparison.png"),
    ("plots/experiments_full/ablation_meanfield.png", "ablation_meanfield.png"),
    ("plots/experiments_full/stress_multiseed.png", "stress_multiseed.png"),
    ("plots/experiments_full/breaking_point.png", "breaking_point.png"),
    ("plots/experiments_full/sensitivity.png", "sensitivity.png"),
]


def main():
    parser = argparse.ArgumentParser(description="Build preprint PDF")
    parser.add_argument("--plots-dir", default="plots/final",
                        help="Directory containing generated plots")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preprint_dir = os.path.join(repo_root, "report", "preprint")
    fig_dir = os.path.join(preprint_dir, "figures")
    plots_dir = os.path.join(repo_root, args.plots_dir)

    # Copy figures
    os.makedirs(fig_dir, exist_ok=True)
    copied = 0
    for fig in FIGURES:
        src = os.path.join(plots_dir, fig)
        dst = os.path.join(fig_dir, fig)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"Warning: {src} not found, skipping")

    # Copy extra figures from other directories
    for src_rel, dst_name in EXTRA_FIGURES:
        src = os.path.join(repo_root, src_rel)
        dst = os.path.join(fig_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"Warning: {src} not found, skipping")
    print(f"Copied {copied} figures total to {fig_dir}")

    # Compile LaTeX (two passes)
    for pass_num in (1, 2):
        print(f"\n--- pdflatex pass {pass_num} ---")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=preprint_dir,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Check if it's just warnings (common with references)
            if pass_num == 2:
                print("pdflatex warnings/errors:")
                for line in result.stdout.split("\n"):
                    if "!" in line or "Error" in line:
                        print(f"  {line}")

    pdf_path = os.path.join(preprint_dir, "main.pdf")
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"\nBuild complete: {pdf_path} ({size_kb:.0f} KB)")
    else:
        print("\nBuild FAILED — no PDF produced")
        sys.exit(1)


if __name__ == "__main__":
    main()
