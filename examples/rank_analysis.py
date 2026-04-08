#!/usr/bin/env python3
"""
Post-hoc analysis of Gaussian rank predictions across training runs and checkpoints.

Two modes:

  single  — analyse one run's checkpoint directory (score distributions, bucket %)
  compare — compare multiple runs against a gold-standard run/step (quota fractions)

Usage:
  # Single-run analysis
  python rank_analysis.py single <checkpoint_dir> <output_dir>

  # Multi-run comparison (gold = sh3 run at step 15000)
  python rank_analysis.py compare \\
      --runs sh3=results/counter_sh3 lora32=results/counter_lora32 lora2=results/counter_lora2 \\
      --gold_run sh3 --gold_step 15000 \\
      --output_dir results/counter_comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rank_analysis_utils import CheckpointData, RankAnalysis


# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_checkpoints(checkpoint_dir: Path) -> Dict[int, CheckpointData]:
    """Load all CheckpointData files saved under <result_dir>/rank_analysis/."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = {}
    for metadata_file in sorted(checkpoint_dir.glob("step_*_metadata.json")):
        step_str = metadata_file.stem.replace("step_", "").replace("_metadata", "")
        try:
            step = int(step_str)
            checkpoints[step] = CheckpointData.load(checkpoint_dir, step)
            print(f"  loaded step {step} ({checkpoints[step].num_gaussians} Gaussians)")
        except Exception as e:
            print(f"  warning: skipping step {step_str}: {e}")
    return checkpoints


# ── Single-run mode ────────────────────────────────────────────────────────────

def run_single(args):
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_buckets = tuple(args.rank_buckets)

    print(f"\nLoading checkpoints from {checkpoint_dir}")
    checkpoints = load_checkpoints(checkpoint_dir)
    if not checkpoints:
        print("Error: no checkpoints found.")
        return
    print(f"Steps: {sorted(checkpoints.keys())}")

    # Score distributions
    all_score_methods = set()
    for ckpt in checkpoints.values():
        all_score_methods.update(ckpt.scores.keys())

    for sm in sorted(all_score_methods):
        relevant = {s: c for s, c in checkpoints.items() if sm in c.scores}
        if relevant:
            RankAnalysis.plot_score_distributions(
                relevant, sm, rank_buckets,
                output_dir / f"score_dist_{sm}.png",
            )

    # Bucket percentage evolution
    all_combos = set()
    for ckpt in checkpoints.values():
        for cm, sdata in ckpt.assignments.items():
            for sm in sdata:
                all_combos.add((cm, sm))

    for cm, sm in sorted(all_combos):
        RankAnalysis.plot_bucket_percentages(
            checkpoints, cm, sm, rank_buckets,
            output_dir / f"buckets_{cm}_{sm}.png",
        )

    # Percentile correlations vs the latest step
    steps = sorted(checkpoints.keys())
    if len(steps) >= 2:
        latest = steps[-1]
        rows = []
        for early_step in steps[:-1]:
            for sm in all_score_methods:
                early_ckpt = checkpoints[early_step]
                late_ckpt  = checkpoints[latest]
                if sm not in early_ckpt.scores or sm not in late_ckpt.scores:
                    continue
                rho, pval = RankAnalysis.percentile_correlation(
                    early_ckpt.scores[sm], late_ckpt.scores[sm]
                )
                rows.append({
                    "early_step": early_step, "late_step": latest,
                    "score_method": sm,
                    "spearman_rho": round(rho, 4) if not np.isnan(rho) else None,
                    "p_value": round(pval, 6) if not np.isnan(pval) else None,
                })
        if rows:
            df = pd.DataFrame(rows)
            csv_path = output_dir / "percentile_correlations.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nPercentile correlations vs step {latest}:")
            print(df.to_string(index=False))
            print(f"Saved {csv_path}")

    print(f"\nSingle-run analysis complete → {output_dir}")


# ── Multi-run comparison mode ──────────────────────────────────────────────────

def run_compare(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_buckets = tuple(args.rank_buckets)

    # Parse --runs name=path name=path ...
    runs: Dict[str, Dict[int, CheckpointData]] = {}
    for entry in args.runs:
        if "=" not in entry:
            print(f"Error: --runs entries must be name=path, got: {entry!r}")
            return
        name, path = entry.split("=", 1)
        analysis_dir = Path(path) / "rank_analysis"
        print(f"\nLoading {name!r} from {analysis_dir}")
        ckpts = load_checkpoints(analysis_dir)
        if ckpts:
            runs[name] = ckpts
        else:
            print(f"  warning: no checkpoints found for {name!r}, skipping.")

    if not runs:
        print("Error: no runs loaded.")
        return

    if args.gold_run not in runs:
        print(f"Error: gold_run {args.gold_run!r} not found in loaded runs ({list(runs)})")
        return

    gold_step = args.gold_step
    if gold_step not in runs[args.gold_run]:
        available = sorted(runs[args.gold_run].keys())
        print(f"Error: gold_step {gold_step} not in {args.gold_run} steps {available}")
        return

    df = RankAnalysis.plot_multi_run_comparison(
        runs=runs,
        gold_run=args.gold_run,
        gold_step=gold_step,
        rank_buckets=rank_buckets,
        output_dir=output_dir,
    )

    if not df.empty:
        # Summary: for each run/score/cluster method, how close are early steps to gold?
        gold_rows = df[
            (df["run"] == args.gold_run) & (df["step"] == gold_step)
        ][["cluster_method", "score_method", "bucket", "fraction"]].rename(
            columns={"fraction": "gold_fraction"}
        )
        merged = df[df["run"] != args.gold_run].merge(
            gold_rows, on=["cluster_method", "score_method", "bucket"], how="left"
        )
        merged["abs_error"] = (merged["fraction"] - merged["gold_fraction"]).abs()
        summary = (
            merged.groupby(["run", "step", "cluster_method", "score_method"])["abs_error"]
            .mean()
            .reset_index()
            .rename(columns={"abs_error": "mean_abs_error_vs_gold"})
            .sort_values(["run", "cluster_method", "score_method", "step"])
        )
        summary_path = output_dir / "prediction_error_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nPrediction error vs {args.gold_run} @ step {gold_step}:")
        print(summary.to_string(index=False))
        print(f"Saved {summary_path}")

    print(f"\nComparison complete → {output_dir}")


# ── Multi-scene comparison mode ───────────────────────────────────────────────

def run_multi_scene(args):
    """Answer two questions across all scenes:

    1. Which scoring method best predicts the gold-standard quota early in training?
       → mean absolute error vs gold, averaged across scenes, per (scorer, step).

    2. How scene-dependent are the gold-standard quota fractions?
       → side-by-side bar chart of gold quotas across all scenes.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_buckets = tuple(args.rank_buckets)
    results_base = Path(args.results_base)
    gold_step = 14999
    gold_run = "sh3"
    bucket_labels = ["low", "mid", "high"]
    bar_colors = ["#4878d0", "#6acc65", "#d65f5f"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Load per-scene quota_fractions.csv (produced by compare mode) ──────────
    all_fracs: List[pd.DataFrame] = []
    for scene_name in args.scenes:
        csv_path = results_base / scene_name / "comparison" / "quota_fractions.csv"
        if not csv_path.exists():
            print(f"  warning: {csv_path} not found, skipping {scene_name}")
            continue
        df = pd.read_csv(csv_path)
        df["scene"] = scene_name
        all_fracs.append(df)

    if not all_fracs:
        print("Error: no quota_fractions.csv files found. Run compare mode first.")
        return

    df_all = pd.concat(all_fracs, ignore_index=True)
    df_all.to_csv(output_dir / "all_scenes_quota_fractions.csv", index=False)

    # ── Q1: Which scorer predicts earliest? ────────────────────────────────────
    # For each non-gold run, compute abs error vs gold per (scene, scorer, step)
    gold_ref = df_all[(df_all["run"] == gold_run) & (df_all["step"] == gold_step)][
        ["scene", "cluster_method", "score_method", "bucket", "fraction"]
    ].rename(columns={"fraction": "gold_fraction"})

    non_gold = df_all[df_all["run"] != gold_run].copy()
    merged = non_gold.merge(gold_ref, on=["scene", "cluster_method", "score_method", "bucket"], how="left")
    merged["abs_error"] = (merged["fraction"] - merged["gold_fraction"]).abs()

    # Average across scenes and buckets → (run, cluster_method, score_method, step)
    predictor_summary = (
        merged.groupby(["run", "cluster_method", "score_method", "step"])["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "mean_abs_error"})
        .sort_values(["run", "cluster_method", "score_method", "step"])
    )
    predictor_summary.to_csv(output_dir / "predictor_summary.csv", index=False)
    print("\nPredictor summary (avg error across scenes):")
    print(predictor_summary.to_string(index=False))

    # Plot: for each cluster_method, one figure with lines per (run × score_method)
    for cm in predictor_summary["cluster_method"].unique():
        sub = predictor_summary[predictor_summary["cluster_method"] == cm]
        combos = [(r, sm) for r in sub["run"].unique() for sm in sub["score_method"].unique()]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10.colors
        for i, (run_name, sm) in enumerate(combos):
            row = sub[(sub["run"] == run_name) & (sub["score_method"] == sm)]
            if row.empty:
                continue
            ax.plot(row["step"], row["mean_abs_error"], marker="o",
                    color=colors[i % len(colors)], label=f"{run_name} / {sm}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean abs error vs gold (avg across scenes)")
        ax.set_title(f"Predictor quality — {cm} clustering")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = output_dir / f"predictor_quality_{cm}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")

    # ── Q2: How scene-dependent are the gold quotas? ───────────────────────────
    gold_quotas = df_all[
        (df_all["run"] == gold_run) & (df_all["step"] == gold_step)
    ].copy()

    for cm in gold_quotas["cluster_method"].unique():
        score_methods = gold_quotas["score_method"].unique()
        n_sm = len(score_methods)
        fig, axes = plt.subplots(1, n_sm, figsize=(5 * n_sm, 5), squeeze=False)
        fig.suptitle(f"Gold-standard quotas by scene — {cm} clustering\n"
                     f"({gold_run} @ step {gold_step})")
        for ax, sm in zip(axes[0], score_methods):
            sub = gold_quotas[
                (gold_quotas["cluster_method"] == cm) &
                (gold_quotas["score_method"] == sm)
            ]
            scenes_present = sorted(sub["scene"].unique())
            x = np.arange(len(scenes_present))
            width = 0.25
            for b_idx, (label, rank) in enumerate(zip(bucket_labels, rank_buckets)):
                vals = [
                    sub[(sub["scene"] == s) & (sub["bucket"] == label)]["fraction"].values
                    for s in scenes_present
                ]
                vals = [v[0] if len(v) > 0 else 0.0 for v in vals]
                bars = ax.bar(x + b_idx * width, vals, width, label=label,
                              color=bar_colors[b_idx], alpha=0.85)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=6)
            ax.set_xticks(x + width)
            ax.set_xticklabels(scenes_present, rotation=20, ha="right")
            ax.set_ylabel("Fraction of Gaussians")
            ax.set_ylim(0, 1.1)
            ax.set_title(sm)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = output_dir / f"scene_quota_variability_{cm}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")

    print(f"\nMulti-scene analysis complete → {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank analysis: single-run, multi-run comparison, or multi-scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rank_buckets", type=int, nargs=3, default=[2, 8, 32],
        help="Rank bucket values (default: 2 8 32)",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # single mode
    p_single = sub.add_parser("single", help="Analyse one run")
    p_single.add_argument("checkpoint_dir", type=str,
                          help="<result_dir>/rank_analysis/ directory")
    p_single.add_argument("output_dir", type=str, help="Where to write plots")

    # compare mode
    p_compare = sub.add_parser("compare", help="Compare multiple runs vs gold standard")
    p_compare.add_argument(
        "--runs", nargs="+", required=True, metavar="NAME=RESULT_DIR",
        help="One or more name=<result_dir> pairs (rank_analysis/ subdir is appended)",
    )
    p_compare.add_argument("--gold_run", required=True,
                           help="Name of the gold-standard run (must be in --runs)")
    p_compare.add_argument("--gold_step", type=int, required=True,
                           help="Step within gold_run to use as reference")
    p_compare.add_argument("--output_dir", required=True, help="Where to write plots")

    # multi_scene mode
    p_ms = sub.add_parser("multi_scene", help="Cross-scene predictor and quota variability")
    p_ms.add_argument(
        "--scenes", nargs="+", required=True,
        help="Scene names (must match subdirs under --results_base)",
    )
    p_ms.add_argument(
        "--results_base", required=True,
        help="Base directory containing per-scene result folders",
    )
    p_ms.add_argument("--output_dir", required=True, help="Where to write plots")

    args = parser.parse_args()

    if args.mode == "single":
        run_single(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "multi_scene":
        run_multi_scene(args)


if __name__ == "__main__":
    main()
