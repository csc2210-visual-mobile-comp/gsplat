"""
Summarize experiment results into per-scene CSV tables.

Reads val_step*.json files from each result directory defined in
run_experiments.py and outputs one CSV per result_base (scene+resolution),
comparing PSNR / SSIM / LPIPS across all experiment variants.

Usage:
    python summarize_results.py                    # prints tables + writes CSVs
    python summarize_results.py --results-root .   # custom root (default: current dir)
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ── mirror the scene list from run_experiments.py ────────────────────────────
# Each entry: (data_dir, result_base_dir, data_factor)
SCENES: List[Tuple[str, str, int]] = [
    ("data/sedan", "results/sedan_f4", 4),
    # ("data/sedan", "results/sedan_f2", 2),
    # ("data/pitcher_scene007", "results/pitcher_scene007", 1),
]
# ─────────────────────────────────────────────────────────────────────────────

METRICS = ["psnr", "ssim", "lpips"]
OPTIONAL_METRICS = ["cc_psnr", "cc_ssim", "cc_lpips", "avg_rank", "num_GS"]
TRAIN_METRICS = ["train_mem_gb"]  # sourced from train_step*.json


def load_val_stats(result_dir: Path) -> Optional[Dict]:
    """Return the last val_step*.json found in result_dir/stats/, or None."""
    stats_dir = result_dir / "stats"
    if not stats_dir.exists():
        return None
    files = sorted(stats_dir.glob("val_step*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def load_train_stats(result_dir: Path) -> Optional[Dict]:
    """Return the last train_step*.json found in result_dir/stats/, or None."""
    stats_dir = result_dir / "stats"
    if not stats_dir.exists():
        return None
    files = sorted(stats_dir.glob("train_step*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def load_rank_dist(result_dir: Path) -> Optional[Dict[str, float]]:
    """Return per-rank percentage dict from the latest checkpoint, or None.

    Reads splats['current_ranks'] (shape [N, 1]) from the .pt checkpoint.
    Returns e.g. {2: 90.6, 8: 8.9, 32: 0.5} (rank -> % of Gaussians).
    """
    ckpt_dir = result_dir / "ckpts"
    if not ckpt_dir.exists():
        return None
    ckpt_files = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"))
    if not ckpt_files:
        return None
    ckpt = torch.load(ckpt_files[-1], map_location="cpu", weights_only=True)
    ranks = ckpt.get("splats", {}).get("current_ranks")
    if ranks is None:
        return None
    ranks = ranks.flatten().float()
    n = ranks.numel()
    values, counts = torch.unique(ranks, return_counts=True)
    return {int(v): round(100.0 * c.item() / n, 2) for v, c in zip(values, counts)}


def collect_rows(result_base: Path) -> List[Dict]:
    """Walk all subdirectories of result_base and collect one row per experiment."""
    rows = []
    if not result_base.exists():
        return rows

    for exp_dir in sorted(result_base.iterdir()):
        if not exp_dir.is_dir():
            continue
        stats = load_val_stats(exp_dir)
        train_stats = load_train_stats(exp_dir)
        row = {"experiment": exp_dir.name}
        if stats is None:
            row["status"] = "missing"
            for m in METRICS + OPTIONAL_METRICS + TRAIN_METRICS:
                row[m] = ""
        else:
            row["status"] = "ok"
            for m in METRICS:
                row[m] = f"{stats[m]:.4f}" if m in stats else ""
            for m in OPTIONAL_METRICS:
                val = stats.get(m)
                if val is None:
                    row[m] = ""
                elif m == "num_GS":
                    row[m] = str(int(val))
                else:
                    row[m] = f"{val:.4f}"
            # Peak GPU memory from training stats (separate JSON file)
            mem = (train_stats or {}).get("mem")
            row["train_mem_gb"] = f"{mem:.2f}" if mem is not None else ""
        rank_dist = load_rank_dist(exp_dir)
        row["_rank_dist"] = rank_dist  # raw dict, used to build dynamic columns
        rows.append(row)
    return rows


def _rank_columns(rows: List[Dict]) -> List[str]:
    """Return sorted rank-bucket column names found across all rows."""
    all_ranks: set = set()
    for row in rows:
        dist = row.get("_rank_dist") or {}
        all_ranks.update(dist.keys())
    return [f"r{r}_pct" for r in sorted(all_ranks)]


def _finalize_rows(rows: List[Dict], rank_cols: List[str]) -> List[Dict]:
    """Expand _rank_dist into individual r{rank}_pct columns."""
    for row in rows:
        dist = row.pop("_rank_dist", None) or {}
        for col in rank_cols:
            rank = int(col[1:-4])  # "r32_pct" -> 32
            row[col] = f"{dist[rank]:.2f}" if rank in dist else ""
    return rows


def print_table(result_base: str, rows: List[Dict], rank_cols: List[str]) -> None:
    if not rows:
        print(f"\n[{result_base}] — no results found")
        return

    cols = ["experiment", "status"] + METRICS + OPTIONAL_METRICS + TRAIN_METRICS + rank_cols
    widths = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)

    print(f"\n{'='*len(header)}")
    print(f"Scene: {result_base}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


def write_csv(csv_path: Path, rows: List[Dict], rank_cols: List[str]) -> None:
    cols = ["experiment", "status"] + METRICS + OPTIONAL_METRICS + TRAIN_METRICS + rank_cols
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → saved {csv_path}")


def main() -> None:
    # default root = same directory as this script so it works from any cwd
    results_root = Path(__file__).parent
    if "--results-root" in sys.argv:
        idx = sys.argv.index("--results-root")
        results_root = Path(sys.argv[idx + 1]).resolve()

    for (_data_dir, result_base_str, _factor) in SCENES:
        result_base = results_root / result_base_str
        rows = collect_rows(result_base)
        rank_cols = _rank_columns(rows)
        _finalize_rows(rows, rank_cols)
        print_table(result_base_str, rows, rank_cols)

        if rows:
            csv_path = result_base / "summary.csv"
            result_base.mkdir(parents=True, exist_ok=True)
            write_csv(csv_path, rows, rank_cols)


if __name__ == "__main__":
    main()
