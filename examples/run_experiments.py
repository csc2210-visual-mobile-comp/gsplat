"""
Experiment runner for gsplat with LoRA SH correction.

Runs the following suite per (scene, resolution):
  2.1  No LoRA, SH=1
  2.2  No LoRA, SH=2
  2.3  No LoRA, SH=3
  2.4  LoRA static,  rank=2
  2.5  LoRA static,  rank=16
  2.6  LoRA dynamic, per (strategy, threshold) pair in LORA_DYNAMIC_CONFIGS

To run:
    python run_experiments.py           # full suite
    python run_experiments.py --dry-run # print commands without executing

After training, summarize results with:
    python summarize_results.py
"""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# =============================================================================
# EXPERIMENT CONFIGURATION — edit this section
# =============================================================================

# Each entry: (data_dir, result_base_dir, data_factor)
#   data_factor 1 = original resolution, 4 = images_4 downsampled
SCENES: List[Tuple[str, str, int]] = [
    ("data/pitcher_scene007", "results/pitcher_scene007", 1),  # downsampled
    # ("data/360_v2/counter", "results/counter_f1", 1),  # original resolution
]

# Training duration
MAX_STEPS: int = 7_000

# SH degree used for all LoRA experiments
LORA_SH_DEGREE: int = 3

# --- Which experiment groups to run ---
RUN_NO_LORA:      bool = False   # 2.1-2.3: baselines at SH=1, 2, 3
RUN_LORA_STATIC:  bool = False   # 2.4-2.5: static LoRA rank=2 and rank=16
RUN_LORA_DYNAMIC: bool = True   # 2.6:     dynamic LoRA

# Dynamic LoRA shared parameters
LORA_MAX_RANK:      int   = 32
LORA_MIN_RANK:      int   = 2
LORA_RANK_INTERVAL: int   = 100
LORA_QUOTA: Tuple[float, float, float] = (0.4, 0.4, 0.2)
LORA_STATS_K:       float = 1.0
LORA_WARMUP_CYCLES: int   = 5
LORA_KMEANS_ITERS:  int   = 10

# Each entry: (strategy, threshold)
#   strategy  : "gradient" | "opacity_grad" | "sh_energy"
#   threshold : "percentile" | "stats" | "kmeans"
LORA_DYNAMIC_CONFIGS: List[Tuple[str, str]] = [
    ("gradient",  "percentile"),
    # ("gradient",  "kmeans"),
    ("sh_energy", "percentile"),
    # ("sh_energy", "kmeans"),
]

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


@dataclass
class Run:
    label: str
    data_dir: Path
    result_dir: Path
    data_factor: int
    sh_degree: int
    lora_mode: str               # "none" | "static" | "dynamic"
    lora_rank: Optional[int] = None
    lora_rank_strategy: str = "gradient"
    lora_rank_threshold: str = "percentile"
    lora_stats_k: float = LORA_STATS_K
    lora_warmup_cycles: int = LORA_WARMUP_CYCLES
    lora_kmeans_iters: int = LORA_KMEANS_ITERS
    lora_max_rank: int = LORA_MAX_RANK
    lora_min_rank: int = LORA_MIN_RANK
    lora_rank_interval: int = LORA_RANK_INTERVAL
    lora_quota: Tuple[float, float, float] = field(default_factory=lambda: LORA_QUOTA)


def build_runs() -> List[Run]:
    runs: List[Run] = []
    for (data_dir, result_base, data_factor) in SCENES:
        base = Path(result_base)

        if RUN_NO_LORA:
            for sh in [1, 2, 3]:
                runs.append(Run(
                    label=f"{result_base} | no_lora sh={sh}",
                    data_dir=Path(data_dir),
                    result_dir=base / f"no_lora_sh{sh}",
                    data_factor=data_factor,
                    sh_degree=sh,
                    lora_mode="none",
                ))

        if RUN_LORA_STATIC:
            for rank in [2, 16]:
                runs.append(Run(
                    label=f"{result_base} | lora_static rank={rank} sh={LORA_SH_DEGREE}",
                    data_dir=Path(data_dir),
                    result_dir=base / f"lora_static_r{rank}",
                    data_factor=data_factor,
                    sh_degree=LORA_SH_DEGREE,
                    lora_mode="static",
                    lora_rank=rank,
                ))

        if RUN_LORA_DYNAMIC:
            quota_tag = "_".join(str(int(q * 10)) for q in LORA_QUOTA)
            for (strategy, threshold) in LORA_DYNAMIC_CONFIGS:
                runs.append(Run(
                    label=f"{result_base} | lora_dynamic {strategy}/{threshold} quota={quota_tag} max={LORA_MAX_RANK} sh={LORA_SH_DEGREE}",
                    data_dir=Path(data_dir),
                    result_dir=base / f"lora_dynamic_{strategy}_{threshold}_q{quota_tag}_max{LORA_MAX_RANK}",
                    data_factor=data_factor,
                    sh_degree=LORA_SH_DEGREE,
                    lora_mode="dynamic",
                    lora_rank_strategy=strategy,
                    lora_rank_threshold=threshold,
                ))

    return runs


def run_training(run: Run, dry_run: bool = False) -> bool:
    cmd = [
        sys.executable, "simple_trainer.py", "default",
        "--data_dir",    str(run.data_dir),
        "--result_dir",  str(run.result_dir),
        "--data_factor", str(run.data_factor),
        "--sh_degree",   str(run.sh_degree),
        "--max_steps",   str(MAX_STEPS),
        "--eval_steps",  str(MAX_STEPS),
        "--save_steps",  str(MAX_STEPS),
        "--disable_viewer",
    ]

    if run.lora_mode == "static":
        cmd += [
            "--disable_dynamic_rank",
            "--lora_rank", str(run.lora_rank),
        ]
    elif run.lora_mode == "dynamic":
        cmd += [
            "--no-disable_dynamic_rank",
            "--lora_max_rank",       str(run.lora_max_rank),
            "--lora_min_rank",       str(run.lora_min_rank),
            "--lora_rank_interval",  str(run.lora_rank_interval),
            "--lora_quota",          str(run.lora_quota[0]),
                                     str(run.lora_quota[1]),
                                     str(run.lora_quota[2]),
            "--lora_rank_strategy",  run.lora_rank_strategy,
            "--lora_rank_threshold", run.lora_rank_threshold,
            "--lora_stats_k",        str(run.lora_stats_k),
            "--lora_warmup_cycles",  str(run.lora_warmup_cycles),
            "--lora_kmeans_iters",   str(run.lora_kmeans_iters),
        ]
    # "none": no extra lora flags — trainer defaults apply

    print("\n" + "=" * 60)
    print(f"  {run.label}")
    print(f"  result_dir : {run.result_dir}")
    print("  cmd        :", " ".join(cmd))
    print("=" * 60)

    if dry_run:
        print("  [dry-run] skipping execution")
        return True

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    runs = build_runs()
    print(f"Planned {len(runs)} run(s):")
    for i, r in enumerate(runs, 1):
        print(f"  {i:2d}. {r.label}")

    failures: List[str] = []
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}]")
        ok = run_training(run, dry_run=dry_run)
        if not ok:
            print(f"  WARNING: run failed — {run.result_dir}")
            failures.append(run.label)

    if failures:
        print("\nFailed runs:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll runs completed. Run summarize_results.py to compare metrics.")


if __name__ == "__main__":
    main()
