"""
Experiment runner for gsplat with LoRA SH correction.

Runs the following suite per scene:
  2.1  No LoRA,       SH=3
  2.2  No LoRA,       SH=1
  2.3  LoRA static,   rank=2,  SH=LORA_SH_DEGREE
  2.4  LoRA static,   rank=16, SH=LORA_SH_DEGREE
  2.5  LoRA dynamic,           SH=LORA_SH_DEGREE

To run:
    python run_experiments.py           # full suite
    python run_experiments.py --dry-run # print commands without executing
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# =============================================================================
# EXPERIMENT CONFIGURATION — edit this section
# =============================================================================

# List of (data_dir, result_base_dir, data_factor) tuples.
# data_factor: 1 = full res, 2 or 4 = downsampled.
SCENES: List[Tuple[str, str, int]] = [
    # ("data/pitcher_scene001", "results/pitcher_scene001", 1),
    ("data/360_v2/bicycle",   "results/360_v2_bicycle",   4),
]

# Training duration
MAX_STEPS: int = 7_000

# SH degree used for all LoRA experiments (2.3, 2.4, 2.5)
LORA_SH_DEGREE: int = 3

# --- Which experiments to run (toggle True/False to include/exclude) ---
RUN_NO_LORA_SH3:       bool = False   # 2.1 baseline, no LoRA, SH=3
RUN_NO_LORA_SH1:       bool = False   # 2.2 baseline, no LoRA, SH=1
RUN_LORA_STATIC_R2:    bool = False   # 2.3 static LoRA rank=2
RUN_LORA_STATIC_R16:   bool = False   # 2.4 static LoRA rank=16
RUN_LORA_DYNAMIC:      bool = True   # 2.5 dynamic LoRA

# Dynamic LoRA parameters
LORA_MAX_RANK: int = 32
LORA_MIN_RANK: int = 2
LORA_RANK_INTERVAL: int = 100
# Quota = (top_fraction, middle_fraction, bottom_fraction) — must sum to 1.0
# Top gets max_rank, middle gets mid-rank, bottom gets min_rank.
LORA_QUOTA: Tuple[float, float, float] = (0.2, 0.6, 0.2)

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


@dataclass
class Run:
    label: str                   # human-readable experiment name
    data_dir: Path
    result_dir: Path
    data_factor: int
    sh_degree: int
    lora_mode: str               # "none" | "static" | "dynamic"
    lora_rank: Optional[int] = None          # static only
    lora_max_rank: int = LORA_MAX_RANK
    lora_min_rank: int = LORA_MIN_RANK
    lora_rank_interval: int = LORA_RANK_INTERVAL
    lora_quota: Tuple[float, float, float] = field(default_factory=lambda: LORA_QUOTA)


def build_runs() -> List[Run]:
    runs: List[Run] = []
    for (data_dir, result_base, data_factor) in SCENES:
        base = Path(result_base)
        scene_name = Path(data_dir).name

        if RUN_NO_LORA_SH3:
            runs.append(Run(
                label=f"{scene_name} | no_lora sh=3",
                data_dir=Path(data_dir),
                result_dir=base / "no_lora_sh3",
                data_factor=data_factor,
                sh_degree=3,
                lora_mode="none",
            ))

        if RUN_NO_LORA_SH1:
            runs.append(Run(
                label=f"{scene_name} | no_lora sh=1",
                data_dir=Path(data_dir),
                result_dir=base / "no_lora_sh1",
                data_factor=data_factor,
                sh_degree=1,
                lora_mode="none",
            ))

        if RUN_LORA_STATIC_R2:
            runs.append(Run(
                label=f"{scene_name} | lora_static rank=2 sh={LORA_SH_DEGREE}",
                data_dir=Path(data_dir),
                result_dir=base / "lora_static_r2",
                data_factor=data_factor,
                sh_degree=LORA_SH_DEGREE,
                lora_mode="static",
                lora_rank=2,
            ))

        if RUN_LORA_STATIC_R16:
            runs.append(Run(
                label=f"{scene_name} | lora_static rank=16 sh={LORA_SH_DEGREE}",
                data_dir=Path(data_dir),
                result_dir=base / "lora_static_r16",
                data_factor=data_factor,
                sh_degree=LORA_SH_DEGREE,
                lora_mode="static",
                lora_rank=16,
            ))

        if RUN_LORA_DYNAMIC:
            runs.append(Run(
                label=f"{scene_name} | lora_dynamic max={LORA_MAX_RANK} sh={LORA_SH_DEGREE}",
                data_dir=Path(data_dir),
                result_dir=base / f"lora_dynamic_max{LORA_MAX_RANK}",
                data_factor=data_factor,
                sh_degree=LORA_SH_DEGREE,
                lora_mode="dynamic",
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
            "--lora_max_rank",      str(run.lora_max_rank),
            "--lora_min_rank",      str(run.lora_min_rank),
            "--lora_rank_interval", str(run.lora_rank_interval),
            "--lora_quota",         str(run.lora_quota[0]),
                                    str(run.lora_quota[1]),
                                    str(run.lora_quota[2]),
        ]
    # "none": no extra lora flags

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


def print_summary(runs: List[Run]) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for run in runs:
        stats_dir = run.result_dir / "stats"
        if not stats_dir.exists():
            print(f"\n[missing] {run.label}")
            continue
        print(f"\n--- {run.label} ---")
        for stats_file in sorted(stats_dir.glob("val*.json")):
            data = json.loads(stats_file.read_text())
            print(f"  {stats_file.name}: {data}")


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    runs = build_runs()
    print(f"Planned {len(runs)} run(s):")
    for i, r in enumerate(runs, 1):
        print(f"  {i}. {r.label}")

    failures: List[str] = []
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}]")
        ok = run_training(run, dry_run=dry_run)
        if not ok:
            print(f"  WARNING: run failed — {run.result_dir}")
            failures.append(run.label)

    print_summary(runs)

    if failures:
        print("\nFailed runs:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
