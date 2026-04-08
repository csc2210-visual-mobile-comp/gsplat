#!/usr/bin/env python3
"""
run_rank_study.py — Launch the 3 training runs per scene for the rank quota study.

Run A: sh3 normal (no LoRA)   → snapshots at 1100, 2100, 3100, 6000, 14999
Run B: LoRA static rank 32    → snapshots at 1100, 2100, 3100, 6000
Run C: LoRA static rank 2     → snapshots at 1100, 2100, 3100, 6000

After each scene finishes, per-scene comparison is saved.
After all scenes finish, multi-scene comparison is saved.

Usage:
    python run_rank_study.py                        # all scenes
    python run_rank_study.py --scenes counter room  # subset
    python run_rank_study.py --skip_training        # analysis only
    python run_rank_study.py --scenes counter --only sh3  # redo sh3 for one scene
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Scene name → (data_dir, result_base_dir)
# All scenes use data_factor=4
SCENES = {
    "counter": ("data/360_v2/counter", "results/rank_study/counter"),
    "room":    ("data/360_v2/room",    "results/rank_study/room"),
    "kitchen": ("data/360_v2/kitchen", "results/rank_study/kitchen"),
    "bicycle": ("data/360_v2/bicycle", "results/rank_study/bicycle"),
    "sedan":   ("data/sedan",          "results/rank_study/sedan"),
}

DATA_FACTOR = "4"
MULTI_SCENE_OUT = "results/rank_study/multi_scene"


def run(cmd: list, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).parent)


def run_scene(scene_name: str, data_dir: str, out_base: Path,
              run_sh3: bool, run_lora32: bool, run_lora2: bool, python: str):
    sh3_dir      = out_base / "sh3"
    lora32_dir   = out_base / "lora32"
    lora2_dir    = out_base / "lora2"
    analysis_dir = out_base / "comparison"

    if run_sh3:
        run([
            python, "simple_trainer.py", "default",
            "--data_dir", data_dir,
            "--data_factor", DATA_FACTOR,
            "--result_dir", str(sh3_dir),
            "--max_steps", "15000",
            "--sh_degree", "3",
            "--lora_mode", "none",
            "--rank_analysis_steps", "1100", "2100", "3100", "6000", "14999",
            "--disable_viewer",
            "--eval_steps", "15000",
            "--save_steps", "15000",
        ], label=f"[{scene_name}] sh3 normal (no LoRA)")

    if run_lora32:
        run([
            python, "simple_trainer.py", "default",
            "--data_dir", data_dir,
            "--data_factor", DATA_FACTOR,
            "--result_dir", str(lora32_dir),
            "--max_steps", "6000",
            "--sh_degree", "3",
            "--lora_mode", "static",
            "--lora_rank", "32",
            "--lora_rank_buckets", "2", "8", "32",
            "--rank_analysis_steps", "1100", "2100", "3100", "6000",
            "--disable_viewer",
            "--eval_steps", "6000",
            "--save_steps", "6000",
        ], label=f"[{scene_name}] LoRA static rank 32")

    if run_lora2:
        run([
            python, "simple_trainer.py", "default",
            "--data_dir", data_dir,
            "--data_factor", DATA_FACTOR,
            "--result_dir", str(lora2_dir),
            "--max_steps", "6000",
            "--sh_degree", "3",
            "--lora_mode", "static",
            "--lora_rank", "2",
            "--lora_rank_buckets", "2", "8", "32",
            "--rank_analysis_steps", "1100", "2100", "3100", "6000",
            "--disable_viewer",
            "--eval_steps", "6000",
            "--save_steps", "6000",
        ], label=f"[{scene_name}] LoRA static rank 2")

    run([
        python, "rank_analysis.py", "compare",
        "--runs",
            f"sh3={sh3_dir}",
            f"lora32={lora32_dir}",
            f"lora2={lora2_dir}",
        "--gold_run", "sh3",
        "--gold_step", "14999",
        "--output_dir", str(analysis_dir),
    ], label=f"[{scene_name}] Per-scene comparison")


def main():
    parser = argparse.ArgumentParser(description="Rank quota study — multi-scene runner")
    parser.add_argument(
        "--scenes", nargs="+", choices=list(SCENES), default=list(SCENES),
        help="Scenes to run (default: all)",
    )
    parser.add_argument(
        "--skip_training", action="store_true",
        help="Skip all training, only re-run analysis",
    )
    parser.add_argument(
        "--only", choices=["sh3", "lora32", "lora2"],
        help="Run only this training config (analysis always runs after)",
    )
    args = parser.parse_args()

    python = sys.executable
    run_sh3    = not args.skip_training and args.only in (None, "sh3")
    run_lora32 = not args.skip_training and args.only in (None, "lora32")
    run_lora2  = not args.skip_training and args.only in (None, "lora2")

    for scene_name in args.scenes:
        data_dir, out_base_str = SCENES[scene_name]
        print(f"\n{'#'*60}\n  Scene: {scene_name}\n{'#'*60}")
        run_scene(scene_name, data_dir, Path(out_base_str),
                  run_sh3, run_lora32, run_lora2, python)

    # Multi-scene comparison
    run([
        python, "rank_analysis.py", "multi_scene",
        "--scenes", *args.scenes,
        "--results_base", "results/rank_study",
        "--output_dir", MULTI_SCENE_OUT,
    ], label="Multi-scene comparison")

    print(f"\nAll done. Multi-scene results in {MULTI_SCENE_OUT}")


if __name__ == "__main__":
    main()
