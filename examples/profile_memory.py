#!/usr/bin/env python3
"""
profile_memory.py

Runs a standard sh_degree=3 Gaussian Splatting training for 7000 steps and
records GPU memory, number of Gaussians, and active SH degree at every step.

After training, generates a plot showing:
  - GPU memory (GB) as a function of training step
  - Number of Gaussians on a secondary axis
  - Vertical lines + shaded regions when each SH degree activates
  - Annotated peak

Usage:
    cd examples/
    python profile_memory.py

Outputs (written to result_dir/):
    memory_profile.csv    — raw per-step log
    memory_profile.png    — the plot
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

# Must be set before any PIL import so the main process picks it up.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent))
from simple_trainer import Config, Runner


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class ProfileConfig:
    data_dir: str    = "data/sedan"  # scene to profile (must be in data/)
    data_factor: int = 2
    result_dir: str  = "results/memory_profile/sedan_2"
    # Log one CSV row every this many steps (tb_every is set to match)
    log_every: int   = 10


# ── Worker init — propagate truncation flag to DataLoader workers ──────────────

def _worker_init(worker_id: int) -> None:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Profiling Runner ───────────────────────────────────────────────────────────

class ProfilingRunner(Runner):
    """
    Subclass of Runner that:
      1. Injects a worker_init_fn into the DataLoader so truncated images
         don't crash worker processes.
      2. Hooks into the existing writer.add_scalar calls to write a CSV row
         at every TB log point (no duplication of the training loop).
    """

    def __init__(self, cfg: Config, csv_path: str):
        super().__init__(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
        self._csv_path  = csv_path
        self._csv_file  = open(csv_path, "w", newline="")
        writer = csv.writer(self._csv_file)
        writer.writerow(["step", "num_GS", "sh_degree", "gpu_mem_current_GB", "gpu_mem_peak_GB"])
        self._csv_writer = writer

    def _write_row(self, step: int) -> None:
        num_GS   = len(self.splats["means"])
        sh_deg   = min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)
        curr_mem = torch.cuda.memory_allocated()  / 1024**3
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        self._csv_writer.writerow(
            [step, num_GS, sh_deg, f"{curr_mem:.4f}", f"{peak_mem:.4f}"]
        )
        self._csv_file.flush()

    def train(self) -> None:
        # ── Patch DataLoader to pass worker_init_fn ────────────────────────
        original_DataLoader = torch.utils.data.DataLoader

        def patched_DataLoader(dataset, *args, **kwargs):
            kwargs.setdefault("worker_init_fn", _worker_init)
            return original_DataLoader(dataset, *args, **kwargs)

        torch.utils.data.DataLoader = patched_DataLoader

        # ── Hook writer.add_scalar to log one CSV row per TB step ─────────
        original_add_scalar = self.writer.add_scalar

        def hooked_add_scalar(tag, value, global_step=None, *args, **kwargs):
            original_add_scalar(tag, value, global_step, *args, **kwargs)
            # "train/loss" is the first tag written at each TB log point
            if tag == "train/loss" and global_step is not None:
                self._write_row(int(global_step))

        self.writer.add_scalar = hooked_add_scalar

        try:
            super().train()
        finally:
            torch.utils.data.DataLoader = original_DataLoader
            self.writer.add_scalar      = original_add_scalar
            self._csv_file.close()
            print(f"\nProfile CSV saved: {self._csv_path}")


# ── Plotting ───────────────────────────────────────────────────────────────────

SH_COLORS = {0: "#4c72b0", 1: "#55a868", 2: "#c44e52", 3: "#8172b2"}


def plot_profile(csv_path: str, output_path: str, sh_degree_interval: int) -> None:
    """
    Read the profile CSV and produce a two-panel figure:
      Top    — GPU memory (current + peak) vs step, shaded by SH degree
      Bottom — number of Gaussians vs step
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    steps       = data[:, 0].astype(int)
    num_GS      = data[:, 1].astype(int)
    sh_degrees  = data[:, 2].astype(int)
    mem_current = data[:, 3]
    mem_peak    = data[:, 4]

    # Step at which each SH degree > 0 first appears
    sh_transitions = {}
    for deg in [1, 2, 3]:
        idxs = np.where(sh_degrees >= deg)[0]
        if len(idxs):
            sh_transitions[deg] = steps[idxs[0]]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle("GPU Memory Profile During Training  (SH degree 3, no LoRA)", fontsize=13)

    # ── Top panel: memory ──────────────────────────────────────────────────
    ax1.plot(steps, mem_current, lw=1.2, color="steelblue", label="Current memory (GB)")
    ax1.plot(steps, mem_peak,    lw=1.5, color="crimson",   label="Peak memory (GB)", alpha=0.85)

    # Shade background by active SH degree
    boundaries = [steps[0]] + [sh_transitions[d] for d in sorted(sh_transitions)] + [steps[-1]]
    deg_labels = [0] + sorted(sh_transitions.keys())
    for i, deg in enumerate(deg_labels):
        ax1.axvspan(boundaries[i], boundaries[i + 1], alpha=0.07, color=SH_COLORS[deg])

    # Vertical lines at transitions
    ymin, ymax = mem_peak.min(), mem_peak.max()
    for deg, step in sh_transitions.items():
        ax1.axvline(step, color=SH_COLORS[deg], linestyle="--", lw=1.5,
                    label=f"SH degree → {deg}  (step {step})")
        ax1.text(step + (steps[-1] - steps[0]) * 0.005, ymax * 0.98,
                 f"SH={deg}", color=SH_COLORS[deg], fontsize=8, va="top")

    # Annotate peak
    peak_idx = int(np.argmax(mem_peak))
    ax1.annotate(
        f"Peak {mem_peak[peak_idx]:.2f} GB\n"
        f"step {steps[peak_idx]},  SH={sh_degrees[peak_idx]},  {num_GS[peak_idx]:,} GS",
        xy=(steps[peak_idx], mem_peak[peak_idx]),
        xytext=(steps[peak_idx] - (steps[-1] - steps[0]) * 0.15,
                mem_peak[peak_idx] - (ymax - ymin) * 0.15),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )

    ax1.set_ylabel("GPU Memory (GB)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax1.grid(axis="y", alpha=0.3)

    # ── Bottom panel: num Gaussians ────────────────────────────────────────
    ax2.plot(steps, num_GS / 1e3, lw=1.2, color="darkorange")

    for deg, step in sh_transitions.items():
        idx = int(np.searchsorted(steps, step))
        idx = min(idx, len(num_GS) - 1)
        gs_k = num_GS[idx] / 1e3
        ax2.axvline(step, color=SH_COLORS[deg], linestyle="--", lw=1.5)
        ax2.annotate(
            f"{gs_k:.1f}k",
            xy=(step, gs_k),
            xytext=(step + (steps[-1] - steps[0]) * 0.01, gs_k),
            fontsize=8, color=SH_COLORS[deg],
        )

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Num Gaussians (k)")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pcfg = ProfileConfig()

    result_dir = Path(pcfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = str(result_dir / "memory_profile.csv")
    plot_path = str(result_dir / "memory_profile.png")

    cfg = Config(
        data_dir=pcfg.data_dir,
        data_factor=pcfg.data_factor,
        result_dir=str(result_dir),
        max_steps=7000,
        sh_degree=3,
        sh_degree_interval=1000,
        lora_mode="none",
        disable_viewer=True,
        tb_every=pcfg.log_every,
        save_steps=[7000],
        eval_steps=[7000],
    )

    print(f"Starting profiling run: {pcfg.data_dir}")
    print(f"  max_steps={cfg.max_steps}, sh_degree_interval={cfg.sh_degree_interval}")
    print(f"  logging every {pcfg.log_every} steps → {csv_path}")

    runner = ProfilingRunner(cfg, csv_path)
    runner.train()

    print("\nGenerating plot...")
    plot_profile(csv_path, plot_path, cfg.sh_degree_interval)
    print(f"\nDone. Results in: {result_dir}/")
