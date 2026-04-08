"""
Utilities for Gaussian rank assignment analysis.

Provides reusable scoring, clustering, and analysis functions for both training
(simple_trainer.py) and post-hoc analysis (rank_analysis.py).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from torch import Tensor


class RankScoring:
    """Compute per-Gaussian scores for rank assignment."""

    @staticmethod
    def sh_energy_lora(lora_A: Tensor, lora_B: Tensor, current_ranks: Tensor, 
                       rank_buckets: Tuple[int, int, int]) -> Tensor:
        """
        Measure L2 energy of LoRA correction for each Gaussian.
        
        Computes ||shN_computed||² where shN = lora_A @ lora_B
        High energy = complex view-dependent color correction needed
        
        Args:
            lora_A: [N, max_rank]
            lora_B: [max_rank, sh_dim]
            current_ranks: [N, 1]
            rank_buckets: (min_rank, mid_rank, max_rank)
            
        Returns:
            score: [N, 1] per-Gaussian energy
        """
        shN = torch.zeros(
            (lora_A.shape[0], lora_B.shape[1]),
            device=lora_A.device,
            dtype=lora_A.dtype,
        )
        ranks_sq = current_ranks.squeeze()
        for r in rank_buckets:
            mask_r = ranks_sq == r
            if not mask_r.any():
                continue
            shN[mask_r] = lora_A[mask_r, :r] @ lora_B[:r, :]

        # L2 energy per Gaussian
        score = (shN ** 2).sum(dim=1, keepdim=True)
        return score
    
    @staticmethod
    def sh_energy_direct(shN: Tensor) -> Tensor:
        """
        Measure L2 energy of SH coefficients directly.
        
        For non-LoRA runs: compute energy from stored higher-degree SH coefficients.
        High energy = Gaussian expresses complex view-dependent appearance
        
        Args:
            shN: [N, sh_dim] tensor of higher-degree SH coefficients
            
        Returns:
            score: [N, 1] per-Gaussian energy
        """
        score = (shN ** 2).sum(dim=1, keepdim=True)
        return score

    @staticmethod
    def gradient_score(grad_accum: Tensor, current_ranks: Tensor) -> Tensor:
        """
        Score = sum(|grad_lora_A|) / current_rank
        
        Measures optimization pressure per active LoRA parameter.
        A Gaussian whose few active dimensions are being pushed hard
        is "frustrated" and gets promoted.
        """
        score = grad_accum / (current_ranks + 1e-8)
        return score

    @staticmethod
    def opacity_grad_score(grad_accum: Tensor, current_ranks: Tensor, 
                          opacities: Tensor) -> Tensor:
        """
        Score = gradient_score / opacity
        
        Divide by opacity: nearly-transparent Gaussians contribute little
        to the image, so gradient pressure is less meaningful.
        """
        score = grad_accum / (current_ranks + 1e-8)
        opacity = torch.sigmoid(opacities).unsqueeze(-1).detach()
        score = score / (opacity + 1e-4)
        return score

    @staticmethod
    def color_variance_score(variances: np.ndarray) -> Tensor:
        """
        Convert per-point photometric variance to a Gaussian score.
        
        High variance → Gaussian exhibits complex view-dependent appearance
        → likely needs higher LoRA rank.
        
        Args:
            variances: [N] numpy array of color variances per Gaussian
            
        Returns:
            score: [N, 1] tensor
        """
        score = torch.from_numpy(variances).float().unsqueeze(-1)
        return score


class RankClustering:
    """Assign Gaussians to rank buckets based on scores."""

    @staticmethod
    def kmeans_thresholds(score: Tensor, n_iter: int = 10) -> Tuple[Tensor, Tensor]:
        """
        1-D k-means with 3 centroids for scene-adaptive thresholding.
        
        Returns (top_thresh, bottom_thresh) — thresholds that maximize
        separation between low/mid/high rank buckets.
        """
        s = score.float().squeeze()
        centroids = torch.quantile(s, torch.tensor([0.25, 0.5, 0.75], device=s.device))

        for _ in range(n_iter):
            dists = (s.unsqueeze(1) - centroids.unsqueeze(0)).abs()
            assignments = dists.argmin(dim=1)
            for i in range(3):
                mask = assignments == i
                if mask.any():
                    centroids[i] = s[mask].mean()

        sorted_c, _ = centroids.sort()
        bottom_thresh = (sorted_c[0] + sorted_c[1]) / 2.0
        top_thresh = (sorted_c[1] + sorted_c[2]) / 2.0
        return top_thresh, bottom_thresh

    @staticmethod
    def gmm_thresholds(score: Tensor, n_iter: int = 20) -> Tuple[Tensor, Tensor]:
        """
        1-D Gaussian Mixture Model with 3 components via EM.
        
        Variance-aware: tight low-cluster and spread high-tail modeled
        separately, so boundaries adapt to actual distribution shape.
        
        Returns (top_thresh, bottom_thresh) — thresholds via component means.
        """
        s = score.float().squeeze()
        N = s.shape[0]

        means = torch.quantile(s, torch.tensor([0.25, 0.5, 0.75], device=s.device))
        vars_ = torch.full((3,), s.var().clamp(min=1e-6).item(), device=s.device)
        weights = torch.full((3,), 1.0 / 3.0, device=s.device)

        for _ in range(n_iter):
            diff = s.unsqueeze(1) - means.unsqueeze(0)
            log_resp = (
                -0.5 * diff ** 2 / vars_.unsqueeze(0)
                - 0.5 * vars_.log().unsqueeze(0)
                + weights.log().unsqueeze(0)
            )
            log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
            resp = log_resp.exp()

            Nk = resp.sum(dim=0).clamp(min=1e-6)
            means = (resp * s.unsqueeze(1)).sum(dim=0) / Nk
            diff = s.unsqueeze(1) - means.unsqueeze(0)
            vars_ = ((resp * diff ** 2).sum(dim=0) / Nk).clamp(min=1e-6)
            weights = Nk / N

        sorted_means, _ = means.sort()
        bottom_thresh = (sorted_means[0] + sorted_means[1]) / 2.0
        top_thresh = (sorted_means[1] + sorted_means[2]) / 2.0
        return top_thresh, bottom_thresh

    @staticmethod
    def assign_buckets(
        score: Tensor,
        rank_buckets: Tuple[int, int, int],
        method: str = "kmeans",
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Assign Gaussians to rank buckets.
        
        Args:
            score: [N, 1] per-Gaussian scores
            rank_buckets: (low, mid, high) rank values
            method: "kmeans" or "gmm"
            **kwargs: passed to clustering method (e.g., n_iter)
            
        Returns:
            (ranks, top_thresh, bottom_thresh): per-Gaussian rank assignments
        """
        if method == "kmeans":
            top_thresh, bottom_thresh = RankClustering.kmeans_thresholds(
                score, n_iter=kwargs.get("n_iter", 10)
            )
        elif method == "gmm":
            top_thresh, bottom_thresh = RankClustering.gmm_thresholds(
                score, n_iter=kwargs.get("n_iter", 20)
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        top_mask = score.squeeze() >= top_thresh
        bottom_mask = score.squeeze() <= bottom_thresh
        middle_mask = ~(top_mask | bottom_mask)

        ranks = torch.full((score.shape[0],), float(rank_buckets[0]), device=score.device)
        ranks[middle_mask] = float(rank_buckets[1])
        ranks[top_mask] = float(rank_buckets[2])

        return ranks, top_thresh, bottom_thresh


@dataclass
class CheckpointData:
    """Container for scores and assignments at a single checkpoint."""
    
    step: int
    num_gaussians: int
    
    # Scores: dict[scoring_method_name] -> [N,1] tensor
    scores: Dict[str, np.ndarray]
    
    # Bucket assignments: dict[clustering_method_name] -> dict[scoring_method_name] -> [N] tensor
    assignments: Dict[str, Dict[str, np.ndarray]]
    
    # Thresholds: dict[clustering_method_name] -> dict[scoring_method_name] -> (top, bottom)
    thresholds: Dict[str, Dict[str, Tuple[float, float]]]
    
    def save(self, checkpoint_dir: Path):
        """Save checkpoint data to disk (JSON + NPZ)."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        step_file = checkpoint_dir / f"step_{self.step:06d}"
        
        # Save metadata
        metadata = {
            "step": self.step,
            "num_gaussians": self.num_gaussians,
        }
        with open(f"{step_file}_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Save scores and assignments as NPZ
        np_data = {}
        for scoring_method, score in self.scores.items():
            np_data[f"score_{scoring_method}"] = score
        
        for clustering_method, cluster_data in self.assignments.items():
            for scoring_method, assignment in cluster_data.items():
                np_data[f"assign_{clustering_method}_{scoring_method}"] = assignment
        
        np.savez(f"{step_file}.npz", **np_data)
        
        # Save thresholds as JSON
        threshold_dict = {}
        for clustering_method, cluster_thresh in self.thresholds.items():
            threshold_dict[clustering_method] = {}
            for scoring_method, (top, bottom) in cluster_thresh.items():
                threshold_dict[clustering_method][scoring_method] = [
                    float(top), float(bottom)
                ]
        
        with open(f"{step_file}_thresholds.json", "w") as f:
            json.dump(threshold_dict, f, indent=2)
    
    @classmethod
    def load(cls, checkpoint_dir: Path, step: int):
        """Load checkpoint data from disk."""
        checkpoint_dir = Path(checkpoint_dir)
        step_file = checkpoint_dir / f"step_{step:06d}"
        
        # Load metadata
        with open(f"{step_file}_metadata.json") as f:
            metadata = json.load(f)
        
        # Load numpy data
        np_data = np.load(f"{step_file}.npz")
        
        # Reconstruct scores
        scores = {}
        for key in np_data.files:
            if key.startswith("score_"):
                scoring_method = key[6:]  # Remove "score_" prefix
                scores[scoring_method] = np_data[key]
        
        # Reconstruct assignments
        assignments = {}
        for key in np_data.files:
            if key.startswith("assign_"):
                parts = key[7:].split("_", 1)  # Remove "assign_" and split
                if len(parts) == 2:
                    clustering_method, scoring_method = parts
                    if clustering_method not in assignments:
                        assignments[clustering_method] = {}
                    assignments[clustering_method][scoring_method] = np_data[key]
        
        # Load thresholds
        with open(f"{step_file}_thresholds.json") as f:
            threshold_dict = json.load(f)
        
        thresholds = {}
        for clustering_method, cluster_thresh in threshold_dict.items():
            thresholds[clustering_method] = {}
            for scoring_method, (top, bottom) in cluster_thresh.items():
                thresholds[clustering_method][scoring_method] = (top, bottom)
        
        return cls(
            step=metadata["step"],
            num_gaussians=metadata["num_gaussians"],
            scores=scores,
            assignments=assignments,
            thresholds=thresholds,
        )


class RankAnalysis:
    """Post-hoc analysis of rank predictions across checkpoints."""

    @staticmethod
    def compute_bucket_percentages(assignments: np.ndarray, rank_buckets: Tuple[int, int, int]) -> Dict[int, float]:
        """
        Compute percentage of Gaussians in each bucket.
        
        Args:
            assignments: [N] array of rank assignments
            rank_buckets: (low, mid, high)
            
        Returns:
            dict: {rank: percentage_of_gaussians}
        """
        total = len(assignments)
        percentages = {}
        for rank in rank_buckets:
            count = np.sum(assignments == rank)
            percentages[rank] = 100.0 * count / total
        return percentages

    @staticmethod
    def percentile_correlation(
        scores_early: np.ndarray,
        scores_late: np.ndarray,
        min_gaussians: int = 100,
    ) -> Tuple[float, float]:
        """
        Compute Spearman rank correlation of percentiles between early and late checkpoints.
        
        Since Gaussians change between checkpoints, we compute the percentile rank of each
        Gaussian's score, then correlate percentile distributions.
        
        Args:
            scores_early: [N_early] scores at early checkpoint
            scores_late: [N_late] scores at late checkpoint
            min_gaussians: minimum Gaussians to compute correlation
            
        Returns:
            (correlation, p_value)
        """
        if len(scores_early) < min_gaussians or len(scores_late) < min_gaussians:
            return np.nan, 1.0
        
        # Compute percentile ranks (0-100)
        percentiles_early = 100.0 * (np.argsort(np.argsort(scores_early)) / len(scores_early))
        percentiles_late = 100.0 * (np.argsort(np.argsort(scores_late)) / len(scores_late))
        
        rho, pval = spearmanr(percentiles_early, percentiles_late)
        return rho, pval

    @staticmethod
    def plot_score_distributions(
        checkpoints: Dict[int, CheckpointData],
        scoring_method: str,
        rank_buckets: Tuple[int, int, int],
        output_file: Path,
    ):
        """
        Plot score distributions across checkpoints with thresholds.
        
        Creates two subplots: linear and log scale.
        """
        steps = sorted(checkpoints.keys())
        n_steps = len(steps)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        for ax_idx, ax in enumerate(axes):
            scale = "linear" if ax_idx == 0 else "log"
            
            for step in steps:
                checkpoint = checkpoints[step]
                if scoring_method not in checkpoint.scores:
                    continue
                
                scores = checkpoint.scores[scoring_method]
                ax.hist(
                    scores,
                    bins=50,
                    alpha=0.5,
                    label=f"step {step}",
                    density=True,
                )
            
            # Add threshold lines (from final checkpoint)
            final_checkpoint = checkpoints[steps[-1]]
            if "kmeans" in final_checkpoint.thresholds and scoring_method in final_checkpoint.thresholds["kmeans"]:
                top_thresh, bottom_thresh = final_checkpoint.thresholds["kmeans"][scoring_method]
                ax.axvline(top_thresh, color="red", linestyle="--", alpha=0.7, label=f"K-means high thresh")
                ax.axvline(bottom_thresh, color="blue", linestyle="--", alpha=0.7, label=f"K-means low thresh")
            
            if "gmm" in final_checkpoint.thresholds and scoring_method in final_checkpoint.thresholds["gmm"]:
                top_thresh, bottom_thresh = final_checkpoint.thresholds["gmm"][scoring_method]
                ax.axvline(top_thresh, color="orange", linestyle=":", alpha=0.7, label=f"GMM high thresh")
                ax.axvline(bottom_thresh, color="cyan", linestyle=":", alpha=0.7, label=f"GMM low thresh")
            
            ax.set_xlabel(f"{scoring_method} score")
            ax.set_ylabel("Density")
            ax.set_yscale(scale)
            ax.set_title(f"{scoring_method} Distribution ({scale} scale)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Saved score distribution plot: {output_file}")
        plt.close()

    @staticmethod
    def plot_bucket_percentages(
        checkpoints: Dict[int, CheckpointData],
        clustering_method: str,
        scoring_method: str,
        rank_buckets: Tuple[int, int, int],
        output_file: Path,
    ):
        """Plot bucket percentage evolution across checkpoints."""
        steps = sorted(checkpoints.keys())
        percentages_by_rank = {rank: [] for rank in rank_buckets}
        
        for step in steps:
            checkpoint = checkpoints[step]
            if clustering_method not in checkpoint.assignments:
                continue
            if scoring_method not in checkpoint.assignments[clustering_method]:
                continue
            
            assignments = checkpoint.assignments[clustering_method][scoring_method]
            percentages = RankAnalysis.compute_bucket_percentages(assignments, rank_buckets)
            
            for rank in rank_buckets:
                percentages_by_rank[rank].append(percentages.get(rank, 0.0))
        
        if not percentages_by_rank[rank_buckets[0]]:
            print(f"No data for {clustering_method}/{scoring_method}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ["blue", "green", "red"]
        for rank, color in zip(rank_buckets, colors):
            ax.plot(steps, percentages_by_rank[rank], marker="o", color=color, label=f"Rank {int(rank)}")
        
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Percentage of Gaussians (%)")
        ax.set_title(f"Bucket Distribution: {clustering_method} + {scoring_method}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Saved bucket percentage plot: {output_file}")
        plt.close()

    @staticmethod
    def plot_multi_run_comparison(
        runs: Dict[str, Dict[int, "CheckpointData"]],
        gold_run: str,
        gold_step: int,
        rank_buckets: Tuple[int, int, int],
        output_dir: Path,
    ) -> pd.DataFrame:
        """Compare bucket quota fractions across multiple training runs vs a gold standard.

        For each (clustering_method, scoring_method) pair present in the data, saves:
          quota_<cluster>_<score>.png  — fraction-over-time per run, gold standard as dashed line

        Also saves quota_fractions.csv with all values.

        Args:
            runs:       {run_name: {step: CheckpointData}}
            gold_run:   key in `runs` to use as the gold standard
            gold_step:  step within gold_run to draw as horizontal reference
            rank_buckets: (low, mid, high) rank values
            output_dir: where to write PNGs and CSV

        Returns:
            DataFrame with columns: run, step, cluster_method, score_method, bucket, rank, fraction
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Gather all (cluster_method, score_method) combos present across all checkpoints
        all_combos: set = set()
        for run_data in runs.values():
            for ckpt in run_data.values():
                for cm, score_data in ckpt.assignments.items():
                    for sm in score_data:
                        all_combos.add((cm, sm))

        gold_ckpt = runs.get(gold_run, {}).get(gold_step)
        bucket_labels = ["low", "mid", "high"]
        run_names = sorted(runs.keys())
        colors = plt.cm.tab10.colors
        rows = []

        for cluster_method, score_method in sorted(all_combos):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            fig.suptitle(f"{cluster_method} + {score_method} — bucket quota evolution")

            for run_idx, run_name in enumerate(run_names):
                run_data = runs[run_name]
                steps = sorted(run_data.keys())
                color = colors[run_idx % len(colors)]

                for b_idx, (rank, label) in enumerate(zip(rank_buckets, bucket_labels)):
                    fracs = []
                    for step in steps:
                        ckpt = run_data[step]
                        assign = (ckpt.assignments
                                  .get(cluster_method, {})
                                  .get(score_method))
                        if assign is None:
                            fracs.append(np.nan)
                            continue
                        pct = RankAnalysis.compute_bucket_percentages(assign, rank_buckets)
                        frac = pct.get(rank, 0.0) / 100.0
                        fracs.append(frac)
                        rows.append({
                            "run": run_name, "step": step,
                            "cluster_method": cluster_method,
                            "score_method": score_method,
                            "bucket": label, "rank": rank, "fraction": frac,
                        })
                    axes[b_idx].plot(steps, fracs, marker="o", label=run_name, color=color)

            # Gold standard horizontal lines
            if gold_ckpt is not None:
                gold_assign = (gold_ckpt.assignments
                               .get(cluster_method, {})
                               .get(score_method))
                if gold_assign is not None:
                    gold_pct = RankAnalysis.compute_bucket_percentages(gold_assign, rank_buckets)
                    for b_idx, (rank, label) in enumerate(zip(rank_buckets, bucket_labels)):
                        gold_frac = gold_pct.get(rank, 0.0) / 100.0
                        axes[b_idx].axhline(
                            gold_frac, color="black", linestyle="--", linewidth=2,
                            label=f"{gold_run} @ step {gold_step} (gold)",
                        )

            for b_idx, label in enumerate(bucket_labels):
                axes[b_idx].set_title(f"{label} (rank {rank_buckets[b_idx]})")
                axes[b_idx].set_xlabel("Training step")
                axes[b_idx].set_ylabel("Fraction of Gaussians")
                axes[b_idx].set_ylim(0, 1)
                axes[b_idx].legend(fontsize=7)
                axes[b_idx].grid(True, alpha=0.3)

            out_path = output_dir / f"quota_{cluster_method}_{score_method}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved {out_path}")

        df = pd.DataFrame(rows)
        if not df.empty:
            csv_path = output_dir / "quota_fractions.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")

        return df
