#!/usr/bin/env python3
"""
sfm_color_variance.py

Analyze photometric consistency of SfM points across training images.

For each 3D point in the COLMAP reconstruction (= Gaussian initialization points):
  1. Project it into every training image where COLMAP says it is visible.
  2. Sample the observed RGB color at that projected pixel location.
  3. Compute the variance of those colors across all observing training views.

This measures how "reliable" each point's color is — high variance means
the point is seen with very different colors from different cameras (e.g.
due to specularity, occlusion errors, or moving objects).

Outputs (all written to output_dir/):
  per_point_variance.csv   — one row per SfM point (position, stats, #obs)
  per_image_variance.csv   — one row per training image (mean/median variance)
  variance_histogram.png   — histogram of per-point variance distribution
  variance_heatmap.png     — reference image with points colored by variance
  summary.txt              — aggregate statistics

To run on a different scene, edit the Config fields at the bottom of this file
or import run() and pass a different Config from another script.

Color units: raw [0, 255] pixel space. Variance values are therefore in
[0, 255²] and std-dev values are in [0, 255].
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageFile
from tqdm import tqdm

# Allow loading images that are slightly truncated (common in some datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Allow running from the repo root or from examples/
sys.path.insert(0, str(Path(__file__).parent))
from datasets.colmap import Parser


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    All parameters for one analysis run.
    Edit or subclass to run on multiple scenes.
    """

    # Path to the scene root directory (must contain sparse/ and images/)
    data_dir: str = "data/360_v2/garden"

    # Downsample factor — must match the value used during GS training
    data_factor: int = 4

    # Every N-th image (sorted by filename) is held out as validation.
    # Must match the test_every used during GS training.
    test_every: int = 8

    # Where to write all output files
    output_dir: str = "results/garden_color_variance"

    # A point must be visible in at least this many training images to be
    # included (< 3 means no variance can be computed).
    min_observations: int = 3

    # Reference image for the heatmap overlay.
    # Set to an image filename (e.g. "frame_00042.jpg") to pin a specific view.
    # None → auto-select the training image with the most analyzed points.
    reference_image_name: Optional[str] = None

    # Size of scatter points in the heatmap overlay (matplotlib s= parameter)
    heatmap_point_size: int = 8

    # DPI for all saved figures
    figure_dpi: int = 150


# ── Step 1: Identify training images ──────────────────────────────────────────

def get_train_indices(num_images: int, test_every: int) -> np.ndarray:
    """
    Return the parser-level indices of training images.
    Mirrors the split logic in datasets/colmap.py Dataset.__init__.
    """
    all_indices = np.arange(num_images)
    return all_indices[all_indices % test_every != 0]


# ── Step 2: Project points and sample colors ───────────────────────────────────

def project_points(
    points_world: np.ndarray,   # (N, 3) float32, world coordinates
    camtoworld: np.ndarray,     # (4, 4) float64
    K: np.ndarray,              # (3, 3) float64, undistorted intrinsics
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D world-space points into 2D image pixel coordinates.

    Returns
    -------
    uv : (N, 2) float32 — pixel coordinates (x=col, y=row)
    in_front : (N,) bool — True when the point has positive camera-space depth
    """
    worldtocam = np.linalg.inv(camtoworld)
    R = worldtocam[:3, :3]
    t = worldtocam[:3, 3]

    pts_cam = (R @ points_world.T + t[:, None]).T   # (N, 3)
    in_front = pts_cam[:, 2] > 0

    proj = (K @ pts_cam.T).T                        # (N, 3)
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8)
    return uv.astype(np.float32), in_front


def load_and_undistort(parser: Parser, parser_idx: int) -> np.ndarray:
    """
    Load the image for parser_idx and apply undistortion if needed.
    Returns an (H, W, 3) uint8 array.
    """
    image = imageio.imread(parser.image_paths[parser_idx])[..., :3]
    camera_id = parser.camera_ids[parser_idx]
    params = parser.params_dict[camera_id]

    if len(params) > 0 and camera_id in parser.mapx_dict:
        mapx = parser.mapx_dict[camera_id]
        mapy = parser.mapy_dict[camera_id]
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        x0, y0, w, h = parser.roi_undist_dict[camera_id]
        image = image[y0: y0 + h, x0: x0 + w]

    return image


def sample_colors_nearest(
    image: np.ndarray,   # (H, W, 3) uint8
    uv: np.ndarray,      # (N, 2) float32, pixel coords (x=col, y=row)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample image colors at the given pixel coordinates (nearest-neighbor).

    Returns
    -------
    colors   : (N, 3) float32 in [0, 255]. Zero for out-of-bounds points.
    in_bounds: (N,) bool
    """
    H, W = image.shape[:2]
    x, y = uv[:, 0], uv[:, 1]
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    colors = np.zeros((len(uv), 3), dtype=np.float32)
    if in_bounds.any():
        xi = x[in_bounds].astype(np.int32).clip(0, W - 1)
        yi = y[in_bounds].astype(np.int32).clip(0, H - 1)
        colors[in_bounds] = image[yi, xi].astype(np.float32)

    return colors, in_bounds


# ── Step 3: Gather observations per point ──────────────────────────────────────

def gather_observations(
    parser: Parser,
    train_indices: np.ndarray,
) -> Dict[int, List[np.ndarray]]:
    """
    For every SfM point, collect its observed RGB color from each training image
    that COLMAP says it is visible in.

    Returns
    -------
    observations : dict mapping global point_idx → list of (3,) float32 arrays,
                   one array per training-image observation.
    """
    observations: Dict[int, List[np.ndarray]] = {}

    # point_indices keys come directly from COLMAP image names, which may differ
    # in case from parser.image_names (e.g. .JPG vs .jpg on some datasets).
    point_indices_lower = {k.lower(): v for k, v in parser.point_indices.items()}

    for parser_idx in tqdm(train_indices, desc="Gathering observations"):
        image_name = parser.image_names[parser_idx]
        visible_pts = point_indices_lower.get(image_name.lower())
        if visible_pts is None:
            continue
        if len(visible_pts) == 0:
            continue

        camera_id = parser.camera_ids[parser_idx]
        K = parser.Ks_dict[camera_id]
        camtoworld = parser.camtoworlds[parser_idx]

        image = load_and_undistort(parser, parser_idx)
        points_world = parser.points[visible_pts]        # (M, 3)

        uv, in_front = project_points(points_world, camtoworld, K)
        colors, in_bounds = sample_colors_nearest(image, uv)

        keep = in_front & in_bounds
        for local_i, global_idx in enumerate(visible_pts):
            if keep[local_i]:
                observations.setdefault(int(global_idx), []).append(colors[local_i])

    return observations


# ── Step 4: Compute statistics ─────────────────────────────────────────────────

def compute_per_point_stats(
    observations: Dict[int, List[np.ndarray]],
    min_observations: int,
    parser: Parser,
) -> pd.DataFrame:
    """
    Compute per-point variance statistics.

    Returns a DataFrame sorted by mean_variance (highest first), containing:
      point_idx, x, y, z           — 3D position
      n_observations                — number of training-image samples
      mean_R/G/B                    — mean observed color per channel
      std_R/G/B                     — std-dev of observed color (units: [0,255])
      var_R/G/B                     — variance of observed color (units: [0,255]²)
      mean_variance                 — average of var_R, var_G, var_B (main metric)
      sfm_R/G/B                     — color stored by COLMAP in the sparse model
    """
    rows = []
    for point_idx, color_list in observations.items():
        if len(color_list) < min_observations:
            continue

        colors = np.stack(color_list)           # (K, 3)
        mean_rgb = colors.mean(axis=0)
        std_rgb  = colors.std(axis=0)
        var_rgb  = colors.var(axis=0)

        xyz     = parser.points[point_idx]
        sfm_rgb = parser.points_rgb[point_idx]

        rows.append({
            "point_idx":     int(point_idx),
            "x":             float(xyz[0]),
            "y":             float(xyz[1]),
            "z":             float(xyz[2]),
            "n_observations": len(color_list),
            "mean_R":        float(mean_rgb[0]),
            "mean_G":        float(mean_rgb[1]),
            "mean_B":        float(mean_rgb[2]),
            "std_R":         float(std_rgb[0]),
            "std_G":         float(std_rgb[1]),
            "std_B":         float(std_rgb[2]),
            "var_R":         float(var_rgb[0]),
            "var_G":         float(var_rgb[1]),
            "var_B":         float(var_rgb[2]),
            "mean_variance": float(var_rgb.mean()),
            "sfm_R":         int(sfm_rgb[0]),
            "sfm_G":         int(sfm_rgb[1]),
            "sfm_B":         int(sfm_rgb[2]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("mean_variance", ascending=False)


def compute_per_image_stats(
    parser: Parser,
    train_indices: np.ndarray,
    per_point_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-training-image statistics by looking at which analyzed points
    are visible in each image and aggregating their variances.

    Returns a DataFrame sorted by mean_variance_of_visible_points (highest first).
    """
    if per_point_df.empty:
        return pd.DataFrame()
    variance_lookup = dict(zip(per_point_df["point_idx"], per_point_df["mean_variance"]))

    point_indices_lower = {k.lower(): v for k, v in parser.point_indices.items()}

    rows = []
    for parser_idx in train_indices:
        image_name = parser.image_names[parser_idx]
        visible_pts = point_indices_lower.get(image_name.lower())
        if visible_pts is None:
            continue

        variances = [variance_lookup[int(p)] for p in visible_pts if int(p) in variance_lookup]

        if len(variances) == 0:
            continue

        variances = np.array(variances)
        rows.append({
            "image_name":           image_name,
            "n_visible_sfm_points": len(visible_pts),
            "n_analyzed_points":    len(variances),
            "mean_variance":        float(variances.mean()),
            "median_variance":      float(np.median(variances)),
            "max_variance":         float(variances.max()),
            "p90_variance":         float(np.percentile(variances, 90)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("mean_variance", ascending=False)


# ── Variance binning ─────────────────────────────────────────────────────────

BIN_COLORS = {"low": "cornflowerblue", "medium": "orange", "high": "crimson"}
BIN_ORDER  = ["low", "medium", "high"]


def assign_kmeans_bins(
    per_point_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """KMeans k=3 on mean_variance. Returns (df_with_bin_col, centers, boundaries)."""
    from sklearn.cluster import KMeans

    values = per_point_df["mean_variance"].values
    log_values = np.log1p(values)

    km = KMeans(n_clusters=3, random_state=0, n_init=10)
    km.fit(log_values.reshape(-1, 1))

    order = np.argsort(km.cluster_centers_.ravel())
    rank = np.empty_like(order)
    rank[order] = np.arange(3)

    per_point_df = per_point_df.copy()
    per_point_df["variance_bin"] = [BIN_ORDER[i] for i in rank[km.labels_]]

    log_centers    = np.sort(km.cluster_centers_.ravel())
    log_boundaries = (log_centers[:-1] + log_centers[1:]) / 2.0
    centers    = np.expm1(log_centers)
    boundaries = np.expm1(log_boundaries)
    return per_point_df, centers, boundaries


def assign_jenks_bins(
    per_point_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Classify per-point mean_variance into 3 natural groups using Fisher-Jenks.

    Fisher-Jenks (natural breaks) minimises within-class variance while
    maximising between-class variance — well suited to skewed, long-tailed
    distributions where KMeans tends to produce unbalanced splits.

    Requires: pip install jenkspy

    Adds a 'variance_bin' column ("low" / "medium" / "high") to per_point_df.
    The two break values can be applied to classify per-image stats with the
    same thresholds.

    Returns
    -------
    per_point_df : copy of input DataFrame with 'variance_bin' column added
    centers      : (3,) mean variance of each bin (low, medium, high)
    boundaries   : (2,) Fisher-Jenks break points
                   boundaries[0] separates low from medium,
                   boundaries[1] separates medium from high
    """
    import jenkspy

    values = per_point_df["mean_variance"].values
    # Run Jenks in log space so the skewed tail doesn't dominate break placement.
    # Add 1 before log to handle any zero-variance points safely.
    log_values = np.log1p(values)
    log_breaks = jenkspy.jenks_breaks(log_values, n_classes=3)
    # Convert break points back to original variance space
    boundaries = np.expm1(np.array([log_breaks[1], log_breaks[2]]))

    per_point_df = per_point_df.copy()
    per_point_df["variance_bin"] = pd.cut(
        per_point_df["mean_variance"],
        bins=[-np.inf, boundaries[0], boundaries[1], np.inf],
        labels=BIN_ORDER,
    )

    centers = np.array([
        per_point_df.loc[per_point_df["variance_bin"] == b, "mean_variance"].mean()
        for b in BIN_ORDER
    ])
    return per_point_df, centers, boundaries


def _classify_by_boundaries(values: pd.Series, boundaries: np.ndarray) -> pd.Series:
    """Apply the Fisher-Jenks boundaries to classify any series of variance values."""
    labels = pd.cut(
        values,
        bins=[-np.inf, boundaries[0], boundaries[1], np.inf],
        labels=BIN_ORDER,
    )
    return labels


# ── Step 5: Outputs ────────────────────────────────────────────────────────────

def save_summary(
    per_point_df: pd.DataFrame,
    per_image_df: pd.DataFrame,
    centers: np.ndarray,
    boundaries: np.ndarray,
    output_path: str,
) -> None:
    """Print and save aggregate statistics to a text file."""
    col = "mean_variance"

    def bin_breakdown(labels: pd.Series, total: int) -> List[str]:
        counts = labels.value_counts()
        return [
            f"  Jenks bin means : low={centers[0]:.1f}  medium={centers[1]:.1f}  high={centers[2]:.1f}",
            f"  Boundaries             : low|medium at {boundaries[0]:.1f},  medium|high at {boundaries[1]:.1f}",
            f"  Low    (< {boundaries[0]:7.1f}) : {counts.get('low',    0):6d}  ({100*counts.get('low',    0)/total:5.1f}%)",
            f"  Medium ({boundaries[0]:7.1f} – {boundaries[1]:.1f}) : {counts.get('medium', 0):6d}  ({100*counts.get('medium', 0)/total:5.1f}%)",
            f"  High   (> {boundaries[1]:7.1f}) : {counts.get('high',   0):6d}  ({100*counts.get('high',   0)/total:5.1f}%)",
        ]

    image_bins = _classify_by_boundaries(per_image_df[col], boundaries)

    lines = [
        "=" * 60,
        "SfM Color Variance Analysis — Summary",
        "=" * 60,
        "",
        f"Points analyzed (>= min_observations): {len(per_point_df)}",
        f"Training images analyzed:              {len(per_image_df)}",
        "",
        "Per-point mean_variance distribution (var in [0,255]² units):",
    ]
    for stat, val in per_point_df[col].describe().items():
        lines.append(f"  {stat:8s}: {val:.2f}")

    lines += ["", "Per-point variance bins  (Fisher-Jenks k=3 on point distribution):"]
    lines += bin_breakdown(per_point_df["variance_bin"], len(per_point_df))

    lines += ["", "Per-image variance bins  (same boundaries applied to each image's mean_variance):"]
    lines += bin_breakdown(image_bins, len(per_image_df))

    lines += [
        "",
        "Top 15 highest-variance points:",
        per_point_df.head(15)[
            ["point_idx", "x", "y", "z", "n_observations", "mean_variance", "variance_bin"]
        ].to_string(index=False),
        "",
        "Top 10 highest-variance training images:",
        per_image_df.head(10)[
            ["image_name", "n_analyzed_points", "mean_variance", "max_variance"]
        ].to_string(index=False),
    ]

    text = "\n".join(lines)
    print(text)
    with open(output_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved: {output_path}")


def save_histogram(
    per_point_df: pd.DataFrame,
    jenks_boundaries: np.ndarray,
    kmeans_boundaries: np.ndarray,
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    Left panel : mean_variance (log x-axis) with both clustering boundaries overlaid.
    Right panel: per-channel variance (log x-axis).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Log-spaced bin edges so each bar covers equal width in log space
    log_bins = np.expm1(np.linspace(0, np.log1p(per_point_df["mean_variance"].max()), 101))

    # ── Left: both sets of boundaries on the same plot ──
    ax = axes[0]
    data = per_point_df["mean_variance"]
    ax.hist(data, bins=log_bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(data.median(), color="black", linestyle=":", linewidth=1.5,
               label=f"Median ({data.median():.1f})")
    ax.axvline(jenks_boundaries[0], color=BIN_COLORS["medium"], linestyle="-", linewidth=2,
               label=f"Jenks  low|med  ({jenks_boundaries[0]:.1f})")
    ax.axvline(jenks_boundaries[1], color=BIN_COLORS["high"],   linestyle="-", linewidth=2,
               label=f"Jenks  med|high ({jenks_boundaries[1]:.1f})")
    ax.axvline(kmeans_boundaries[0], color=BIN_COLORS["medium"], linestyle="--", linewidth=2,
               label=f"KMeans low|med  ({kmeans_boundaries[0]:.1f})")
    ax.axvline(kmeans_boundaries[1], color=BIN_COLORS["high"],   linestyle="--", linewidth=2,
               label=f"KMeans med|high ({kmeans_boundaries[1]:.1f})")
    ax.set_xscale("log")
    ax.set_xlabel("Mean RGB Variance  (log scale, pixel² units)")
    ax.set_ylabel("Number of SfM Points")
    ax.set_title("Per-Point Color Variance  —  Jenks (solid) vs KMeans (dashed)")
    ax.legend(fontsize=8)

    # ── Right: per-channel variance ──
    ax = axes[1]
    channel_colors = {"var_R": "red", "var_G": "green", "var_B": "blue"}
    for col, color in channel_colors.items():
        ch_bins = np.expm1(np.linspace(0, np.log1p(per_point_df[col].max()), 101))
        ax.hist(per_point_df[col], bins=ch_bins, alpha=0.45, color=color,
                label=col, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("Per-Channel Variance  (log scale, pixel² units)")
    ax.set_ylabel("Number of SfM Points")
    ax.set_title("Per-Channel Variance Distribution")
    ax.legend()

    plt.suptitle("SfM Point Color Variance — Histogram", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved: {output_path}")


def _pick_reference_image(
    parser: Parser,
    train_indices: np.ndarray,
    per_point_df: pd.DataFrame,
    requested_name: Optional[str],
) -> int:
    """
    Return the parser index of the reference image to use for the heatmap.
    If requested_name is given, find it; otherwise pick the training image
    with the most analyzed points (best coverage).
    """
    analyzed_set = set(per_point_df["point_idx"].astype(int))

    if requested_name is not None:
        for idx in train_indices:
            if parser.image_names[idx] == requested_name:
                return idx
        raise ValueError(
            f"Reference image '{requested_name}' not found among training images."
        )

    point_indices_lower = {k.lower(): v for k, v in parser.point_indices.items()}

    # Auto-select: training image with the most analyzed points
    best_idx, best_count = train_indices[0], 0
    for idx in train_indices:
        name = parser.image_names[idx]
        pts = point_indices_lower.get(name.lower())
        if pts is None:
            continue
        count = sum(1 for p in pts if int(p) in analyzed_set)
        if count > best_count:
            best_count = count
            best_idx = idx

    print(
        f"Auto-selected reference image: '{parser.image_names[best_idx]}' "
        f"({best_count} analyzed points visible)"
    )
    return best_idx


def save_heatmap_overlay(
    parser: Parser,
    per_point_df: pd.DataFrame,
    train_indices: np.ndarray,
    reference_image_name: Optional[str],
    output_path: str,
    point_size: int = 8,
    dpi: int = 150,
) -> None:
    """
    Overlay SfM points on the reference image using 3 discrete colors
    that correspond to the KMeans variance bins (low / medium / high).
    Points outside the image or not in per_point_df are skipped.
    """
    ref_parser_idx = _pick_reference_image(
        parser, train_indices, per_point_df, reference_image_name
    )

    image_name = parser.image_names[ref_parser_idx]
    camera_id  = parser.camera_ids[ref_parser_idx]
    K          = parser.Ks_dict[camera_id]
    camtoworld = parser.camtoworlds[ref_parser_idx]

    image = load_and_undistort(parser, ref_parser_idx)
    H, W  = image.shape[:2]

    # Build lookup: point_idx -> variance_bin label
    bin_lookup = dict(zip(per_point_df["point_idx"].astype(int),
                          per_point_df["variance_bin"]))

    point_indices_lower = {k.lower(): v for k, v in parser.point_indices.items()}
    visible_pts = point_indices_lower.get(image_name.lower())
    if visible_pts is None:
        print("Warning: reference image has no COLMAP point observations.")
        return
    analyzed_mask = np.array([int(p) in bin_lookup for p in visible_pts])
    analyzed_pts  = visible_pts[analyzed_mask]

    if len(analyzed_pts) == 0:
        print("Warning: no analyzed points are visible in the reference image.")
        return

    points_world = parser.points[analyzed_pts]
    uv, in_front = project_points(points_world, camtoworld, K)

    in_bounds = (
        in_front
        & (uv[:, 0] >= 0) & (uv[:, 0] < W)
        & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    uv_valid   = uv[in_bounds]
    bin_labels = [bin_lookup[int(p)]
                  for p, keep in zip(analyzed_pts, in_bounds) if keep]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    # Draw each bin as a separate scatter so we get a clean legend
    for bin_name in BIN_ORDER:
        mask = [b == bin_name for b in bin_labels]
        if not any(mask):
            continue
        mask = np.array(mask)
        ax.scatter(
            uv_valid[mask, 0], uv_valid[mask, 1],
            c=BIN_COLORS[bin_name],
            s=point_size,
            alpha=0.85,
            linewidths=0,
            label=f"{bin_name}  (n={mask.sum()})",
        )

    ax.legend(loc="upper right", framealpha=0.8, fontsize=10)
    ax.set_title(
        f"SfM Point Color Variance — {image_name}\n"
        f"({len(bin_labels)} points shown, colored by Fisher-Jenks bin)",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Main entry point ──────────────────────────────────────────────────────────

def save_cross_scene_summary(scene_stats: List[dict], output_dir: str) -> None:
    """
    Write a cross-scene comparison table to CSV and a formatted text file.

    Each row is one scene. Columns include per-bin percentages and the
    clustering boundaries for both methods. A footer shows the mean and
    std of each numeric column across all scenes.
    """
    df = pd.DataFrame(scene_stats)
    out = Path(output_dir)

    numeric_cols = df.select_dtypes(include=np.number).columns
    mean_row = df[numeric_cols].mean().to_dict()
    std_row  = df[numeric_cols].std().to_dict()
    mean_row["scene"] = "MEAN"
    std_row["scene"]  = "STD"

    summary_df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    summary_df.to_csv(out / "cross_scene_summary.csv", index=False, float_format="%.2f")
    print(f"Saved: {out / 'cross_scene_summary.csv'}")

    lines = [
        "=" * 80,
        "Cross-Scene Color Variance Summary",
        "=" * 80,
        "",
        summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
        "",
        "Columns:",
        "  low/med/high_pct   — % of SfM points in each Jenks bin",
        "  j_b1, j_b2        — Jenks boundaries (low|med, med|high) in pixel² units",
        "  k_b1, k_b2        — KMeans boundaries (same)",
        "  mean_var           — mean point variance across the scene",
        "  median_var         — median point variance across the scene",
        "",
        "MEAN / STD rows show variation across scenes.",
    ]
    text = "\n".join(lines)
    print(text)
    txt_path = out / "cross_scene_summary.txt"
    with open(txt_path, "w") as f:
        f.write(text + "\n")
    print(f"Saved: {txt_path}")


def run(cfg: Config) -> dict:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load SfM data ──────────────────────────────────────────────────────
    print(f"\nLoading COLMAP data from: {cfg.data_dir}")
    parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=False,           # keep world coordinates as-is
        test_every=cfg.test_every,
        load_exposure=False,
    )
    print(f"  {len(parser.image_names)} total images, {len(parser.points)} SfM points")

    # ── Identify training images ───────────────────────────────────────────
    train_indices = get_train_indices(len(parser.image_names), cfg.test_every)
    print(f"  {len(train_indices)} training images "
          f"(test_every={cfg.test_every}, "
          f"{len(parser.image_names) - len(train_indices)} validation)")

    # ── Gather per-point color observations ────────────────────────────────
    observations = gather_observations(parser, train_indices)
    print(f"\n  {len(observations)} points observed in at least 1 training image")

    # ── Compute statistics ─────────────────────────────────────────────────
    per_point_df = compute_per_point_stats(observations, cfg.min_observations, parser)
    print(f"  {len(per_point_df)} points with >= {cfg.min_observations} observations")

    if per_point_df.empty:
        print(f"  WARNING: no points passed the min_observations={cfg.min_observations} "
              f"threshold for scene '{cfg.data_dir}'. Skipping.")
        return {"scene": Path(cfg.data_dir).name, "n_points": 0, "n_images": 0,
                **{k: float("nan") for k in
                   ["low_pct", "medium_pct", "high_pct",
                    "j_b1", "j_b2", "k_b1", "k_b2", "mean_var", "median_var"]}}

    per_image_df = compute_per_image_stats(parser, train_indices, per_point_df)

    # ── Clustering ────────────────────────────────────────────────────────────
    print("  Clustering variance distribution (k=3)...")
    per_point_df, centers, jenks_boundaries   = assign_jenks_bins(per_point_df)
    _,            _,       kmeans_boundaries  = assign_kmeans_bins(per_point_df)

    # ── Save outputs ───────────────────────────────────────────────────────
    print("\nSaving outputs...")

    per_point_df.to_csv(output_dir / "per_point_variance.csv",
                        index=False, float_format="%.4f")
    print(f"Saved: {output_dir / 'per_point_variance.csv'}")

    per_image_df.to_csv(output_dir / "per_image_variance.csv",
                        index=False, float_format="%.4f")
    print(f"Saved: {output_dir / 'per_image_variance.csv'}")

    save_summary(per_point_df, per_image_df, centers, jenks_boundaries,
                 str(output_dir / "summary.txt"))

    save_histogram(per_point_df, jenks_boundaries, kmeans_boundaries,
                   str(output_dir / "variance_histogram.png"), dpi=cfg.figure_dpi)

    save_heatmap_overlay(
        parser, per_point_df, train_indices,
        reference_image_name=cfg.reference_image_name,
        output_path=str(output_dir / "variance_heatmap.png"),
        point_size=cfg.heatmap_point_size,
        dpi=cfg.figure_dpi,
    )

    print(f"\nDone. All results saved to: {output_dir}/")

    # Per-bin percentages (Jenks)
    bin_counts = per_point_df["variance_bin"].value_counts()
    n = len(per_point_df)

    return {
        "scene":       Path(cfg.data_dir).name,
        "n_points":    n,
        "n_images":    len(per_image_df),
        "low_pct":     100 * bin_counts.get("low",    0) / n,
        "medium_pct":  100 * bin_counts.get("medium", 0) / n,
        "high_pct":    100 * bin_counts.get("high",   0) / n,
        "j_b1":        jenks_boundaries[0],
        "j_b2":        jenks_boundaries[1],
        "k_b1":        kmeans_boundaries[0],
        "k_b2":        kmeans_boundaries[1],
        "mean_var":    per_point_df["mean_variance"].mean(),
        "median_var":  per_point_df["mean_variance"].median(),
    }


# ── Scene configurations ───────────────────────────────────────────────────────

RESULTS_ROOT = "results/color_variance_all"

# 360_v2 outdoor scenes
SCENES_360 = [
    "bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump",
]

# Pitcher scenes — masked background, COLMAP already uses the masks/
SCENES_PITCHER = [
    "pitcher_scene001", "pitcher_scene005", "pitcher_scene007",
]

ALL_SCENES: List[Config] = (
    [
        Config(
            data_dir=f"data/360_v2/{name}",
            data_factor=4,
            test_every=8,
            output_dir=f"{RESULTS_ROOT}/{name}",
        )
        for name in SCENES_360
    ]
    + [
        Config(
            data_dir=f"data/{name}",
            data_factor=1,
            test_every=8,
            output_dir=f"{RESULTS_ROOT}/{name}",
        )
        for name in SCENES_PITCHER
    ]
    + [
        Config(
            data_dir="data/sedan",
            data_factor=4,
            test_every=8,
            output_dir=f"{RESULTS_ROOT}/sedan",
        ),
    ]
)


def run_scenes(scenes_to_run: List[Config]) -> None:
    """Run analysis for each scene and save per-scene outputs."""
    for cfg in scenes_to_run:
        print(f"\n{'='*60}")
        print(f"Scene: {cfg.data_dir}")
        print(f"{'='*60}")
        run(cfg)


def summarize_all() -> None:
    """
    Gather results from all scene subfolders under RESULTS_ROOT and write
    the cross-scene summary files. Reads per_point_variance.csv from each
    subfolder — no need to re-run the analysis.
    """
    results_root = Path(RESULTS_ROOT)
    scene_stats = []

    for scene_dir in sorted(results_root.iterdir()):
        csv_path = scene_dir / "per_point_variance.csv"
        if not csv_path.exists():
            continue

        per_point_df = pd.read_csv(csv_path)
        if per_point_df.empty or "mean_variance" not in per_point_df.columns:
            continue

        img_csv = scene_dir / "per_image_variance.csv"
        n_images = len(pd.read_csv(img_csv)) if img_csv.exists() else 0

        bin_counts = per_point_df["variance_bin"].value_counts() if "variance_bin" in per_point_df.columns else {}
        n = len(per_point_df)

        scene_stats.append({
            "scene":      scene_dir.name,
            "n_points":   n,
            "n_images":   n_images,
            "low_pct":    100 * bin_counts.get("low",    0) / n,
            "medium_pct": 100 * bin_counts.get("medium", 0) / n,
            "high_pct":   100 * bin_counts.get("high",   0) / n,
            "mean_var":   per_point_df["mean_variance"].mean(),
            "median_var": per_point_df["mean_variance"].median(),
            "p75_var":    per_point_df["mean_variance"].quantile(0.75),
            "p95_var":    per_point_df["mean_variance"].quantile(0.95),
        })

    if not scene_stats:
        print(f"No scene results found under {RESULTS_ROOT}/")
        return

    print(f"\n{'='*60}")
    print(f"Cross-scene summary ({len(scene_stats)} scenes)")
    print(f"{'='*60}")
    save_cross_scene_summary(scene_stats, RESULTS_ROOT)


if __name__ == "__main__":
    # run_scenes([cfg for cfg in ALL_SCENES if "sedan" in cfg.data_dir])
    summarize_all()
