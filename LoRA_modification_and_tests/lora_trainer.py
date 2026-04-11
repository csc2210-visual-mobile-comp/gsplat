import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = True

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # OUR CHANGE: LoRA rank for SH correction
    disable_dynamic_rank: bool = False
    warmup_step: int = 4000
    lora_rank: int = 16
    lora_max_rank: int = 32
    lora_min_rank: int = 2
    disable_quota_analysis: bool = False
    lora_quota: Tuple[float, float, float] = (0.2, 0.6, 0.2)
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    lora_rank: int = 16, # OUR CHANGE: Maximum LoRA rank for SH correction
    disable_dynamic_rank: bool = False,  # OUR CHANGE: Whether to use static rank (affects optimizer choice for lora_A)
    lora_min_rank: int = 2,  # OUR CHANGE: Minimum LoRA rank for bucket initialization
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        # OUR CHANGE: LoRA Modifier 1
        # params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
        if disable_dynamic_rank:
            # Static rank mode: lora_A lives in splats with Adam optimizer
            lora_A = torch.randn((N, lora_rank)) * 0.01
            params.append(("lora_A", torch.nn.Parameter(lora_A), shN_lr))

        # OUR CHANGE: Rank Tracker Mask
        # In dynamic mode, current_ranks and lora_grad_accum are always present.
        # In static mode, current_ranks is still useful for heatmap rendering.
        initial_rank = float(lora_rank) if disable_dynamic_rank else float(lora_min_rank)
        current_ranks = torch.full((N, 1), initial_rank)
        params.append(("current_ranks", torch.nn.Parameter(current_ranks, requires_grad=False), 0.0))
        lora_grad_accum = torch.zeros((N, 1))
        params.append(("lora_grad_accum", torch.nn.Parameter(lora_grad_accum, requires_grad=False), 0.0))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    # OUR CHANGE: In dynamic rank mode, lora_A is NOT in splats at all — per-bucket
    # nn.Parameter tensors (self.lora_A_buckets) handle it instead.
    # In static rank mode, lora_A is in splats and gets a normal Adam optimizer.
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, param, lr in params
        if param.requires_grad
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            load_exposure=cfg.load_exposure,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if cfg.post_processing == "ppisp" and isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError(
                f"PPISP post-processing requires MCMCStrategy at the moment."
            )

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            lora_rank=cfg.lora_rank if cfg.disable_dynamic_rank else cfg.lora_max_rank, # OUR CHANGE: Maximum LoRA rank for SH correction
            disable_dynamic_rank=cfg.disable_dynamic_rank,  # OUR CHANGE: Pass through for optimizer selection
            lora_min_rank=cfg.lora_min_rank,  # OUR CHANGE: Pass min rank for bucket initialization
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # OUR CHANGE: LoRA Modifier 2
        shN_bands = (cfg.sh_degree + 1) ** 2 - 1
        rank = cfg.lora_rank if cfg.disable_dynamic_rank else cfg.lora_max_rank
        self.lora_B = torch.nn.Parameter(torch.randn(rank, 3 * shN_bands, device=self.device) * 0.01)
        
        # We must give Matrix B its own optimizer so Adam updates it!
        self.lora_optimizers = [
            torch.optim.Adam(
                [self.lora_B],
                lr=cfg.shN_lr * math.sqrt(cfg.batch_size * world_size),
                eps=1e-15
            )
        ]

        self.gmm_quota_calculated = False

        # OUR CHANGE: Per-bucket nn.Parameter tensors + Adam optimizers for lora_A (dynamic rank only).
        # Each bucket has its own compact [N_r, r] parameter and optimizer, so memory
        # scales with actual rank usage rather than a full [N, max_rank] tensor.
        if not cfg.disable_dynamic_rank:
            self._init_bucket_tensors()

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps
                / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("[Distillation] Gaussian parameters frozen")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            # OUR CHANGE: Apply Dynamic Correction
            # colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
            shN_bands = (self.cfg.sh_degree + 1) ** 2 - 1

            if self.cfg.disable_dynamic_rank:
                lora_A = self.splats["lora_A"]
                shN_computed = torch.matmul(lora_A, self.lora_B)
            else:
                # OUR CHANGE: Use per-bucket nn.Parameter tensors for lora_A
                buckets = [self.cfg.lora_min_rank, 8, self.cfg.lora_max_rank]
                N = self.splats["current_ranks"].shape[0]
                shN_computed = torch.zeros(N, self.lora_B.shape[1], device=self.device)
                for r in buckets:
                    idxs = self.lora_A_bucket_indices[r]
                    if len(idxs) == 0:
                        continue
                    shN_computed[idxs] = self.lora_A_buckets[r] @ self.lora_B[:r, :]

            shN_computed = shN_computed.view(-1, shN_bands, 3) 
            colors = torch.cat([self.splats["sh0"], shN_computed], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0

        if self.cfg.post_processing is not None:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, info

    # OUR CHANGE: Heatmap helper
    @torch.no_grad()
    def render_rank_heatmap(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        min_rank: int,
        max_rank: int,
    ) -> Tensor:
        """Renders a thermal heatmap of the current Gaussian LoRA ranks."""
        # 1. Grab current geometry
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        # 2. Normalize ranks (min_rank = 0.0, max_rank = 1.0)
        ranks = self.splats["current_ranks"].float()
        norm_ranks = (ranks - min_rank) / (max_rank - min_rank + 1e-8)
        
        # 3. Map to RGB (Blue = Low Rank, Red = High Rank)
        heatmap_rgb = torch.zeros((ranks.shape[0], 3), device=self.device)
        heatmap_rgb[:, 0] = norm_ranks.squeeze()        # Red channel
        heatmap_rgb[:, 2] = 1.0 - norm_ranks.squeeze()  # Blue channel
        
        # 4. Convert RGB directly to base Spherical Harmonics (SH0)
        SH_C0 = 0.28209479177387814
        heatmap_sh0 = ((heatmap_rgb - 0.5) / SH_C0).unsqueeze(1) # [N, 1, 3]

        # 5. Call the core rasterizer directly
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=heatmap_sh0,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=0, # Force flat colors
            packed=self.cfg.packed,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
        )
        
        return torch.clamp(render_colors, 0.0, 1.0)

    # OUR CHANGE: Per-bucket nn.Parameter + Adam optimizer helpers for memory-efficient dynamic LoRA rank

    def _init_bucket_tensors(self):
        """Initialize per-bucket lora_A nn.Parameters and Adam optimizers.

        Creates compact [N_r, r] parameter tensors (initially all N at min_rank)
        with matching per-bucket Adam optimizers, so parameter + gradient +
        optimizer state all scale with actual rank usage.
        """
        cfg = self.cfg
        buckets = [cfg.lora_min_rank, 8, cfg.lora_max_rank]
        N = len(self.splats["means"])
        device = self.device
        BS = cfg.batch_size * self.world_size
        lr = cfg.shN_lr * math.sqrt(BS)
        betas = (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999))
        eps = 1e-15 / math.sqrt(BS)

        # Initially all Gaussians are at min_rank
        self.lora_A_bucket_indices = {
            cfg.lora_min_rank: torch.arange(N, device=device),
            8: torch.empty(0, dtype=torch.long, device=device),
            cfg.lora_max_rank: torch.empty(0, dtype=torch.long, device=device),
        }
        self.lora_A_buckets = {}
        self.lora_A_bucket_optims = {}
        for r in buckets:
            n_r = len(self.lora_A_bucket_indices[r])
            param = torch.nn.Parameter(torch.randn(n_r, r, device=device) * 0.01)
            self.lora_A_buckets[r] = param
            self.lora_A_bucket_optims[r] = torch.optim.Adam(
                [param], lr=lr, betas=betas, eps=eps
            )
        self.N_prev_for_densification = N

    @torch.no_grad()
    def _scatter_buckets_to_dense(self) -> torch.Tensor:
        """Scatter per-bucket lora_A data into a dense [N, max_rank] tensor.

        Returns a detached float32 zeros tensor with each bucket's .data
        placed in the correct rows and columns.
        """
        cfg = self.cfg
        N = self.splats["current_ranks"].shape[0]
        buckets = [cfg.lora_min_rank, 8, cfg.lora_max_rank]
        dense = torch.zeros(N, cfg.lora_max_rank, device=self.device, dtype=torch.float16)
        for r in buckets:
            idxs = self.lora_A_bucket_indices[r]
            if len(idxs) == 0:
                continue
            dense[idxs, :r] = self.lora_A_buckets[r].data.half()
        return dense

    @torch.no_grad()
    def _rebuild_buckets_from_dense(self, dense: torch.Tensor):
        """Rebuild bucket tensors and fresh Adam optimizers from a dense [N_new, max_rank] tensor.

        Called after densification when N changes. Uses current_ranks to determine
        bucket membership. Optimizer state is reset to zero (acceptable — densification
        already resets optimizer state for other params).

        Args:
            dense: [N_new, max_rank] tensor from splats["lora_A_dense"].data after densification.
        """
        cfg = self.cfg
        buckets = [cfg.lora_min_rank, 8, cfg.lora_max_rank]
        N = dense.shape[0]
        device = self.device
        BS = cfg.batch_size * self.world_size
        lr = cfg.shN_lr * math.sqrt(BS)
        betas = (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999))
        eps = 1e-15 / math.sqrt(BS)

        ranks = self.splats["current_ranks"].data.squeeze().long()  # [N_new]

        new_bucket_indices = {}
        new_buckets = {}
        new_optims = {}
        for r in buckets:
            idxs = (ranks == r).nonzero(as_tuple=True)[0]
            new_bucket_indices[r] = idxs
            data = dense[idxs, :r].float().clone() if len(idxs) > 0 else torch.zeros(0, r, device=device)
            param = torch.nn.Parameter(data)
            new_buckets[r] = param
            new_optims[r] = torch.optim.Adam([param], lr=lr, betas=betas, eps=eps)

        self.lora_A_bucket_indices = new_bucket_indices
        self.lora_A_buckets = new_buckets
        self.lora_A_bucket_optims = new_optims
        self.N_prev_for_densification = N

    @torch.no_grad()
    def _update_buckets_for_rank_change(self, new_r_per_gaussian: torch.Tensor):
        """Rebuild bucket tensors and Adam optimizers after a rank-adjustment step.

        For each Gaussian, carries over param data and Adam exp_avg/exp_avg_sq for
        columns that remain active (min(old_r, new_r)). For upranked Gaussians,
        new columns are initialized with fresh noise. Builds new nn.Parameters and
        Adam optimizers with manually set state so carried momentum is preserved
        while bias correction restarts cleanly.

        Args:
            new_r_per_gaussian: [N] int tensor of new rank values per Gaussian.
        """
        cfg = self.cfg
        buckets = [cfg.lora_min_rank, 8, cfg.lora_max_rank]
        device = self.device
        N = len(self.splats["means"])
        BS = cfg.batch_size * self.world_size
        lr = cfg.shN_lr * math.sqrt(BS)
        betas = (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999))
        eps = 1e-15 / math.sqrt(BS)

        # Build reverse lookup: gaussian index -> row in old bucket tensor
        gauss_to_old_row = torch.empty(N, dtype=torch.long, device=device)
        old_r_per_gauss = torch.zeros(N, dtype=torch.long, device=device)
        for r in buckets:
            old_idxs = self.lora_A_bucket_indices[r]
            if len(old_idxs) == 0:
                continue
            gauss_to_old_row[old_idxs] = torch.arange(len(old_idxs), device=device)
            old_r_per_gauss[old_idxs] = r

        new_bucket_indices = {}
        new_buckets = {}
        new_optims = {}
        current_step = torch.tensor(0.0, device=device)
        for temp_optim in self.lora_A_bucket_optims.values():
            if len(temp_optim.state) > 0:
                state_dict = next(iter(temp_optim.state.values()))
                if "step" in state_dict:
                    current_step = state_dict["step"].clone()
                    break
        for new_r in buckets:
            new_idxs = (new_r_per_gaussian == new_r).nonzero(as_tuple=True)[0]
            new_bucket_indices[new_r] = new_idxs
            n = len(new_idxs)

            new_data = torch.zeros(n, new_r, device=device)
            new_exp_avg = torch.zeros(n, new_r, device=device)
            new_exp_avg_sq = torch.zeros(n, new_r, device=device)

            if n > 0:
                gauss_old_rs = old_r_per_gauss[new_idxs]
                gauss_old_rows = gauss_to_old_row[new_idxs]

                for old_r in buckets:
                    from_old_r = (gauss_old_rs == old_r)
                    if not from_old_r.any():
                        continue

                    pos = from_old_r.nonzero(as_tuple=True)[0]  # positions in new bucket
                    rows = gauss_old_rows[pos]                   # rows in old bucket tensor
                    keep = min(old_r, new_r)

                    # Carry over parameter data for retained columns
                    new_data[pos, :keep] = self.lora_A_buckets[old_r].data[rows, :keep]

                    # Carry over Adam state for retained columns
                    old_optim = self.lora_A_bucket_optims[old_r]
                    old_param = self.lora_A_buckets[old_r]
                    if old_param in old_optim.state and "exp_avg" in old_optim.state[old_param]:
                        state = old_optim.state[old_param]
                        new_exp_avg[pos, :keep] = state["exp_avg"][rows, :keep]
                        new_exp_avg_sq[pos, :keep] = state["exp_avg_sq"][rows, :keep]

            param = torch.nn.Parameter(new_data)
            optim = torch.optim.Adam([param], lr=lr, betas=betas, eps=eps)
            # Manually set optimizer state so carried-over momentum is preserved
            # while bias correction restarts (step=0)
            optim.state[param] = {
                "step": current_step,
                "exp_avg": new_exp_avg,
                "exp_avg_sq": new_exp_avg_sq,
            }
            new_buckets[new_r] = param
            new_optims[new_r] = optim

        for r in [cfg.lora_min_rank, 8, cfg.lora_max_rank]:
            self.lora_A_bucket_optims[r].state.clear()
            self.lora_A_bucket_optims[r].param_groups.clear()

        self.lora_A_bucket_indices = new_bucket_indices
        self.lora_A_buckets = new_buckets
        self.lora_A_bucket_optims = new_optims
    
    def _gmm_thresholds(self, score: torch.Tensor, n_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """1-D Gaussian Mixture Model with 3 components fitted via EM."""
        s = score.float().squeeze()  # [N]
        N = s.shape[0]

        # Initialise means at quartiles, shared variance, uniform weights
        means   = torch.quantile(s, torch.tensor([0.25, 0.5, 0.75], device=s.device))
        vars_   = torch.full((3,), s.var().clamp(min=1e-6).item(), device=s.device)
        weights = torch.full((3,), 1.0 / 3.0, device=s.device)

        for _ in range(n_iter):
            # E-step: log-responsibilities [N, 3]
            diff     = s.unsqueeze(1) - means.unsqueeze(0)           # [N, 3]
            log_resp = (
                -0.5 * diff ** 2 / vars_.unsqueeze(0)
                - 0.5 * vars_.log().unsqueeze(0)
                + weights.log().unsqueeze(0)
            )
            log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
            resp     = log_resp.exp()                                  # [N, 3]

            # M-step
            Nk      = resp.sum(dim=0).clamp(min=1e-6)                 # [3]
            means   = (resp * s.unsqueeze(1)).sum(dim=0) / Nk
            diff    = s.unsqueeze(1) - means.unsqueeze(0)
            vars_   = ((resp * diff ** 2).sum(dim=0) / Nk).clamp(min=1e-6)
            weights = Nk / N

        sorted_means, _ = means.sort()
        bottom_thresh = (sorted_means[0] + sorted_means[1]) / 2.0
        top_thresh    = (sorted_means[1] + sorted_means[2]) / 2.0
        return top_thresh, bottom_thresh
    
    @torch.no_grad()
    def _compute_gaussian_color_variance(self) -> Optional[torch.Tensor]:
        """Project each Gaussian center into all training images and compute color variance."""
        import cv2 as _cv2

        parser = self.parser
        means_np = self.splats["means"].detach().cpu().numpy()  
        N = means_np.shape[0]

        obs_sum  = np.zeros((N, 3), dtype=np.float64)   
        obs_sq   = np.zeros((N, 3), dtype=np.float64)   
        obs_cnt  = np.zeros(N, dtype=np.int32)

        train_indices = np.arange(len(parser.image_names))
        train_indices = train_indices[train_indices % parser.test_every != 0]

        for parser_idx in train_indices:
            camera_id   = parser.camera_ids[parser_idx]
            K           = parser.Ks_dict[camera_id]           
            c2w         = parser.camtoworlds[parser_idx]       
            w2c         = np.linalg.inv(c2w)
            R, t        = w2c[:3, :3], w2c[:3, 3]

            pts_cam = (R @ means_np.T + t[:, None]).T  
            in_front = pts_cam[:, 2] > 0
            proj = (K @ pts_cam.T).T                   
            uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8)  
            uv = np.nan_to_num(uv, nan=-1000.0, posinf=-1000.0, neginf=-1000.0)
            uv = np.clip(uv, -10000.0, 10000.0)

            image = imageio.imread(parser.image_paths[parser_idx])[..., :3].astype(np.float32)
            if len(parser.params_dict.get(camera_id, [])) > 0 and camera_id in parser.mapx_dict:
                image = _cv2.remap(image, parser.mapx_dict[camera_id],
                                   parser.mapy_dict[camera_id], _cv2.INTER_LINEAR)
                x0, y0, w, h = parser.roi_undist_dict[camera_id]
                image = image[y0:y0 + h, x0:x0 + w]

            H, W = image.shape[:2]
            xi = np.round(uv[:, 0]).astype(np.int32)
            yi = np.round(uv[:, 1]).astype(np.int32)
            in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            valid = in_front & in_bounds

            xi_v, yi_v = xi[valid], yi[valid]
            rgb = image[yi_v, xi_v]  

            idx = np.where(valid)[0]
            obs_sum[idx]  += rgb
            obs_sq[idx]   += rgb ** 2
            obs_cnt[idx]  += 1

        enough = obs_cnt >= 3
        variance = np.zeros(N, dtype=np.float32)
        cnt = obs_cnt[enough].reshape(-1, 1).astype(np.float64)
        mean_sq  = obs_sq[enough]  / cnt
        sq_mean  = (obs_sum[enough] / cnt) ** 2
        variance[enough] = (mean_sq - sq_mean).mean(axis=1).astype(np.float32)

        return torch.from_numpy(variance).unsqueeze(-1).to(self.device)
    
    @torch.no_grad()
    def _compute_quota_from_color_variance(self) -> Tuple[float, float, float]:
        """Calculates variance, finds GMM clusters, and returns the (high, mid, low) quota."""

        variances = self._compute_gaussian_color_variance()
        
        if variances is None:
            print("[Warning] Not enough images for color variance. Falling back to default quota.")
            return self.cfg.lora_quota

        # Use GMM instead of K-Means to find the natural boundaries
        top_thresh, bottom_thresh = self._gmm_thresholds(variances, n_iter=20)

        var_sq = variances.squeeze()
        high_mask = var_sq >= top_thresh
        low_mask = var_sq <= bottom_thresh
        mid_mask = ~(high_mask | low_mask)

        N = var_sq.shape[0]
        high_frac = float(high_mask.sum().item() / N)
        mid_frac = float(mid_mask.sum().item() / N)
        low_frac = float(low_mask.sum().item() / N)

        print(f"[Warmup] GMM Quota Assigned: High={high_frac:.2f}, Mid={mid_frac:.2f}, Low={low_frac:.2f}\n")
        return (high_frac, mid_frac, low_frac)
    
    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Freeze Gaussians when PPISP controller distillation starts
            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            exposure = (
                data["exposure"].to(device) if "exposure" in data else None
            )  # [B,]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            # OUR CHANGE: Quantized Dynamic LoRA Rank Adjustment — accumulate per-bucket gradients
            if not cfg.disable_dynamic_rank and step >= cfg.warmup_step:
                with torch.no_grad():
                    for r in [cfg.lora_min_rank, 8, cfg.lora_max_rank]:
                        bucket_param = self.lora_A_buckets[r]
                        if bucket_param.grad is not None:
                            frustration = bucket_param.grad.abs().sum(dim=-1, keepdim=True)
                            self.splats["lora_grad_accum"][self.lora_A_bucket_indices[r]] += frustration

            if not cfg.disable_dynamic_rank:
                allocation_interval = len(self.trainset)
                if hasattr(self.cfg.strategy, "refine_stop_iter"):
                    freeze_step = self.cfg.strategy.refine_stop_iter
                else:
                    freeze_step = 15000
                if step > cfg.warmup_step and step % allocation_interval == 0 and step <= freeze_step:
                    if not self.gmm_quota_calculated:
                        if not cfg.disable_quota_analysis:
                            new_quota = self._compute_quota_from_color_variance()
                            self.cfg.lora_quota = new_quota
                        self.gmm_quota_calculated = True

                    with torch.no_grad():
                        grad_frustration = self.splats["lora_grad_accum"] / (self.splats["current_ranks"].data + 1e-8)

                        N_gs = grad_frustration.shape[0]
                        k_high = int(N_gs * cfg.lora_quota[0])
                        k_low = int(N_gs * cfg.lora_quota[2])

                        _, top_idx = torch.topk(grad_frustration.squeeze(), k_high)
                        _, bottom_idx = torch.topk(-grad_frustration.squeeze(), k_low)

                        top_mask = torch.zeros(N_gs, dtype=torch.bool, device=self.device)
                        bottom_mask = torch.zeros(N_gs, dtype=torch.bool, device=self.device)
                        
                        top_mask[top_idx] = True
                        bottom_mask[bottom_idx] = True
                        bottom_mask &= ~top_mask 
                        middle_mask = ~(top_mask | bottom_mask)

                        # Compute new rank per Gaussian BEFORE updating current_ranks,
                        # so _update_buckets_for_rank_change can compare old vs new.
                        new_r_per_gaussian = self.splats["current_ranks"].data.squeeze().long().clone()
                        new_r_per_gaussian[top_mask.squeeze()] = cfg.lora_max_rank
                        new_r_per_gaussian[middle_mask.squeeze()] = 8
                        new_r_per_gaussian[bottom_mask.squeeze()] = cfg.lora_min_rank

                        r_data = self.splats["current_ranks"].data
                        r_data[top_mask] = float(cfg.lora_max_rank)
                        r_data[middle_mask] = 8.0
                        r_data[bottom_mask] = float(cfg.lora_min_rank)

                        # OUR CHANGE: Rebuild per-bucket nn.Parameters with carried-over param
                        # data and Adam state. Dead columns (rank shrunk) are discarded;
                        # new columns (upranked) get fresh noise — all handled in the method.
                        self._update_buckets_for_rank_change(new_r_per_gaussian)

                        self.splats["lora_grad_accum"].mul_(0.5)

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                # OUR CHANGE: Log LoRA-specific metrics to TensorBoard
                # Check if any bucket has a gradient (use min_rank bucket as proxy)
                _any_lora_grad = any(
                    self.lora_A_buckets[r].grad is not None
                    for r in [cfg.lora_min_rank, 8, cfg.lora_max_rank]
                ) if not cfg.disable_dynamic_rank else self.splats["lora_A"].grad is not None
                if _any_lora_grad:
                    # compute grad_frustration using lora_grad_accum (already accumulated above)
                    grad_frustration = self.splats["lora_grad_accum"].squeeze() / (self.splats["current_ranks"].data.squeeze() + 1e-8)
                    avg_grad = grad_frustration.mean().item()
                    avg_rank = self.splats["current_ranks"].float().mean().item()
                    self.writer.add_scalar("lora/avg_grad_frustration", avg_grad, step)
                    self.writer.add_scalar("lora/avg_rank", avg_rank, step)

                # OUR CHANGE: Detailed LoRA memory breakdown logging for debugging.
                #
                # Why this matters: max_memory_allocated() reports the historical PEAK
                # (inflated by densification temporaries), masking the steady-state
                # savings. These scalars show CURRENT allocation and break down exactly
                # where lora_A memory goes.
                if not cfg.disable_dynamic_rank:
                    N_gs = len(self.splats["means"])
                    # Current (not peak) GPU allocation
                    mem_current_gb = torch.cuda.memory_allocated() / 1024**3
                    self.writer.add_scalar("mem/current_gb", mem_current_gb, step)
                    self.writer.add_scalar("mem/peak_gb", mem, step)

                    # Per-bucket parameter memory (compact [N_r, r] tensors)
                    bucket_param_mb = sum(
                        self.lora_A_buckets[r].numel() * 4 / 1024**2
                        for r in [cfg.lora_min_rank, 8, cfg.lora_max_rank]
                    )
                    self.writer.add_scalar("lora_mem/bucket_param_mb", bucket_param_mb, step)

                    # Hypothetical full param memory if we used [N, max_rank]
                    hypothetical_full_param_mb = N_gs * cfg.lora_max_rank * 4 / 1024**2
                    self.writer.add_scalar("lora_mem/hypothetical_full_param_mb", hypothetical_full_param_mb, step)
                    self.writer.add_scalar("lora_mem/param_saved_mb", hypothetical_full_param_mb - bucket_param_mb, step)

                    # Compact optimizer state from actual bucket Adam states
                    compact_state_mb = 0.0
                    for r in [cfg.lora_min_rank, 8, cfg.lora_max_rank]:
                        param = self.lora_A_buckets[r]
                        if param in self.lora_A_bucket_optims[r].state:
                            s = self.lora_A_bucket_optims[r].state[param]
                            for key in ["exp_avg", "exp_avg_sq"]:
                                if key in s:
                                    compact_state_mb += s[key].numel() * 4 / 1024**2
                    self.writer.add_scalar("lora_mem/compact_optimizer_state_mb", compact_state_mb, step)

                    # Hypothetical Adam state if we used full [N, max_rank] tensors
                    hypothetical_adam_mb = 2 * N_gs * cfg.lora_max_rank * 4 / 1024**2
                    self.writer.add_scalar("lora_mem/hypothetical_adam_mb", hypothetical_adam_mb, step)
                    self.writer.add_scalar(
                        "lora_mem/optimizer_state_saved_mb",
                        hypothetical_adam_mb - compact_state_mb,
                        step,
                    )

                    # Per-bucket Gaussian counts — tells you the rank distribution
                    buckets = [cfg.lora_min_rank, 8, cfg.lora_max_rank]
                    for r in buckets:
                        n_r = len(self.lora_A_bucket_indices[r])
                        self.writer.add_scalar(f"lora_rank/n_at_rank_{r}", n_r, step)
                        self.writer.add_scalar(
                            f"lora_rank/frac_at_rank_{r}", n_r / max(N_gs, 1), step
                        )

                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }

                # OUR CHANGE: Include parameter count in stats for monitoring model
                total_params = sum(p.numel() for p in self.splats.values())
                if hasattr(self, "lora_B"):
                    total_params += self.lora_B.numel()
                stats["total_params"] = total_params

                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if hasattr(self, "lora_B"):
                    data["lora_B"] = self.lora_B.data
                if not self.cfg.disable_dynamic_rank and hasattr(self, "lora_A_buckets"):
                    data["lora_A_buckets"] = {r: self.lora_A_buckets[r].data for r in [self.cfg.lora_min_rank, 8, self.cfg.lora_max_rank]}
                    data["lora_A_bucket_indices"] = {r: self.lora_A_bucket_indices[r] for r in [self.cfg.lora_min_rank, 8, self.cfg.lora_max_rank]}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                if self.post_processing_module is not None:
                    data["post_processing"] = self.post_processing_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    # OUR CHANGE: Use the LoRA-corrected SH coefficients for export
                    # shN = self.splats["shN"]
                    shN_bands = (self.cfg.sh_degree + 1) ** 2 - 1
                    if self.cfg.disable_dynamic_rank:
                        lora_A = self.splats["lora_A"]
                        shN_computed = torch.matmul(lora_A, self.lora_B)
                    else:
                        N_ply = len(self.splats["means"])
                        shN_computed = torch.zeros(N_ply, self.lora_B.shape[1], device=self.device)
                        for r in [self.cfg.lora_min_rank, 8, self.cfg.lora_max_rank]:
                            idxs = self.lora_A_bucket_indices[r]
                            if len(idxs) > 0:
                                shN_computed[idxs] = self.lora_A_buckets[r].data @ self.lora_B.data[:r, :]

                    shN = shN_computed.view(-1, shN_bands, 3)

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # OUR CHANGE: GRADIENT CLIPPING
            if not cfg.disable_dynamic_rank:
                lora_params = [self.lora_A_buckets[r] for r in [cfg.lora_min_rank, 8, cfg.lora_max_rank]
                               if self.lora_A_buckets[r].grad is not None]
                if lora_params and self.lora_B.grad is not None:
                    torch.nn.utils.clip_grad_norm_(lora_params + [self.lora_B], max_norm=1.0)
            else:
                if self.splats["lora_A"].grad is not None and self.lora_B.grad is not None:
                    torch.nn.utils.clip_grad_norm_([self.splats["lora_A"], self.lora_B], max_norm=1.0)

            # optimize
            for name, optimizer in self.optimizers.items():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # OUR CHANGE: Step the LoRA optimizers (lora_B)
            for optimizer in self.lora_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # OUR CHANGE: Step per-bucket lora_A optimizers
            if not cfg.disable_dynamic_rank:
                for r in [cfg.lora_min_rank, 8, cfg.lora_max_rank]:
                    self.lora_A_bucket_optims[r].step()
                    self.lora_A_bucket_optims[r].zero_grad(set_to_none=True)
            
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # OUR CHANGE: Scatter bucket tensors into a temporary dense splat so the
            # densification strategy can resize lora_A values along with other params.
            if not cfg.disable_dynamic_rank:
                _lora_A_dense_for_densification = self._scatter_buckets_to_dense()
                self.splats["lora_A_dense"] = torch.nn.Parameter(
                    _lora_A_dense_for_densification, requires_grad=False
                )

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # OUR CHANGE: After densification, N may have changed. Rebuild bucket tensors
            # from the dense splat that was sized alongside other params by the strategy.
            if not cfg.disable_dynamic_rank:
                N_current = len(self.splats["means"])
                if N_current != self.N_prev_for_densification:
                    self._rebuild_buckets_from_dense(self.splats["lora_A_dense"].data)
                del self.splats["lora_A_dense"]
                _lora_A_dense_for_densification = None
                self.N_prev_for_densification = N_current

            # eval the full set
            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=None,  # For novel views, pass None (no per-frame parameters available)
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)

            canvas_list = [pixels, colors]

            # OUR CHANGE: Generate the Heatmap
            heatmap_colors = self.render_rank_heatmap(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                min_rank=cfg.lora_min_rank,
                max_rank=cfg.lora_max_rank,
            )

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                # OUR CHANGE: Save the LoRA rank heatmap for visualization
                heatmap_canvas = heatmap_colors.squeeze(0).cpu().numpy()
                heatmap_canvas = (heatmap_canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_heatmap.png",
                    heatmap_canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                # Compute color-corrected metrics for fair comparison across methods
                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_color_correction_metric:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports...")

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names from COLMAP camera IDs
        # camera_id_to_idx maps COLMAP ID -> 0-based index
        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)

        if hasattr(runner, "lora_B") and "lora_B" in ckpts[0]:
            runner.lora_B.data = ckpts[0]["lora_B"].to(runner.device)
            
        if not cfg.disable_dynamic_rank and "lora_A_buckets" in ckpts[0]:
            runner.lora_A_bucket_indices = ckpts[0]["lora_A_bucket_indices"]
            for r in [cfg.lora_min_rank, 8, cfg.lora_max_rank]:
                if r in ckpts[0]["lora_A_buckets"]:
                    runner.lora_A_buckets[r].data = ckpts[0]["lora_A_buckets"][r].to(runner.device)

        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
