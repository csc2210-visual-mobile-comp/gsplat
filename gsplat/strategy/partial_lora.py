from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from .base import Strategy
from gsplat.utils import normalized_quat_to_rotmat


# Each entry is (internal_parameter_name, trailing_shape).
# Example for SH colors + opacity:
#   [("sh0", (1, 3)), ("shN", (15, 3)), ("opacities", (1,))]
LoraLayout = List[Tuple[str, Tuple[int, ...]]]


@torch.no_grad()
def _update_param_with_optimizer(
    param_fn,
    optimizer_fn,
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    names: Optional[List[str]] = None,
):
    """Update params in-place and keep optimizer state aligned.

    This version is tolerant of frozen params that do not have an optimizer.
    """
    if names is None:
        names = list(params.keys())

    for name in names:
        param = params[name]
        new_param = param_fn(name, param)
        params[name] = new_param

        if name not in optimizers:
            # Allowed for frozen params.
            if param.requires_grad:
                raise AssertionError(
                    f"Optimizer for trainable parameter '{name}' is missing."
                )
            continue

        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state.get(param, {})
            if param in optimizer.state:
                del optimizer.state[param]

            for key in list(param_state.keys()):
                if key != "step":
                    param_state[key] = optimizer_fn(key, param_state[key])

            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state


@torch.no_grad()
def duplicate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Duplicate selected GSs exactly."""
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            torch.cat([p, p[sel]], dim=0), requires_grad=p.requires_grad
        )

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat(
            [v, torch.zeros((len(sel), *v.shape[1:]), device=device, dtype=v.dtype)],
            dim=0,
        )

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat([v, v[sel]], dim=0)


@torch.no_grad()
def remove(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Remove selected GSs."""
    sel = torch.where(~mask)[0]

    def param_fn(name: str, p: Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[sel]

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[sel]


@torch.no_grad()
def reset_opa(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    value: float,
):
    """Reset base opacities to the given post-sigmoid value."""
    max_logit = torch.logit(torch.tensor(value, device=params["opacities"].device)).item()

    def param_fn(name: str, p: Tensor) -> torch.nn.Parameter:
        if name != "opacities":
            raise ValueError(f"Unexpected parameter name: {name}")
        opacities = torch.clamp(p, max=max_logit)
        return torch.nn.Parameter(opacities, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )


def _pack_lora_layout(
    tensors: Union[Dict[str, Tensor], torch.nn.ParameterDict],
    lora_layout: LoraLayout,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Pack the LoRA-targeted tensors into a flat [N, D] matrix."""
    chunks: List[Tensor] = []
    for name, _shape in lora_layout:
        t = tensors[name]
        if name == "opacities":
            t = t.unsqueeze(-1)
        if indices is not None:
            t = t[indices]
        chunks.append(t.reshape(t.shape[0], -1))

    if len(chunks) == 0:
        n = 0 if indices is None else len(indices)
        device = next(iter(tensors.values())).device
        return torch.empty((n, 0), device=device)

    return torch.cat(chunks, dim=-1)


@torch.no_grad()
def _split_lora_target_eff_flat(
    eff_params: Union[Dict[str, Tensor], torch.nn.ParameterDict],
    mask: Tensor,
    lora_layout: LoraLayout,
    revised_opacity: bool = False,
) -> Tensor:
    """Apply DefaultStrategy-like split, but only on the LoRA target subspace.

    Returns:
        [2 * n_sel, D_lora]
    """
    device = mask.device
    sel = torch.where(mask)[0]
    n_sel = len(sel)
    if n_sel == 0:
        width = sum(int(torch.tensor(shape).prod().item()) for _, shape in lora_layout)
        return torch.empty((0, width), device=device)

    # Geometry-derived split samples, same as DefaultStrategy.split().
    scales_lin = torch.exp(eff_params["scales"][sel])  # [N, 3]
    quats_norm = F.normalize(eff_params["quats"][sel], dim=-1)  # [N, 4]
    rotmats = normalized_quat_to_rotmat(quats_norm)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales_lin,
        torch.randn(2, n_sel, 3, device=device),
    )  # [2, N, 3]

    chunks: List[Tensor] = []
    for name, _shape in lora_layout:
        if name == "means":
            chunk = (eff_params["means"][sel] + samples).reshape(2 * n_sel, 3)
        elif name == "scales":
            chunk = torch.log(scales_lin / 1.6).repeat(2, 1)  # [2N, 3]
        elif name == "quats":
            # Default split keeps quaternion values copied; we use effective quats.
            chunk = eff_params["quats"][sel].repeat(2, 1)  # [2N, 4]
        elif name == "opacities":
            opa = eff_params["opacities"][sel]
            if revised_opacity:
                new_opa = 1.0 - torch.sqrt(1.0 - torch.sigmoid(opa))
                chunk = torch.logit(new_opa).repeat(2).unsqueeze(-1)  # [2N, 1]
            else:
                chunk = opa.repeat(2).unsqueeze(-1)  # [2N, 1]
        else:
            # Generic "copy through" for color/features/sh0/shN/etc.
            p = eff_params[name][sel]
            repeats = [2] + [1] * (p.dim() - 1)
            chunk = p.repeat(repeats).reshape(2 * n_sel, -1)

        chunks.append(chunk.reshape(2 * n_sel, -1))

    return torch.cat(chunks, dim=-1)


@torch.no_grad()
def split_approx_ab(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    eff_params: Union[Dict[str, Tensor], torch.nn.ParameterDict],
    B: Tensor,
    lora_layout: LoraLayout,
    revised_opacity: bool = False,
    reg_lambda: float = 1e-4,
):
    """Approximate DefaultStrategy split in LoRA space only.

    Base params are duplicated, and new A rows are solved so that:

        A_new @ B ~= split(eff_params)[lora_targets] - duplicated_base[lora_targets]
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    n_split = len(sel)
    if n_split == 0:
        return

    # Target effective values after split, restricted to LoRA-targeted dimensions.
    split_eff_flat = _split_lora_target_eff_flat(
        eff_params=eff_params,
        mask=mask,
        lora_layout=lora_layout,
        revised_opacity=revised_opacity,
    )  # [2 * n_split, D]

    # Duplicated base values in the same subspace.
    base_flat = _pack_lora_layout(params, lora_layout, indices=sel)  # [n_split, D]
    base_children_flat = base_flat.repeat(2, 1)  # [2 * n_split, D]

    D = B.shape[1]
    if split_eff_flat.shape[1] != D:
        raise ValueError(
            f"LoRA layout width ({split_eff_flat.shape[1]}) does not match B width ({D})."
        )

    # Solve A_new in least squares sense:
    #   A_new B ~= delta
    # where B is [r, D], delta is [2N, D], A_new is [2N, r]
    Bf = B.detach().to(dtype=torch.float32)  # [r, D]
    delta = (split_eff_flat - base_children_flat).to(dtype=torch.float32)  # [2N, D]

    Bt = Bf.T  # [D, r]
    BBt = Bf @ Bt  # [r, r]
    BBt_reg = BBt + reg_lambda * torch.eye(BBt.shape[0], device=Bf.device)
    inv = torch.linalg.inv(BBt_reg)
    A_new = (delta @ Bt @ inv).to(dtype=params["A"].dtype, device=params["A"].device)

    def param_fn(name: str, p: Tensor) -> torch.nn.Parameter:
        reps = [2] + [1] * (p.dim() - 1)
        if name == "A":
            p_new = torch.cat([p[rest], A_new], dim=0)
        else:
            p_split = p[sel].repeat(reps)
            p_new = torch.cat([p[rest], p_split], dim=0)
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros(
            (2 * n_split, *v.shape[1:]), device=device, dtype=v.dtype
        )
        return torch.cat([v[rest], v_split], dim=0)

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            reps = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(reps)
            state[k] = torch.cat([v[rest], v_new], dim=0)


@dataclass
class LoRATargetStrategyAB(Strategy):
    """LoRA-aware densification/pruning strategy with configurable target subspace.

    This is intended for trainers where each Gaussian has:
      - a per-Gaussian low-rank factor `A` in `params["A"]`
      - a shared matrix `B`
      - a layout describing which parameter blocks are controlled by LoRA

    Expected usage from the trainer:

        eff_params = ...  # effective params after applying A @ B
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        strategy.step_post_backward(
            eff_params=eff_params,
            params=params,
            optimizers=optimizers,
            state=state,
            step=step,
            info=info,
            packed=cfg.packed,
            B=B,
            lora_layout=lora_layout,
        )

    `lora_layout` should use internal tensor names, e.g.:
      - SH colors only: [("sh0", (1, 3)), ("shN", ((sh_degree + 1) ** 2 - 1, 3))]
      - SH colors + opacity: [("sh0", (1, 3)), ("shN", (..., 3)), ("opacities", (1,))]
      - app_opt colors + opacity: [("colors", (3,)), ("opacities", (1,))]
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        # Required parameter keys.
        for key in ["means", "scales", "quats", "opacities", "A"]:
            assert key in params, f"{key} is required in params but missing."

        # Optimizers are only required for trainable params.
        for name, p in params.items():
            if p.requires_grad:
                assert name in optimizers, (
                    f"Optimizer for trainable parameter '{name}' is missing."
                )

        # If optimizer exists, it must own exactly one param group with one param.
        for name, opt in optimizers.items():
            assert len(opt.param_groups) == 1, (
                f"Optimizer for '{name}' must have exactly one param_group."
            )
            assert len(opt.param_groups[0]["params"]) == 1, (
                f"Optimizer for '{name}' must optimize exactly one parameter tensor."
            )

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        assert self.key_for_gradient in info, (
            "The 2D means of the Gaussians is required but missing."
        )
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        eff_params: Union[Dict[str, Tensor], torch.nn.ParameterDict],
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
        B: Optional[Tensor] = None,
        lora_layout: Optional[LoraLayout] = None,
    ):
        if step >= self.refine_stop_iter:
            return

        if B is None:
            raise ValueError("LoRATargetStrategyAB requires B in step_post_backward().")
        if lora_layout is None:
            raise ValueError(
                "LoRATargetStrategyAB requires lora_layout in step_post_backward()."
            )

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            n_dupli, n_split = self._grow_gs(
                eff_params=eff_params,
                params=params,
                optimizers=optimizers,
                state=state,
                step=step,
                B=B,
                lora_layout=lora_layout,
            )
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            n_prune = self._prune_gs(
                eff_params=eff_params,
                params=params,
                optimizers=optimizers,
                state=state,
                step=step,
            )
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()

        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n_gaussian = len(params["means"])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state.get("radii", None) is None:
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        if packed:
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"].max(dim=-1).values  # [nnz]
        else:
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )

        if self.refine_scale2d_stop_iter > 0:
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        eff_params: Union[Dict[str, Tensor], torch.nn.ParameterDict],
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        B: Tensor,
        lora_layout: LoraLayout,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(eff_params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = int(is_dupli.sum().item())

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = int(is_split.sum().item())

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # Prevent freshly duplicated rows from being split immediately.
        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)], dim=0
        )

        if n_split > 0:
            split_approx_ab(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                eff_params=eff_params,
                B=B,
                lora_layout=lora_layout,
                revised_opacity=self.revised_opacity,
            )

        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        eff_params: Union[Dict[str, Tensor], torch.nn.ParameterDict],
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(eff_params["opacities"].flatten()) < self.prune_opa

        if step > self.reset_every:
            is_too_big = (
                torch.exp(eff_params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
            is_prune = is_prune | is_too_big

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune