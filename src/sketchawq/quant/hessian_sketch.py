from __future__ import annotations

from dataclasses import dataclass

import torch

from sketchawq.config import SketchConfig
from sketchawq.quant.observer import LayerStats


@dataclass(frozen=True, slots=True)
class HessianSketch:
    diagonal: torch.Tensor | None
    basis: torch.Tensor | None
    rank: int
    mode: str
    factor: torch.Tensor | None = None
    low_rank_representation: str = "orthonormal"


def build_hessian_sketch(stats: LayerStats, config: SketchConfig) -> HessianSketch:
    mode = config.effective_mode()
    diagonal = None
    diagonal_normalizer = None
    if config.uses_diagonal():
        diagonal_normalizer = stats.mean_square.mean().clamp_min(config.diagonal_floor)
        diagonal = stats.mean_square.clamp_min(config.diagonal_floor) / diagonal_normalizer

    basis = None
    factor = None
    if config.uses_low_rank():
        h_omega = stats.hessian_times_omega
        if h_omega is not None and torch.isfinite(h_omega).all():
            if config.low_rank_representation == "nystrom":
                factor = _nystrom_factor(
                    h_omega,
                    stats.sketch_omega,
                    diagonal=diagonal,
                    diagonal_normalizer=diagonal_normalizer,
                    rank=config.rank,
                    diagonal_floor=config.diagonal_floor,
                    eigenvalue_floor=config.nystrom_eigenvalue_floor,
                )
            else:
                basis = _orthonormal_basis(h_omega, config.rank)

    low_rank = factor if factor is not None else basis

    return HessianSketch(
        diagonal=diagonal,
        basis=basis,
        rank=0 if low_rank is None else low_rank.shape[1],
        mode=mode,
        factor=factor,
        low_rank_representation=config.low_rank_representation,
    )


def _orthonormal_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor | None:
    if matrix.ndim != 2 or matrix.numel() == 0:
        return None
    matrix = matrix.float()
    if matrix.abs().sum() == 0:
        return None
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, : min(rank, q.shape[1])].contiguous()


def _nystrom_factor(
    h_omega: torch.Tensor,
    omega: torch.Tensor | None,
    *,
    diagonal: torch.Tensor | None,
    diagonal_normalizer: torch.Tensor | None,
    rank: int,
    diagonal_floor: float,
    eigenvalue_floor: float,
) -> torch.Tensor | None:
    if omega is None or h_omega.ndim != 2 or omega.ndim != 2:
        return None
    if h_omega.shape != omega.shape or h_omega.numel() == 0:
        return None

    normalizer = (
        torch.as_tensor(diagonal_normalizer, device=h_omega.device, dtype=torch.float32)
        if diagonal_normalizer is not None
        else torch.ones((), device=h_omega.device, dtype=torch.float32)
    ).clamp_min(diagonal_floor)
    normalized_h_omega = h_omega.float() / normalizer
    omega = omega.to(device=normalized_h_omega.device, dtype=torch.float32)
    gram = omega.transpose(0, 1) @ normalized_h_omega
    gram = (gram + gram.transpose(0, 1)) * 0.5
    if not torch.isfinite(gram).all():
        return None

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues.abs().max()
    threshold = max(float(diagonal_floor), float(maximum.item()) * eigenvalue_floor)
    keep_indices = torch.nonzero(eigenvalues > threshold, as_tuple=False).flatten()
    if keep_indices.numel() == 0:
        return None
    keep_indices = keep_indices[-min(int(rank), keep_indices.numel()) :]
    selected_values = eigenvalues.index_select(0, keep_indices)
    selected_vectors = eigenvectors.index_select(1, keep_indices)
    factor = normalized_h_omega @ selected_vectors
    factor = factor * selected_values.rsqrt().unsqueeze(0)
    if not torch.isfinite(factor).all():
        return None

    if diagonal is not None:
        target_diagonal = diagonal.to(device=factor.device, dtype=factor.dtype).clamp_min(
            diagonal_floor
        )
        factor_diagonal = factor.square().sum(dim=1)
        row_scale = torch.ones_like(factor_diagonal)
        active = factor_diagonal > target_diagonal
        row_scale[active] = (target_diagonal[active] / factor_diagonal[active]).sqrt()
        factor = factor * row_scale.unsqueeze(1)
    return factor.contiguous()
