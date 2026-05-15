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


def build_hessian_sketch(stats: LayerStats, config: SketchConfig) -> HessianSketch:
    mode = config.effective_mode()
    diagonal = None
    if config.uses_diagonal():
        diagonal = stats.mean_square.clamp_min(config.diagonal_floor)
        diagonal = diagonal / diagonal.mean().clamp_min(config.diagonal_floor)

    basis = None
    if config.uses_low_rank():
        h_omega = stats.hessian_times_omega
        if h_omega is not None and torch.isfinite(h_omega).all():
            basis = _orthonormal_basis(h_omega, config.rank)

    return HessianSketch(
        diagonal=diagonal,
        basis=basis,
        rank=0 if basis is None else basis.shape[1],
        mode=mode,
    )


def _orthonormal_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor | None:
    if matrix.ndim != 2 or matrix.numel() == 0:
        return None
    matrix = matrix.float()
    if matrix.abs().sum() == 0:
        return None
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, : min(rank, q.shape[1])].contiguous()
