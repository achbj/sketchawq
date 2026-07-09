from __future__ import annotations

import torch


def weighted_error_score(
    original: torch.Tensor,
    candidate: torch.Tensor,
    *,
    diagonal: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    factor: torch.Tensor | None = None,
    low_rank_weight: float = 0.15,
    normalize_terms: bool = True,
) -> float:
    """Approximate expected layer-output error from weight reconstruction error.

    The diagonal term estimates sum_j H_jj * ||error[:, j]||^2.
    A spectrum-aware Nystrom factor represents the low-rank covariance directly.
    The legacy orthonormal basis remains available for historical reproductions.
    """
    original_float = original.float().detach()
    error = (original_float - candidate.float()).detach()
    if factor is not None and factor.numel() > 0:
        return _spectrum_aware_error_score(
            original_float,
            error,
            diagonal=diagonal,
            factor=factor,
            normalize_terms=normalize_terms,
        )
    if diagonal is None:
        base_score = error.pow(2).mean()
        if normalize_terms:
            denominator = original_float.pow(2).mean().clamp_min(1e-12)
            score = base_score / denominator
        else:
            score = base_score
    else:
        diag = diagonal.to(device=error.device, dtype=error.dtype).clamp_min(1e-8)
        diag = diag / diag.mean().clamp_min(1e-8)
        base_score = (error.pow(2) * diag.view(1, -1)).mean()
        if normalize_terms:
            denominator = (original_float.pow(2) * diag.view(1, -1)).mean().clamp_min(1e-12)
            score = base_score / denominator
        else:
            score = base_score

    if basis is not None and basis.numel() > 0:
        q = basis.to(device=error.device, dtype=error.dtype)
        low_rank_projection = error @ q
        low_rank_score = low_rank_projection.pow(2).mean()
        if normalize_terms:
            low_rank_denominator = (original_float @ q).pow(2).mean().clamp_min(1e-12)
            low_rank_score = low_rank_score / low_rank_denominator
        score = score + low_rank_weight * low_rank_score

    return float(score.detach().cpu())


def _spectrum_aware_error_score(
    original: torch.Tensor,
    error: torch.Tensor,
    *,
    diagonal: torch.Tensor | None,
    factor: torch.Tensor,
    normalize_terms: bool,
) -> float:
    factor = factor.detach().to(device=error.device, dtype=error.dtype)
    if factor.ndim != 2 or factor.shape[0] != error.shape[1]:
        raise ValueError("factor must have shape [input_channels, rank]")
    if diagonal is not None:
        diagonal = diagonal.detach().to(device=error.device, dtype=error.dtype).clamp_min(0)
        if diagonal.ndim != 1 or diagonal.shape[0] != error.shape[1]:
            raise ValueError("diagonal must have one value per input channel")

    numerator = _quadratic_curvature_energy(error, diagonal=diagonal, factor=factor)
    if normalize_terms:
        denominator = _quadratic_curvature_energy(
            original,
            diagonal=diagonal,
            factor=factor,
        ).clamp_min(1e-12)
        numerator = numerator / denominator
    return float(numerator.detach().cpu())


def _quadratic_curvature_energy(
    values: torch.Tensor,
    *,
    diagonal: torch.Tensor | None,
    factor: torch.Tensor,
) -> torch.Tensor:
    energy = (values @ factor).square().sum()
    if diagonal is not None:
        energy = energy + (values.square() * diagonal.view(1, -1)).sum()
    return energy / max(values.shape[0], 1)


def output_error_score(
    original: torch.Tensor,
    candidate: torch.Tensor,
    activation_rows: torch.Tensor,
    *,
    normalize_terms: bool = True,
) -> float:
    """Measure direct layer-output reconstruction error on sampled activations."""
    rows = activation_rows.detach()
    if rows.numel() == 0:
        return weighted_error_score(
            original,
            candidate,
            diagonal=None,
            basis=None,
            normalize_terms=normalize_terms,
        )
    rows = rows.to(device=original.device, dtype=torch.float32)
    original_float = original.float().detach()
    candidate_float = candidate.float().detach()
    error = original_float - candidate_float
    output_error = error @ rows.transpose(0, 1)
    score = output_error.pow(2).mean()
    if normalize_terms:
        output = original_float @ rows.transpose(0, 1)
        denominator = output.pow(2).mean().clamp_min(1e-12)
        score = score / denominator
    return float(score.detach().cpu())


def row_weighted_error_score(
    original: torch.Tensor,
    candidate: torch.Tensor,
    row_weights: torch.Tensor,
    *,
    normalize_terms: bool = True,
) -> float:
    """Measure reconstruction error with a probability mass over output rows.

    Tied token embeddings are used both as output-projection rows and as input
    lookup rows. Calibration token frequency provides a simple, data-free
    surrogate for the latter use without changing the quantization budget.
    """
    if original.ndim != 2 or candidate.shape != original.shape:
        raise ValueError("original and candidate must be matching two-dimensional tensors")
    weights = row_weights.detach().to(device=original.device, dtype=torch.float32).flatten()
    if weights.shape[0] != original.shape[0]:
        raise ValueError("row_weights must have one value per output row")
    weights = weights.clamp_min(0)
    if not torch.any(weights > 0):
        return 0.0
    weights = weights / weights.sum()
    original_float = original.float().detach()
    error = original_float - candidate.float().detach()
    score = (error.square().mean(dim=1) * weights).sum()
    if normalize_terms:
        denominator = (original_float.square().mean(dim=1) * weights).sum().clamp_min(1e-12)
        score = score / denominator
    return float(score.detach().cpu())
