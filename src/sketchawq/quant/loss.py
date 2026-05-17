from __future__ import annotations

import torch


def weighted_error_score(
    original: torch.Tensor,
    candidate: torch.Tensor,
    *,
    diagonal: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    low_rank_weight: float = 0.15,
    normalize_terms: bool = True,
) -> float:
    return float(
        weighted_error_score_tensor(
            original,
            candidate,
            diagonal=diagonal,
            basis=basis,
            low_rank_weight=low_rank_weight,
            normalize_terms=normalize_terms,
        )
        .detach()
        .cpu()
    )


def weighted_error_score_tensor(
    original: torch.Tensor,
    candidate: torch.Tensor,
    *,
    diagonal: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    low_rank_weight: float = 0.15,
    normalize_terms: bool = True,
) -> torch.Tensor:
    """Approximate expected layer-output error from weight reconstruction error.

    The diagonal term estimates sum_j H_jj * ||error[:, j]||^2.
    The low-rank term estimates a small part of error @ H @ error.T using
    the sketched Hessian basis.
    """
    original_float = original.float().detach()
    error = (original_float - candidate.float()).detach()
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

    return score.detach()


def output_error_score(
    original: torch.Tensor,
    candidate: torch.Tensor,
    activation_rows: torch.Tensor,
    *,
    normalize_terms: bool = True,
) -> float:
    return float(
        output_error_score_tensor(
            original,
            candidate,
            activation_rows,
            normalize_terms=normalize_terms,
        )
        .detach()
        .cpu()
    )


def output_error_score_tensor(
    original: torch.Tensor,
    candidate: torch.Tensor,
    activation_rows: torch.Tensor,
    *,
    normalize_terms: bool = True,
) -> torch.Tensor:
    """Measure direct layer-output reconstruction error on sampled activations."""
    rows = activation_rows.detach()
    if rows.numel() == 0:
        return weighted_error_score_tensor(
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
    return score.detach()
