from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from sketchawq.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [3, 4])
def test_global_refinement_changes_only_the_refinement_policy(bits: int) -> None:
    control = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "spectrum_curvature"
        / f"qwen3_4b_{bits}bit_nystrom_residual.yaml"
    )
    treatment = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "global_refinement"
        / f"qwen3_4b_{bits}bit_global_refine.yaml"
    )

    assert treatment.quantization.scale_refinement == "global_sketch"
    assert treatment.quantization.scale_refinement_steps == 3
    assert treatment.output.run_name == f"qwen3_4b_{bits}bit_nystrom_global_refine_g128"

    control_data = asdict(control)
    treatment_data = asdict(treatment)
    del control_data["quantization"]["scale_refinement"]
    del treatment_data["quantization"]["scale_refinement"]
    del control_data["output"]["run_name"]
    del treatment_data["output"]["run_name"]
    assert treatment_data == control_data
