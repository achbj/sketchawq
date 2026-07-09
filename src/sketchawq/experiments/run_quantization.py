from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from sketchawq.config import load_config
from sketchawq.data.calibration import (
    TextBatchConfig,
    make_calibration_and_holdout_batches,
    make_text_batches,
)
from sketchawq.eval.runner import evaluate_model_variant
from sketchawq.logging_utils import configure_logging
from sketchawq.models.load import choose_device, load_causal_lm
from sketchawq.models.save import save_quantized_artifacts
from sketchawq.quant.pipeline import quantize_model
from sketchawq.reporting.metrics import make_run_metadata
from sketchawq.seed import set_seed

LOGGER = logging.getLogger(__name__)


def run_from_config(
    config_path: str | Path,
    *,
    model_cache_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    configure_logging(config.runtime.log_level)
    set_seed(config.runtime.seed)
    metadata = make_run_metadata(config, config_path=config_path)

    device = choose_device(config.runtime.device)
    LOGGER.info("Using device: %s", device)
    tokenizer, model = load_causal_lm(config.model, cache_dir=model_cache_dir, device=device)
    calibration_batches, holdout_batches = make_calibration_and_holdout_batches(
        tokenizer,
        config.calibration,
        device=device,
    )
    evaluation_batches = None
    if config.evaluation.enabled:
        evaluation_batches = _make_evaluation_batches(
            tokenizer,
            config,
            calibration_batches=calibration_batches,
            device=device,
        )

    evaluations: dict[str, Any] = {}
    model_selection: dict[str, Any] = {}
    if config.model_selection.enabled:
        if not holdout_batches:
            raise RuntimeError("model_selection requires non-empty calibration holdout batches")
        if config.model_selection.compare_original:
            LOGGER.info("Evaluating original model on held-out selection data")
            model_selection["original"] = evaluate_model_variant(
                model,
                holdout_batches,
                config.model_selection,
                variant="original_fp16",
                method="FP16",
            )
    if config.evaluation.enabled and config.evaluation.compare_original:
        LOGGER.info("Evaluating original model before quantization")
        assert evaluation_batches is not None
        evaluations["original"] = evaluate_model_variant(
            model,
            evaluation_batches,
            config.evaluation,
            variant="original_fp16",
            method="FP16",
        )

    quant_summary = quantize_model(
        model,
        calibration_batches,
        config,
        selection_batches=holdout_batches,
    )
    root = Path(output_root) if output_root is not None else Path(config.output.output_dir)
    output_dir = root / config.output.run_name
    report: dict[str, Any] = {
        "metadata": metadata,
        "output_dir": str(output_dir),
        "quantization": quant_summary.to_dict(),
    }

    if config.model_selection.enabled:
        LOGGER.info("Evaluating quantized model on held-out selection data")
        model_selection["quantized"] = evaluate_model_variant(
            model,
            holdout_batches,
            config.model_selection,
            variant="quantized_sketchawq",
            method=quant_summary.method,
        )
        report["model_selection"] = model_selection

    if config.evaluation.enabled:
        LOGGER.info("Evaluating quantized model")
        assert evaluation_batches is not None
        evaluations["quantized"] = evaluate_model_variant(
            model,
            evaluation_batches,
            config.evaluation,
            variant="quantized_sketchawq",
            method=quant_summary.method,
        )
        report["evaluation"] = evaluations

    save_quantized_artifacts(
        model=model,
        tokenizer=tokenizer,
        config=config,
        report=report,
        output_dir=output_dir,
        packed_layers=quant_summary.packed_layers,
    )
    LOGGER.info("Wrote artifacts to %s", output_dir)
    return report


def _make_evaluation_batches(
    tokenizer,
    config,
    *,
    calibration_batches,
    device,
):
    if config.evaluation.use_calibration_data:
        return calibration_batches

    return make_text_batches(
        tokenizer,
        TextBatchConfig.from_evaluation(
            config.evaluation,
            seed=config.calibration.seed,
        ),
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SketchAWQ quantization.")
    parser.add_argument("--config", required=True, help="Path to a SketchAWQ YAML config.")
    parser.add_argument("--model-cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--output-root", default=None, help="Optional root directory for run artifacts.")
    args = parser.parse_args()
    run_from_config(args.config, model_cache_dir=args.model_cache_dir, output_root=args.output_root)


if __name__ == "__main__":
    main()
