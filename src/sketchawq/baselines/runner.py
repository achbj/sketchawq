from __future__ import annotations

import gc
import importlib.util
import json
import logging
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sketchawq.baselines.config import (
    ExternalBaselineConfig,
    baseline_config_to_dict,
    load_baseline_config,
)
from sketchawq.config import CalibrationConfig
from sketchawq.data.calibration import (
    CalibrationBatch,
    _iter_texts,
    _select_packed_chunks,
    make_calibration_batches,
)
from sketchawq.eval.runner import evaluate_model_variant
from sketchawq.logging_utils import configure_logging
from sketchawq.models.load import choose_device, load_causal_lm, parse_dtype
from sketchawq.models.save import save_json
from sketchawq.reporting.metrics import (
    EVALUATION_METRIC_FIELDS,
    RUN_SUMMARY_FIELDS,
    collect_environment_info,
    write_csv,
    write_jsonl,
)
from sketchawq.seed import set_seed


LOGGER = logging.getLogger(__name__)


def run_baseline_from_config(
    config_path: str | Path,
    *,
    model_cache_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_baseline_config(config_path)
    _validate_external_baseline_runtime_support(config)
    configure_logging(config.runtime.log_level)
    set_seed(config.runtime.seed)
    device = choose_device(config.runtime.device)
    LOGGER.info("Using device: %s", device)

    tokenizer = _load_tokenizer(config, cache_dir=model_cache_dir)
    evaluation_batches = _make_evaluation_batches(tokenizer, config, device=device)
    metadata = make_baseline_metadata(config, config_path=config_path)
    evaluations: dict[str, Any] = {}

    if config.evaluation.enabled and config.evaluation.compare_original:
        LOGGER.info("Evaluating original model before external baseline quantization")
        _, original_model = load_causal_lm(config.model, cache_dir=model_cache_dir, device=device)
        evaluations["original"] = evaluate_model_variant(
            original_model,
            evaluation_batches,
            config.evaluation,
            variant="original_fp16",
            method="FP16",
        )
        del original_model
        _release_cuda_memory()

    LOGGER.info("Loading and quantizing external baseline: %s", config.method.display_label())
    quantized_model, quantization = load_external_baseline_model(
        config,
        cache_dir=model_cache_dir,
        device=device,
    )

    if config.evaluation.enabled:
        LOGGER.info("Evaluating external baseline model")
        quantized_device = _infer_model_input_device(quantized_model, preferred=device)
        quantized_batches = _move_evaluation_batches_to_device(evaluation_batches, quantized_device)
        evaluations["quantized"] = evaluate_model_variant(
            quantized_model,
            quantized_batches,
            config.evaluation,
            variant=f"quantized_{config.method.normalized_name()}",
            method=config.method.display_label(),
        )

    root = Path(output_root) if output_root is not None else Path(config.output.output_dir)
    output_dir = root / config.output.run_name
    report: dict[str, Any] = {
        "metadata": metadata,
        "output_dir": str(output_dir),
        "quantization": quantization,
        "evaluation": evaluations,
    }
    write_baseline_artifacts(
        model=quantized_model,
        tokenizer=tokenizer,
        config=config,
        report=report,
        output_dir=output_dir,
    )
    LOGGER.info("Wrote baseline artifacts to %s", output_dir)
    return report


def write_failed_baseline_from_config(
    config_path: str | Path,
    *,
    error: BaseException,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a structured failure artifact so batch baseline jobs can continue."""
    config = load_baseline_config(config_path)
    metadata = make_baseline_metadata(config, config_path=config_path)
    root = Path(output_root) if output_root is not None else Path(config.output.output_dir)
    output_dir = root / config.output.run_name
    report: dict[str, Any] = {
        "metadata": metadata,
        "output_dir": str(output_dir),
        "quantization": {
            "method": config.method.display_label(),
            "backend": config.method.package_name(),
            "error": "".join(traceback.format_exception_only(type(error), error)).strip(),
        },
        "evaluation": {},
        "status": "failed",
    }
    write_baseline_artifacts(
        model=torch.nn.Module(),
        tokenizer=None,
        config=config,
        report=report,
        output_dir=output_dir,
    )
    return report


def make_baseline_metadata(
    config: ExternalBaselineConfig,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    return {
        "run_id": uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "run_name": config.output.run_name,
        "config": baseline_config_to_dict(config),
        "environment": collect_environment_info(),
    }


def load_external_baseline_model(
    config: ExternalBaselineConfig,
    *,
    cache_dir: str | Path | None = None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    method = config.method.normalized_name()
    if method == "hqq":
        return _load_hqq_model(config, cache_dir=cache_dir, device=device)
    if method in {"gptq", "awq"}:
        return _load_gptqmodel_baseline(config, cache_dir=cache_dir, device=device)
    if method == "llmcompressor_awq":
        return _load_llmcompressor_awq_baseline(config, cache_dir=cache_dir, device=device)
    raise ValueError(f"Unsupported baseline method: {config.method.name}")


def _validate_external_baseline_runtime_support(config: ExternalBaselineConfig) -> None:
    if config.method.normalized_name() in {"awq", "llmcompressor_awq"} and (
        config.quantization.bits != 4
    ):
        raise RuntimeError(
            f"{config.method.display_label()} baseline only supports 4-bit quantization; "
            f"got bits={config.quantization.bits}."
        )


def build_baseline_run_summary_row(
    report: dict[str, Any],
    config: ExternalBaselineConfig,
) -> dict[str, Any]:
    quant = report.get("quantization", {})
    original = _variant_evaluation(report.get("evaluation", {}), "original")
    quantized = _variant_evaluation(report.get("evaluation", {}), "quantized")
    original_ppl = original.get("perplexity", {})
    quantized_ppl = quantized.get("perplexity", {})
    original_throughput = original.get("throughput", {})
    quantized_throughput = quantized.get("throughput", {})
    original_perplexity = original_ppl.get("perplexity")
    quantized_perplexity = quantized_ppl.get("perplexity")
    original_tps = original_throughput.get("tokens_per_second")
    quantized_tps = quantized_throughput.get("tokens_per_second")
    metadata = report.get("metadata", {})
    environment = metadata.get("environment", {})
    return {
        "run_id": metadata.get("run_id"),
        "created_at": metadata.get("created_at"),
        "run_name": metadata.get("run_name", config.output.run_name),
        "method": quant.get("method", config.method.display_label()),
        "model_repo_id": config.model.repo_id,
        "model_revision": config.model.revision,
        "bits": config.quantization.bits,
        "group_size": config.quantization.group_size,
        "symmetric": config.quantization.symmetric,
        "scale_clip_search": "",
        "outlier_enabled": "",
        "outlier_strategy": "",
        "outlier_metric": "",
        "outlier_fraction": "",
        "outlier_scale_boost": "",
        "outlier_min_channels": "",
        "outlier_max_channels": "",
        "outlier_protected_bits": "",
        "sketch_enabled": False,
        "sketch_mode": "external_baseline",
        "uses_diagonal": "",
        "uses_low_rank": "",
        "sketch_rank": "",
        "low_rank_weight": "",
        "normalize_error_terms": "",
        "benchmark": config.evaluation.benchmark_name,
        "calibration_source": config.calibration.source,
        "calibration_dataset": config.calibration.dataset_name,
        "calibration_samples": config.calibration.num_samples,
        "calibration_packing": config.calibration.packing,
        "sequence_length": config.calibration.sequence_length,
        "num_layers": quant.get("num_layers"),
        "quantization_seconds": quant.get("elapsed_seconds"),
        "cuda_peak_memory_mb": quant.get("cuda_peak_memory_mb"),
        "fp16_weight_mb": "",
        "estimated_quantized_mb": "",
        "estimated_compression_ratio": "",
        "loss": quantized_ppl.get("loss"),
        "perplexity": quantized_perplexity,
        "tokens_per_second": quantized_tps,
        "original_loss": original_ppl.get("loss"),
        "original_perplexity": original_perplexity,
        "original_tokens_per_second": original_tps,
        "quantized_loss": quantized_ppl.get("loss"),
        "quantized_perplexity": quantized_perplexity,
        "quantized_tokens_per_second": quantized_tps,
        "perplexity_delta": _optional_delta(quantized_perplexity, original_perplexity),
        "perplexity_ratio": _optional_ratio(quantized_perplexity, original_perplexity),
        "throughput_ratio": _optional_ratio(quantized_tps, original_tps),
        "output_dir": report.get("output_dir"),
        "device": config.runtime.device,
        "cuda_available": environment.get("cuda_available"),
        "gpu_name": environment.get("gpu_name"),
    }


def build_baseline_evaluation_rows(
    report: dict[str, Any],
    config: ExternalBaselineConfig,
) -> list[dict[str, Any]]:
    metadata = report.get("metadata", {})
    rows: list[dict[str, Any]] = []
    for key in ("original", "quantized"):
        item = _variant_evaluation(report.get("evaluation", {}), key)
        if not item:
            continue
        perplexity = item.get("perplexity", {})
        throughput = item.get("throughput", {})
        rows.append(
            {
                "run_id": metadata.get("run_id"),
                "created_at": metadata.get("created_at"),
                "run_name": metadata.get("run_name", config.output.run_name),
                "model_repo_id": config.model.repo_id,
                "method": item.get("method"),
                "model_variant": item.get("variant", key),
                "benchmark": item.get("benchmark", config.evaluation.benchmark_name),
                "bits": "" if key == "original" else config.quantization.bits,
                "group_size": "" if key == "original" else config.quantization.group_size,
                "loss": perplexity.get("loss"),
                "perplexity": perplexity.get("perplexity"),
                "tokens_per_second": throughput.get("tokens_per_second"),
                "eval_elapsed_seconds": item.get("elapsed_seconds"),
                "eval_batches": perplexity.get("batches"),
                "tokens": throughput.get("tokens"),
                "cuda_peak_memory_mb": item.get("cuda_peak_memory_mb"),
                "output_dir": report.get("output_dir"),
            }
        )
    return rows


def write_baseline_artifacts(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    config: ExternalBaselineConfig,
    report: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(baseline_config_to_dict(config), output_dir / "baseline_config.json")
    save_json(report, output_dir / "baseline_report.json")
    save_json(report, output_dir / "quantization_report.json")
    summary_row = build_baseline_run_summary_row(report, config)
    evaluation_rows = build_baseline_evaluation_rows(report, config)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary_row, indent=2),
        encoding="utf-8",
    )
    write_csv([summary_row], output_dir / "run_summary.csv", fields=RUN_SUMMARY_FIELDS)
    write_csv(evaluation_rows, output_dir / "evaluation_metrics.csv", fields=EVALUATION_METRIC_FIELDS)
    write_jsonl(evaluation_rows, output_dir / "evaluation_metrics.jsonl")
    if config.output.save_model and report.get("status") != "failed":
        _save_external_model(model, tokenizer, output_dir / "model")


def _load_hqq_model(
    config: ExternalBaselineConfig,
    *,
    cache_dir: str | Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    _require_package("hqq", "Install the `hqq` package or use `modal_baselines.py`.")
    try:
        from hqq.models.hf.base import AutoHQQHFModel
    except ImportError as exc:
        raise RuntimeError(
            "The installed `hqq` package does not expose AutoHQQHFModel. "
            "Install a recent HQQ release or rerun `modal_baselines.py`."
        ) from exc

    dtype = parse_dtype(config.model.dtype)
    model_kwargs = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": config.model.local_files_only,
        "cache_dir": str(cache_dir) if cache_dir else None,
    }

    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model.repo_id, dtype=dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.repo_id,
            torch_dtype=dtype,
            **model_kwargs,
        )
    hqq_config = _make_hqq_quantize_config(config)
    model = AutoHQQHFModel.quantize_model(
        model,
        quant_config=hqq_config,
        compute_dtype=dtype,
        device=_device_argument(device),
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return model, {
        "method": config.method.display_label(),
        "backend": "hqq_native",
        "package": config.method.package_name(),
        "bits": config.quantization.bits,
        "group_size": config.quantization.group_size,
        "symmetric": config.quantization.symmetric,
        "elapsed_seconds": elapsed,
        "cuda_peak_memory_mb": _cuda_peak_memory_mb(),
        "num_layers": _count_linear_layers(model),
    }


def _load_gptqmodel_baseline(
    config: ExternalBaselineConfig,
    *,
    cache_dir: str | Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    _require_package("gptqmodel", "Install the `gptqmodel` package or use `modal_baselines.py`.")
    import gptqmodel as gptqmodel_package
    from gptqmodel import GPTQModel

    tokenizer = _load_tokenizer(config, cache_dir=cache_dir)
    calibration_dataset = _make_gptqmodel_calibration_dataset(tokenizer, config)
    quantize_config = _make_gptqmodel_quantize_config(config, gptqmodel_package, device=device)

    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model_obj = _gptqmodel_load(
        GPTQModel,
        config.model.repo_id,
        quantize_config,
        revision=config.model.revision,
        cache_dir=cache_dir,
        dtype=parse_dtype(config.model.dtype),
        device=device,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    _gptqmodel_quantize(
        model_obj,
        calibration_dataset,
        batch_size=config.method.calibration_batch_size,
        backend=_gptqmodel_quantize_backend(gptqmodel_package, config),
    )
    quantized_reload_dir = Path(tempfile.mkdtemp(prefix="sketchawq_gptqmodel_"))
    _gptqmodel_save(model_obj, quantized_reload_dir)
    reloaded_model_obj = _gptqmodel_reload_quantized(
        GPTQModel,
        gptqmodel_package,
        quantized_reload_dir,
        method=config.method.normalized_name(),
        backend_name=config.method.adapter_kwargs.get("backend"),
        device=device,
    )
    model = _unwrap_torch_model(reloaded_model_obj)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return model, {
        "method": config.method.display_label(),
        "backend": f"gptqmodel_{config.method.normalized_name()}",
        "package": config.method.package_name(),
        "bits": config.quantization.bits,
        "group_size": config.quantization.group_size,
        "symmetric": config.quantization.symmetric,
        "elapsed_seconds": elapsed,
        "cuda_peak_memory_mb": _cuda_peak_memory_mb(),
        "num_layers": _count_linear_layers(model),
    }


def _load_llmcompressor_awq_baseline(
    config: ExternalBaselineConfig,
    *,
    cache_dir: str | Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    _require_package(
        "llmcompressor",
        "Install the `llmcompressor` package or use `modal_baselines.py`.",
    )
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    tokenizer = _load_tokenizer(config, cache_dir=cache_dir)
    calibration_dataset = _make_llmcompressor_calibration_dataset(tokenizer, config)
    recipe = _make_llmcompressor_awq_modifier(config, AWQModifier)
    dtype = _llmcompressor_model_dtype(config)
    model_kwargs = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": config.model.local_files_only,
        "cache_dir": str(cache_dir) if cache_dir else None,
    }

    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model.repo_id, dtype=dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.repo_id,
            torch_dtype=dtype,
            **model_kwargs,
        )
    model = model.to(device)
    oneshot_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "dataset": calibration_dataset,
        "recipe": recipe,
        "batch_size": config.method.calibration_batch_size,
        "num_calibration_samples": config.calibration.num_samples,
        "max_seq_length": config.calibration.sequence_length,
        "shuffle_calibration_samples": False,
        "output_dir": None,
    }
    oneshot_kwargs.update(config.method.adapter_kwargs.get("oneshot_kwargs", {}))
    quantized = oneshot(**oneshot_kwargs)
    model = _unwrap_torch_model(quantized or model)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    modifier_kwargs = _llmcompressor_awq_modifier_kwargs(config)
    return model, {
        "method": config.method.display_label(),
        "backend": "llmcompressor_awq",
        "package": config.method.package_name(),
        "bits": config.quantization.bits,
        "group_size": config.quantization.group_size,
        "symmetric": config.quantization.symmetric,
        "model_dtype": _dtype_label(dtype),
        "scheme": modifier_kwargs.get("scheme"),
        "ignore": modifier_kwargs.get("ignore"),
        "targets": modifier_kwargs.get("targets"),
        "elapsed_seconds": elapsed,
        "cuda_peak_memory_mb": _cuda_peak_memory_mb(),
        "num_layers": _count_linear_layers(model),
    }


def _make_hqq_quantize_config(config: ExternalBaselineConfig) -> dict[str, Any]:
    from hqq.core.quantize import BaseQuantizeConfig

    kwargs = {
        "nbits": config.quantization.bits,
        "group_size": config.quantization.group_size,
        "axis": 1,
    }
    kwargs.update(config.method.adapter_kwargs)
    return BaseQuantizeConfig(**kwargs)


def _make_llmcompressor_awq_modifier(
    config: ExternalBaselineConfig,
    awq_modifier_cls: Any,
) -> Any:
    return awq_modifier_cls(**_llmcompressor_awq_modifier_kwargs(config))


def _llmcompressor_model_dtype(config: ExternalBaselineConfig) -> str | torch.dtype:
    requested = config.method.adapter_kwargs.get(
        "model_dtype",
        config.method.adapter_kwargs.get("dtype"),
    )
    if requested is None:
        return parse_dtype(config.model.dtype)
    if isinstance(requested, torch.dtype):
        return requested
    value = str(requested).strip().lower()
    if value == "auto":
        return "auto"
    return parse_dtype(value)


def _dtype_label(dtype: str | torch.dtype) -> str:
    if isinstance(dtype, str):
        return dtype
    return str(dtype).replace("torch.", "")


def _llmcompressor_awq_modifier_kwargs(config: ExternalBaselineConfig) -> dict[str, Any]:
    kwargs = {
        "ignore": _llmcompressor_awq_ignore(config),
        "scheme": _llmcompressor_awq_scheme(config),
        "targets": _llmcompressor_awq_targets(config),
    }
    kwargs.update(config.method.adapter_kwargs.get("modifier_kwargs", {}))
    return kwargs


def _make_llmcompressor_calibration_dataset(
    tokenizer: Any,
    config: ExternalBaselineConfig,
) -> Any:
    from datasets import Dataset

    batches = make_calibration_batches(tokenizer, config.calibration, device=torch.device("cpu"))
    columns: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": []}
    for batch in batches:
        columns["input_ids"].append(batch.input_ids.squeeze(0).to(dtype=torch.long).tolist())
        columns["attention_mask"].append(
            batch.attention_mask.squeeze(0).to(dtype=torch.long).tolist()
        )
    return Dataset.from_dict(columns)


def _llmcompressor_awq_scheme(config: ExternalBaselineConfig) -> str:
    override = config.method.adapter_kwargs.get("scheme")
    if override:
        return str(override)
    return "W4A16" if config.quantization.symmetric else "W4A16_ASYM"


def _llmcompressor_awq_ignore(config: ExternalBaselineConfig) -> list[str]:
    value = config.method.adapter_kwargs.get("ignore", ["lm_head"])
    return _string_list(value)


def _llmcompressor_awq_targets(config: ExternalBaselineConfig) -> list[str]:
    value = config.method.adapter_kwargs.get("targets", ["Linear"])
    return _string_list(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _make_gptqmodel_calibration_dataset(
    tokenizer: Any,
    config: ExternalBaselineConfig,
) -> list[dict[str, torch.Tensor]]:
    if config.calibration.packing:
        return _make_packed_gptqmodel_calibration_dataset(tokenizer, config)

    dataset: list[dict[str, torch.Tensor]] = []
    for text in _iter_texts(config.calibration):
        encoded = tokenizer(
            text,
            max_length=config.calibration.sequence_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0].to(dtype=torch.long, device="cpu")
        if input_ids.numel() == 0:
            continue
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(encoded["input_ids"])
        dataset.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask[0].to(dtype=torch.long, device="cpu"),
            }
        )
        if len(dataset) >= config.calibration.num_samples:
            break
    if not dataset:
        raise RuntimeError("No calibration samples were loaded for GPTQModel baseline")
    return dataset


def _make_packed_gptqmodel_calibration_dataset(
    tokenizer: Any,
    config: ExternalBaselineConfig,
) -> list[dict[str, torch.Tensor]]:
    chunks = _select_packed_chunks(tokenizer, config.calibration)
    if len(chunks) < config.calibration.num_samples:
        raise RuntimeError(
            "Packed GPTQModel calibration produced "
            f"{len(chunks)} chunks, but num_samples={config.calibration.num_samples}. "
            "Use a longer calibration split or reduce sequence_length."
        )

    dataset: list[dict[str, torch.Tensor]] = []
    for chunk in chunks[: config.calibration.num_samples]:
        input_ids = torch.tensor(chunk, dtype=torch.long, device="cpu")
        dataset.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, device="cpu"),
            }
        )
    return dataset


def _make_gptqmodel_quantize_config(
    config: ExternalBaselineConfig,
    gptqmodel_package: Any,
    *,
    device: torch.device,
) -> Any:
    method = config.method.normalized_name()
    config_name = "GPTQConfig" if method == "gptq" else "AWQConfig"
    config_cls = getattr(gptqmodel_package, config_name, None)
    if config_cls is None:
        config_cls = getattr(gptqmodel_package, "QuantizeConfig", None)
    if config_cls is None:
        raise RuntimeError(
            f"GPTQModel does not expose {config_name} or QuantizeConfig; update gptqmodel."
        )
    kwargs = {
        "bits": config.quantization.bits,
        "group_size": config.quantization.group_size,
        "sym": config.quantization.symmetric,
        "device": _device_argument(device),
    }
    if config_cls.__name__ == "QuantizeConfig":
        kwargs["method"] = method
    kwargs.update(_gptqmodel_quantize_config_kwargs(config.method.adapter_kwargs))
    return config_cls(**kwargs)


def _gptqmodel_quantize_config_kwargs(adapter_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep runtime loader choices out of GPTQModel's quantization config dataclasses."""
    return {key: value for key, value in adapter_kwargs.items() if key != "backend"}


def _gptqmodel_quantize_backend(
    gptqmodel_package: Any,
    config: ExternalBaselineConfig,
) -> Any | None:
    method = config.method.normalized_name()
    requested = config.method.adapter_kwargs.get("backend")
    if method != "awq" and requested is None:
        return None
    return _gptqmodel_backend_argument(gptqmodel_package, method=method, requested=requested)


def _gptqmodel_load(
    gptqmodel_cls: Any,
    repo_id: str,
    quantize_config: Any,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    kwargs = {
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    try:
        return gptqmodel_cls.load(repo_id, quantize_config=quantize_config, **kwargs)
    except TypeError:
        try:
            return gptqmodel_cls.load(repo_id, quantize_config, **kwargs)
        except TypeError:
            return gptqmodel_cls.load(repo_id, quantize_config)


def _gptqmodel_quantize(
    model_obj: Any,
    calibration_dataset: list[dict[str, torch.Tensor]],
    *,
    batch_size: int,
    backend: Any | None,
) -> None:
    kwargs = {"batch_size": batch_size}
    if backend is not None:
        kwargs["backend"] = backend
    try:
        model_obj.quantize(calibration_dataset, **kwargs)
        return
    except TypeError:
        if backend is not None:
            try:
                model_obj.quantize(calibration_dataset=calibration_dataset, **kwargs)
                return
            except TypeError as positional_error:
                raise RuntimeError(
                    "GPTQModel quantize() rejected the requested backend. "
                    "For AWQ, use a recent GPTQModel release that supports the "
                    "AWQ torch backend, or set method.adapter_kwargs.backend explicitly."
                ) from positional_error
        pass
    try:
        model_obj.quantize(calibration_dataset=calibration_dataset, **kwargs)
        return
    except TypeError:
        pass
    model_obj.quantize(calibration_dataset, batch_size=batch_size)


def _gptqmodel_save(model_obj: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model_obj, "save"):
        model_obj.save(str(output_dir))
        return
    if hasattr(model_obj, "save_pretrained"):
        model_obj.save_pretrained(str(output_dir))
        return
    raise RuntimeError("GPTQModel object does not expose save() or save_pretrained().")


def _gptqmodel_reload_quantized(
    gptqmodel_cls: Any,
    gptqmodel_package: Any,
    quantized_model_dir: Path,
    *,
    method: str,
    backend_name: str | None,
    device: torch.device,
) -> Any:
    device_name = _indexed_device_argument(device)
    backend = _gptqmodel_backend_argument(gptqmodel_package, method=method, requested=backend_name)
    try:
        return gptqmodel_cls.load(str(quantized_model_dir), device=device_name, backend=backend)
    except TypeError:
        return gptqmodel_cls.load(str(quantized_model_dir))


def _gptqmodel_backend_argument(
    gptqmodel_package: Any,
    *,
    method: str,
    requested: str | None,
) -> Any:
    backend_enum = getattr(gptqmodel_package, "BACKEND", None)
    if requested is None and method == "awq" and backend_enum is not None:
        for attr_name in ("AWQ_TORCH", "TORCH_AWQ"):
            if hasattr(backend_enum, attr_name):
                return getattr(backend_enum, attr_name)
    backend_name = requested or ("awq_torch" if method == "awq" else "torch")
    normalized = backend_name.replace("-", "_").lower()
    if backend_enum is None:
        return normalized
    try:
        return backend_enum(normalized)
    except (TypeError, ValueError):
        attr_name = normalized.upper()
        return getattr(backend_enum, attr_name, normalized)


def _load_tokenizer(config: ExternalBaselineConfig, *, cache_dir: str | Path | None):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.repo_id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _make_evaluation_batches(tokenizer, config: ExternalBaselineConfig, *, device: torch.device):
    batch_config = (
        config.calibration
        if config.evaluation.use_calibration_data
        else CalibrationConfig(
            source=config.evaluation.source,
            dataset_name=config.evaluation.dataset_name,
            dataset_config=config.evaluation.dataset_config,
            split=config.evaluation.split,
            text_column=config.evaluation.text_column,
            num_samples=config.evaluation.num_samples,
            sequence_length=config.evaluation.sequence_length,
            seed=config.calibration.seed,
            packing=config.evaluation.packing,
        )
    )
    return make_calibration_batches(tokenizer, batch_config, device=device)


def _infer_model_input_device(model: torch.nn.Module, *, preferred: torch.device) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    if isinstance(device, str):
        parsed = torch.device(device)
        if parsed.type != "meta":
            return parsed

    for parameter in model.parameters():
        if not parameter.is_meta:
            return parameter.device
    for buffer in model.buffers():
        if not buffer.is_meta:
            return buffer.device
    return preferred


def _move_evaluation_batches_to_device(
    batches: list[CalibrationBatch],
    device: torch.device,
) -> list[CalibrationBatch]:
    return [
        CalibrationBatch(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
        )
        for batch in batches
    ]


def _require_package(import_name: str, message: str) -> None:
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(message)


def _unwrap_torch_model(model_obj: Any) -> torch.nn.Module:
    if isinstance(model_obj, torch.nn.Module):
        return model_obj
    candidate = getattr(model_obj, "model", None)
    if isinstance(candidate, torch.nn.Module):
        return candidate
    raise RuntimeError(
        "Could not unwrap a torch.nn.Module from the external baseline object. "
        "Check the baseline package version."
    )


def _save_external_model(model: torch.nn.Module, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1_000_000


def _count_linear_layers(model: torch.nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, torch.nn.Linear))


def _device_argument(device: torch.device) -> str:
    if device.type == "cuda":
        return "cuda"
    return "cpu"


def _indexed_device_argument(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index or 0}"
    return "cpu"


def _variant_evaluation(evaluation: dict[str, Any], key: str) -> dict[str, Any]:
    return evaluation.get(key) or {}


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _optional_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in {None, 0}:
        return None
    return float(value) / float(baseline)
