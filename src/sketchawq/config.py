from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


SKETCH_MODES = {"none", "diagonal", "diagonal_low_rank"}
SKETCH_MODE_ALIASES = {
    "awq_only": "none",
    "mse": "none",
    "diag": "diagonal",
    "diag_low_rank": "diagonal_low_rank",
    "low_rank": "diagonal_low_rank",
}
AWQ_SCALE_NORMALIZATION_MODES = {"geometric_mean", "arithmetic_mean", "none"}
AWQ_SCALE_NORMALIZATION_ALIASES = {
    "geometric": "geometric_mean",
    "geomean": "geometric_mean",
    "geo_mean": "geometric_mean",
    "mean": "arithmetic_mean",
    "arithmetic": "arithmetic_mean",
    "arith_mean": "arithmetic_mean",
    "raw": "none",
    "off": "none",
    "false": "none",
}
SEARCH_OBJECTIVES = {"weighted_error", "output_error", "hybrid"}
SEARCH_OBJECTIVE_ALIASES = {
    "weighted": "weighted_error",
    "weight": "weighted_error",
    "hessian": "weighted_error",
    "hessian_weight": "weighted_error",
    "output": "output_error",
    "output_aware": "output_error",
    "activation_output": "output_error",
}
SCALE_REFINEMENT_MODES = {"none", "sketch"}
SCALE_REFINEMENT_ALIASES = {
    "off": "none",
    "disabled": "none",
    "false": "none",
    "hessian": "sketch",
    "curvature": "sketch",
    "sketch_conditioned": "sketch",
}
ROW_SAMPLING_MODES = {"random", "linspace"}
ROW_SAMPLING_ALIASES = {
    "seeded_random": "random",
    "randperm": "random",
    "stride": "linspace",
    "uniform_stride": "linspace",
}
MASK_MISMATCH_ACTIONS = {"warn", "error", "ignore"}
MASK_MISMATCH_ACTION_ALIASES = {
    "raise": "error",
    "strict": "error",
    "silent": "ignore",
    "off": "ignore",
}
CHANNEL_SENSITIVITY_METRICS = {
    "activation",
    "hessian",
    "weight",
    "activation_weight",
    "hessian_weight",
}
PACKING_STRATEGIES = {"stream", "concat"}
PACKING_STRATEGY_ALIASES = {
    "default": "stream",
    "eos": "stream",
    "eos_stream": "stream",
    "gptq": "concat",
    "joined": "concat",
    "concatenate": "concat",
    "concatenated": "concat",
}


def _known_dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


@dataclass(slots=True)
class ModelConfig:
    repo_id: str = "sshleifer/tiny-gpt2"
    revision: str | None = None
    dtype: str = "float16"
    trust_remote_code: bool = False
    local_files_only: bool = False


@dataclass(slots=True)
class QuantizationConfig:
    bits: int = 4
    group_size: int = 128
    symmetric: bool = True
    protected_scale: float = 1.0
    scale_clip_search: str = "joint"
    awq_scale_normalization: str = "geometric_mean"
    awq_min_scale: float = 0.25
    awq_max_scale: float = 4.0
    awq_alpha_candidates: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])
    clip_candidates: list[float] = field(default_factory=lambda: [1.0, 0.98, 0.95, 0.9, 0.85, 0.8])
    search_objective: str = "weighted_error"
    output_error_rows: int = 0
    output_error_weight: float = 1.0
    hessian_error_weight: float = 1.0
    scale_refinement: str = "none"
    scale_refinement_steps: int = 3

    def validate(self) -> None:
        if self.bits not in {2, 3, 4, 8}:
            raise ValueError(f"bits must be one of 2, 3, 4, 8; got {self.bits}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        self.scale_clip_search = self.scale_clip_search.strip().lower().replace("-", "_")
        if self.scale_clip_search not in {"greedy", "joint"}:
            raise ValueError("scale_clip_search must be either 'greedy' or 'joint'")
        self.awq_scale_normalization = (
            self.awq_scale_normalization.strip().lower().replace("-", "_")
        )
        self.awq_scale_normalization = AWQ_SCALE_NORMALIZATION_ALIASES.get(
            self.awq_scale_normalization,
            self.awq_scale_normalization,
        )
        if self.awq_scale_normalization not in AWQ_SCALE_NORMALIZATION_MODES:
            allowed = ", ".join(sorted(AWQ_SCALE_NORMALIZATION_MODES))
            raise ValueError(f"awq_scale_normalization must be one of {allowed}")
        if self.awq_min_scale <= 0:
            raise ValueError("awq_min_scale must be positive")
        if self.awq_max_scale <= 0:
            raise ValueError("awq_max_scale must be positive")
        if self.awq_min_scale > self.awq_max_scale:
            raise ValueError("awq_min_scale must be <= awq_max_scale")
        if not self.awq_alpha_candidates:
            raise ValueError("awq_alpha_candidates must not be empty")
        if not self.clip_candidates:
            raise ValueError("clip_candidates must not be empty")
        self.search_objective = self.search_objective.strip().lower().replace("-", "_")
        self.search_objective = SEARCH_OBJECTIVE_ALIASES.get(
            self.search_objective,
            self.search_objective,
        )
        if self.search_objective not in SEARCH_OBJECTIVES:
            allowed = ", ".join(sorted(SEARCH_OBJECTIVES))
            raise ValueError(f"search_objective must be one of {allowed}")
        if self.output_error_rows < 0:
            raise ValueError("output_error_rows must be non-negative")
        if self.search_objective in {"output_error", "hybrid"} and self.output_error_rows <= 0:
            raise ValueError("output_error_rows must be positive for output-aware search")
        if self.output_error_weight < 0:
            raise ValueError("output_error_weight must be non-negative")
        if self.hessian_error_weight < 0:
            raise ValueError("hessian_error_weight must be non-negative")
        self.scale_refinement = self.scale_refinement.strip().lower().replace("-", "_")
        self.scale_refinement = SCALE_REFINEMENT_ALIASES.get(
            self.scale_refinement,
            self.scale_refinement,
        )
        if self.scale_refinement not in SCALE_REFINEMENT_MODES:
            allowed = ", ".join(sorted(SCALE_REFINEMENT_MODES))
            raise ValueError(f"scale_refinement must be one of {allowed}")
        if self.scale_refinement_steps < 0:
            raise ValueError("scale_refinement_steps must be non-negative")
        if self.scale_refinement == "sketch":
            if self.scale_refinement_steps <= 0:
                raise ValueError(
                    "scale_refinement_steps must be positive for sketch refinement"
                )
            if not self.symmetric:
                raise ValueError(
                    "sketch scale refinement currently requires symmetric quantization"
                )


@dataclass(slots=True)
class OutlierConfig:
    enabled: bool = False
    strategy: str = "scale"
    metric: str = "hessian_weight"
    fraction: float = 0.0
    min_channels: int = 0
    max_channels: int | None = None
    scale_boost: float = 2.0
    renormalize_scales: bool = True
    protected_bits: int = 16

    def validate(self) -> None:
        self.strategy = self.strategy.strip().lower().replace("-", "_")
        strategy_aliases = {
            "protected": "protect",
            "protected_channel": "protect",
            "protected_channels": "protect",
            "restore": "protect",
            "restored": "protect",
        }
        self.strategy = strategy_aliases.get(self.strategy, self.strategy)
        if self.strategy not in {"scale", "protect"}:
            raise ValueError("outlier strategy must be either 'scale' or 'protect'")
        self.metric = self.metric.strip().lower().replace("-", "_")
        allowed_metrics = {
            "activation",
            "hessian",
            "weight",
            "activation_weight",
            "hessian_weight",
        }
        if self.metric not in allowed_metrics:
            allowed = ", ".join(sorted(allowed_metrics))
            raise ValueError(f"outlier metric must be one of {allowed}; got {self.metric!r}")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("outlier fraction must be between 0.0 and 1.0")
        if self.min_channels < 0:
            raise ValueError("outlier min_channels must be non-negative")
        if self.max_channels is not None and self.max_channels < 0:
            raise ValueError("outlier max_channels must be non-negative when set")
        if self.scale_boost <= 0:
            raise ValueError("outlier scale_boost must be positive")
        if self.protected_bits <= 0:
            raise ValueError("outlier protected_bits must be positive")

    def active(self) -> bool:
        return self.enabled and (self.fraction > 0.0 or self.min_channels > 0)

    def protects_channels(self) -> bool:
        return self.active() and self.strategy == "protect"


@dataclass(slots=True)
class SketchConfig:
    enabled: bool = True
    mode: str = "diagonal_low_rank"
    rank: int = 8
    max_rows_per_batch: int = 256
    row_sampling: str = "random"
    mask_mismatch_action: str = "warn"
    diagonal_floor: float = 1e-6
    low_rank_weight: float = 0.15
    normalize_error_terms: bool = True

    def validate(self) -> None:
        mode = self.normalized_mode()
        if mode not in SKETCH_MODES:
            allowed = ", ".join(sorted(SKETCH_MODES))
            raise ValueError(f"sketch mode must be one of {allowed}; got {self.mode!r}")
        if self.rank < 0:
            raise ValueError("sketch rank must be non-negative")
        if self.max_rows_per_batch <= 0:
            raise ValueError("max_rows_per_batch must be positive")
        self.row_sampling = self.row_sampling.strip().lower().replace("-", "_")
        self.row_sampling = ROW_SAMPLING_ALIASES.get(self.row_sampling, self.row_sampling)
        if self.row_sampling not in ROW_SAMPLING_MODES:
            allowed = ", ".join(sorted(ROW_SAMPLING_MODES))
            raise ValueError(f"row_sampling must be one of {allowed}")
        self.mask_mismatch_action = self.mask_mismatch_action.strip().lower().replace("-", "_")
        self.mask_mismatch_action = MASK_MISMATCH_ACTION_ALIASES.get(
            self.mask_mismatch_action,
            self.mask_mismatch_action,
        )
        if self.mask_mismatch_action not in MASK_MISMATCH_ACTIONS:
            allowed = ", ".join(sorted(MASK_MISMATCH_ACTIONS))
            raise ValueError(f"mask_mismatch_action must be one of {allowed}")
        if self.low_rank_weight < 0:
            raise ValueError("low_rank_weight must be non-negative")

    def normalized_mode(self) -> str:
        if not self.enabled:
            return "none"
        mode = self.mode.strip().lower().replace("-", "_")
        return SKETCH_MODE_ALIASES.get(mode, mode)

    def effective_mode(self) -> str:
        mode = self.normalized_mode()
        if mode == "diagonal_low_rank" and self.rank <= 0:
            return "diagonal"
        return mode

    def uses_diagonal(self) -> bool:
        return self.effective_mode() in {"diagonal", "diagonal_low_rank"}

    def uses_low_rank(self) -> bool:
        return self.effective_mode() == "diagonal_low_rank"

    def effective_rank(self) -> int:
        return self.rank if self.uses_low_rank() else 0


@dataclass(slots=True)
class AdaptiveConfig:
    enabled: bool = False
    sensitivity_metric: str = "hessian_weight"
    sensitivity_top_fraction: float = 0.01
    protected_fraction_min: float = 0.0
    protected_fraction_max: float | None = None
    sensitive_group_size: int | None = None
    group_size_top_fraction: float = 0.25

    def validate(self) -> None:
        self.sensitivity_metric = self.sensitivity_metric.strip().lower().replace("-", "_")
        if self.sensitivity_metric not in CHANNEL_SENSITIVITY_METRICS:
            allowed = ", ".join(sorted(CHANNEL_SENSITIVITY_METRICS))
            raise ValueError(f"adaptive sensitivity_metric must be one of {allowed}")
        if not 0.0 < self.sensitivity_top_fraction <= 1.0:
            raise ValueError("adaptive sensitivity_top_fraction must be in (0.0, 1.0]")
        if not 0.0 <= self.protected_fraction_min <= 1.0:
            raise ValueError("adaptive protected_fraction_min must be between 0.0 and 1.0")
        if self.protected_fraction_max is not None:
            if not 0.0 <= self.protected_fraction_max <= 1.0:
                raise ValueError("adaptive protected_fraction_max must be between 0.0 and 1.0")
            if self.protected_fraction_max < self.protected_fraction_min:
                raise ValueError(
                    "adaptive protected_fraction_max must be >= protected_fraction_min"
                )
        if self.sensitive_group_size is not None and self.sensitive_group_size <= 0:
            raise ValueError("adaptive sensitive_group_size must be positive when set")
        if not 0.0 <= self.group_size_top_fraction <= 1.0:
            raise ValueError("adaptive group_size_top_fraction must be between 0.0 and 1.0")


@dataclass(slots=True)
class CalibrationConfig:
    source: str = "synthetic"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "train"
    text_column: str = "text"
    num_samples: int = 8
    sequence_length: int = 128
    seed: int = 42
    packing: bool = False
    packing_strategy: str = "stream"
    batch_size: int = 1

    def validate(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("calibration.batch_size must be positive")
        self.packing_strategy = self.packing_strategy.strip().lower().replace("-", "_")
        self.packing_strategy = PACKING_STRATEGY_ALIASES.get(
            self.packing_strategy,
            self.packing_strategy,
        )
        if self.packing_strategy not in PACKING_STRATEGIES:
            allowed = ", ".join(sorted(PACKING_STRATEGIES))
            raise ValueError(f"calibration.packing_strategy must be one of {allowed}")


@dataclass(slots=True)
class EvaluationConfig:
    enabled: bool = True
    compare_original: bool = True
    benchmark_name: str = "calibration_perplexity"
    use_calibration_data: bool = True
    source: str = "dataset"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "validation"
    text_column: str = "text"
    num_samples: int | None = 32
    sequence_length: int = 1024
    packing: bool = False
    packing_strategy: str = "stream"
    batch_size: int = 1
    max_batches: int | None = 4
    measure_throughput: bool = False
    throughput_warmup_batches: int = 1

    def validate(self) -> None:
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("evaluation.num_samples must be positive")
        if self.sequence_length <= 0:
            raise ValueError("evaluation.sequence_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("evaluation.batch_size must be positive")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("evaluation.max_batches must be positive")
        if self.throughput_warmup_batches < 0:
            raise ValueError("evaluation.throughput_warmup_batches must be non-negative")
        self.packing_strategy = self.packing_strategy.strip().lower().replace("-", "_")
        self.packing_strategy = PACKING_STRATEGY_ALIASES.get(
            self.packing_strategy,
            self.packing_strategy,
        )
        if self.packing_strategy not in PACKING_STRATEGIES:
            allowed = ", ".join(sorted(PACKING_STRATEGIES))
            raise ValueError(f"evaluation.packing_strategy must be one of {allowed}")


@dataclass(slots=True)
class OutputConfig:
    run_name: str = "tiny_debug_sketchawq"
    output_dir: str = "artifacts"
    save_model: bool = True
    save_dense_simulation: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    seed: int = 42
    log_level: str = "INFO"


@dataclass(slots=True)
class SketchAWQConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    sketch: SketchConfig = field(default_factory=SketchConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        self.quantization.validate()
        self.outliers.validate()
        self.sketch.validate()
        self.adaptive.validate()
        if (
            self.adaptive.enabled
            and self.adaptive.sensitive_group_size is not None
            and self.adaptive.sensitive_group_size > self.quantization.group_size
        ):
            raise ValueError(
                "adaptive sensitive_group_size must be <= quantization.group_size"
            )
        self.calibration.validate()
        self.evaluation.validate()


def _coerce_config(data: dict[str, Any]) -> SketchAWQConfig:
    return SketchAWQConfig(
        model=ModelConfig(**_known_dataclass_kwargs(ModelConfig, data.get("model", {}))),
        quantization=QuantizationConfig(
            **_known_dataclass_kwargs(QuantizationConfig, data.get("quantization", {}))
        ),
        outliers=OutlierConfig(**_known_dataclass_kwargs(OutlierConfig, data.get("outliers", {}))),
        sketch=SketchConfig(**_known_dataclass_kwargs(SketchConfig, data.get("sketch", {}))),
        adaptive=AdaptiveConfig(
            **_known_dataclass_kwargs(AdaptiveConfig, data.get("adaptive", {}))
        ),
        calibration=CalibrationConfig(
            **_known_dataclass_kwargs(CalibrationConfig, data.get("calibration", {}))
        ),
        evaluation=EvaluationConfig(
            **_known_dataclass_kwargs(EvaluationConfig, data.get("evaluation", {}))
        ),
        output=OutputConfig(**_known_dataclass_kwargs(OutputConfig, data.get("output", {}))),
        runtime=RuntimeConfig(**_known_dataclass_kwargs(RuntimeConfig, data.get("runtime", {}))),
    )


def load_config(path: str | Path) -> SketchAWQConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = _coerce_config(raw)
    config.validate()
    return config


def config_to_dict(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    return config
