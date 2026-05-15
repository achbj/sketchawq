from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from sketchawq.config import (
    CalibrationConfig,
    EvaluationConfig,
    ModelConfig,
    OutputConfig,
    QuantizationConfig,
    RuntimeConfig,
)


BASELINE_METHODS = {"hqq", "gptq", "awq", "llmcompressor_awq"}
BASELINE_METHOD_ALIASES = {
    "hqq_transformers": "hqq",
    "gptqmodel": "gptq",
    "gptqmodel_gptq": "gptq",
    "gptqmodel-gptq": "gptq",
    "gptqmodel_awq": "awq",
    "gptqmodel-awq": "awq",
    "llm_compressor_awq": "llmcompressor_awq",
    "llm-compressor-awq": "llmcompressor_awq",
    "llmcompressor-awq": "llmcompressor_awq",
    "awq_llmcompressor": "llmcompressor_awq",
    "awq-llmcompressor": "llmcompressor_awq",
    "awq_llm_compressor": "llmcompressor_awq",
    "awq-llm-compressor": "llmcompressor_awq",
}
DEFAULT_LABELS = {
    "hqq": "HQQ",
    "gptq": "GPTQModel-GPTQ",
    "awq": "GPTQModel-AWQ",
    "llmcompressor_awq": "LLMCompressor-AWQ",
}
DEFAULT_PACKAGES = {
    "hqq": "hqq",
    "gptq": "gptqmodel",
    "awq": "gptqmodel",
    "llmcompressor_awq": "llmcompressor",
}


def _known_dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


@dataclass(slots=True)
class BaselineMethodConfig:
    name: str = "hqq"
    label: str | None = None
    package: str | None = None
    calibration_batch_size: int = 1
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    def normalized_name(self) -> str:
        method = self.name.strip().lower().replace(" ", "_")
        return BASELINE_METHOD_ALIASES.get(method, method)

    def display_label(self) -> str:
        method = self.normalized_name()
        return self.label or DEFAULT_LABELS.get(method, method)

    def package_name(self) -> str:
        method = self.normalized_name()
        return self.package or DEFAULT_PACKAGES.get(method, method)

    def validate(self) -> None:
        method = self.normalized_name()
        if method not in BASELINE_METHODS:
            allowed = ", ".join(sorted(BASELINE_METHODS))
            raise ValueError(f"baseline method must be one of {allowed}; got {self.name!r}")
        if self.calibration_batch_size <= 0:
            raise ValueError("method.calibration_batch_size must be positive")


@dataclass(slots=True)
class ExternalBaselineConfig:
    method: BaselineMethodConfig = field(default_factory=BaselineMethodConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(run_name="external_baseline"))
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        self.method.validate()
        self.quantization.validate()
        self.calibration.validate()
        self.evaluation.validate()


def _coerce_baseline_config(data: dict[str, Any]) -> ExternalBaselineConfig:
    return ExternalBaselineConfig(
        method=BaselineMethodConfig(
            **_known_dataclass_kwargs(BaselineMethodConfig, data.get("method", {}))
        ),
        model=ModelConfig(**_known_dataclass_kwargs(ModelConfig, data.get("model", {}))),
        quantization=QuantizationConfig(
            **_known_dataclass_kwargs(QuantizationConfig, data.get("quantization", {}))
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


def load_baseline_config(path: str | Path) -> ExternalBaselineConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = _coerce_baseline_config(raw)
    config.validate()
    return config


def baseline_config_to_dict(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    return config
