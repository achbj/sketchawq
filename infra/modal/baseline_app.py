from __future__ import annotations

import json
import os
from pathlib import Path

import modal


REMOTE_ROOT = "/root/sketchawq"


def _looks_like_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "sketchawq").is_dir()


def _resolve_project_root(
    current_file: str | Path | None = None,
    remote_root: str | Path = REMOTE_ROOT,
) -> Path:
    """Find the local repo root or the copied project root inside Modal."""
    current = Path(current_file or __file__).resolve()
    for candidate in (current.parent, *current.parents):
        if _looks_like_project_root(candidate):
            return candidate

    remote_candidate = Path(remote_root).resolve()
    if _looks_like_project_root(remote_candidate):
        return remote_candidate

    raise RuntimeError(
        "Could not locate the SketchAWQ project root. Run Modal from the clean repo "
        "or ensure the image contains /root/sketchawq."
    )


PROJECT_ROOT = _resolve_project_root()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

app = modal.App("sketchawq-baselines")
GPU_TYPE = os.environ.get("SKETCHAWQ_MODAL_GPU", "A10")

BASELINE_PACKAGES = (
    "accelerate>=0.30",
    "autoawq",
    "datasets>=2.18",
    "gptqmodel",
    "hqq",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "safetensors>=0.4",
    "sentencepiece>=0.2",
    "torch>=2.2",
    "torchvision",
    "tqdm>=4.66",
    "transformers>=4.51.0",
)
LLMCOMPRESSOR_BASELINE_PACKAGES = (
    "llmcompressor==0.10.0.2",
    "pyyaml>=6.0",
    "safetensors>=0.4",
    "sentencepiece>=0.2",
)
LLMCOMPRESSOR_TORCH_PACKAGES = (
    "torch>=2.10,<2.11",
    "torchvision>=0.25,<0.26",
)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "ninja-build")
    .pip_install(*BASELINE_PACKAGES)
    .add_local_dir(PROJECT_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=["artifacts", ".git"])
    .run_commands(f"pip install --no-deps -e {REMOTE_ROOT}")
)

llmcompressor_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "ninja-build")
    .pip_install(*LLMCOMPRESSOR_TORCH_PACKAGES)
    .pip_install(*LLMCOMPRESSOR_BASELINE_PACKAGES)
    .add_local_dir(PROJECT_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=["artifacts", ".git"])
    .run_commands(f"pip install --no-deps -e {REMOTE_ROOT}")
)

models_volume = modal.Volume.from_name("sketchawq-models", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("sketchawq-artifacts", create_if_missing=True)


def huggingface_secrets() -> list[modal.Secret]:
    if os.environ.get("SKETCHAWQ_DISABLE_HF_SECRET") == "1":
        return []
    if os.environ.get("HF_TOKEN"):
        return [modal.Secret.from_local_environ(["HF_TOKEN"])]
    secret_name = os.environ.get("SKETCHAWQ_HF_SECRET", "huggingface-secret")
    return [modal.Secret.from_name(secret_name, required_keys=["HF_TOKEN"])]


@app.function(
    image=image,
    gpu=GPU_TYPE,
    secrets=huggingface_secrets(),
    volumes={"/models": models_volume, "/artifacts": artifacts_volume},
    timeout=60 * 60 * 8,
    memory=32768,
)
def baseline_remote(config_path: str, fail_soft: bool = True) -> dict:
    return _run_baseline_remote(config_path, fail_soft=fail_soft)


@app.function(
    image=llmcompressor_image,
    gpu=GPU_TYPE,
    secrets=huggingface_secrets(),
    volumes={"/models": models_volume, "/artifacts": artifacts_volume},
    timeout=60 * 60 * 8,
    memory=32768,
)
def llmcompressor_baseline_remote(config_path: str, fail_soft: bool = True) -> dict:
    return _run_baseline_remote(config_path, fail_soft=fail_soft)


@app.local_entrypoint()
def main(config: str = "configs/baselines/qwen/qwen2_5_1_5b_hqq_4bit.yaml", configs: str = "") -> None:
    config_list = _split_config_list(configs) or [config]
    reports = []
    for item in config_list:
        report = _run_remote_baseline(item)
        reports.append(report)
        print(json.dumps(report, indent=2))
    if len(reports) > 1:
        print(json.dumps({"completed_baseline_runs": len(reports)}, indent=2))


def _split_config_list(configs: str) -> list[str]:
    return [item.strip() for item in configs.split(",") if item.strip()]


def _to_remote_config_path(config_path: str) -> str:
    if config_path.startswith("/"):
        return config_path
    return f"{REMOTE_ROOT}/{config_path}"


def _run_remote_baseline(config_path: str) -> dict:
    remote_config_path = _to_remote_config_path(config_path)
    if _uses_llmcompressor_awq(config_path):
        return llmcompressor_baseline_remote.remote(remote_config_path)
    return baseline_remote.remote(remote_config_path)


def _run_baseline_remote(config_path: str, fail_soft: bool = True) -> dict:
    os.environ.setdefault("HF_HOME", "/models/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/models/huggingface/hub")
    os.environ.setdefault("PYTEST_CURRENT_TEST", "sketchawq_modal_noninteractive")

    from sketchawq.baselines.runner import write_failed_baseline_from_config
    from sketchawq.experiments.run_baseline import run_from_config

    try:
        report = run_from_config(
            config_path,
            model_cache_dir="/models/huggingface",
            output_root="/artifacts",
        )
    except Exception as exc:
        if not fail_soft:
            raise
        report = write_failed_baseline_from_config(
            config_path,
            error=exc,
            output_root="/artifacts",
        )
    models_volume.commit()
    artifacts_volume.commit()
    return report


def _uses_llmcompressor_awq(config_path: str) -> bool:
    try:
        import yaml

        path = Path(config_path)
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        method = str(config.get("method", {}).get("name", "")).lower().replace("-", "_")
        return method in {
            "llm_compressor_awq",
            "llmcompressor_awq",
            "awq_llm_compressor",
            "awq_llmcompressor",
        }
    except OSError:
        return "llmcompressor" in config_path.lower() or "llm_compressor" in config_path.lower()
