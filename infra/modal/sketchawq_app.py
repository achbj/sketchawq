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

app = modal.App("sketchawq")
GPU_TYPE = os.environ.get("SKETCHAWQ_MODAL_GPU", "A10")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=0.30",
        "datasets>=2.18",
        "numpy>=1.24",
        "pyyaml>=6.0",
        "safetensors>=0.4",
        "sentencepiece>=0.2",
        "torch>=2.2",
        "tqdm>=4.66",
        "transformers>=4.51.0",
    )
    .add_local_dir(PROJECT_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=["artifacts", ".git"])
    .run_commands(f"pip install -e {REMOTE_ROOT}")
)

models_volume = modal.Volume.from_name("sketchawq-models", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("sketchawq-artifacts", create_if_missing=True)


def huggingface_secrets() -> list[modal.Secret]:
    """Return a Hugging Face token secret when the local run provides one.

    Supported patterns:
    - export HF_TOKEN=hf_...
    - modal secret create huggingface HF_TOKEN=hf_...
      export SKETCHAWQ_HF_SECRET=huggingface
    - default Modal secret name: huggingface-secret
    """
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
    timeout=60 * 60 * 6,
    memory=32768,
)
def quantize_remote(config_path: str = f"{REMOTE_ROOT}/configs/smoke/tiny.yaml") -> dict:
    from sketchawq.experiments.run_quantization import run_from_config

    report = run_from_config(
        config_path,
        model_cache_dir="/models/huggingface",
        output_root="/artifacts",
    )
    models_volume.commit()
    artifacts_volume.commit()
    return report


@app.local_entrypoint()
def main(config: str = "configs/smoke/tiny.yaml", configs: str = "") -> None:
    config_list = _split_config_list(configs) or [config]
    uses_llama = any("llama" in item.lower() for item in config_list)
    if uses_llama and os.environ.get("SKETCHAWQ_DISABLE_HF_SECRET") == "1":
        print(
            "This config appears to use a gated LLaMA model. Set HF_TOKEN locally or "
            "SKETCHAWQ_HF_SECRET to a Modal secret name before running."
        )

    reports = []
    for item in config_list:
        report = quantize_remote.remote(_to_remote_config_path(item))
        reports.append(report)
        print(json.dumps(report, indent=2))

    if len(reports) > 1:
        print(json.dumps({"completed_runs": len(reports)}, indent=2))


def _split_config_list(configs: str) -> list[str]:
    return [item.strip() for item in configs.split(",") if item.strip()]


def _to_remote_config_path(config_path: str) -> str:
    if config_path.startswith("/"):
        return config_path
    return f"{REMOTE_ROOT}/{config_path}"
