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

app = modal.App("sketchawq-inference")
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
def inference_remote(
    config_path: str,
    *,
    variant: str = "fp16",
    prompt_lengths: tuple[int, ...] = (128, 512, 2048),
    batch_sizes: tuple[int, ...] = (1, 4),
    generated_tokens: int = 128,
    warmup_runs: int = 1,
    repeats: int = 1,
    output_root: str = "/artifacts/inference",
    name: str | None = None,
    seed_text: str = "SketchAWQ inference benchmark prompt.",
) -> dict:
    os.environ.setdefault("HF_HOME", "/models/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/models/huggingface/hub")

    from sketchawq.experiments.benchmark_inference import benchmark_from_config

    report = benchmark_from_config(
        config_path,
        variant=variant,
        prompt_lengths=list(prompt_lengths),
        batch_sizes=list(batch_sizes),
        generated_tokens=generated_tokens,
        warmup_runs=warmup_runs,
        repeats=repeats,
        model_cache_dir="/models/huggingface",
        output_root=output_root,
        name=name,
        seed_text=seed_text,
    )
    models_volume.commit()
    artifacts_volume.commit()
    return report


@app.local_entrypoint()
def main(
    config: str = "configs/paper/qwen3_4b_4bit.yaml",
    variant: str = "fp16",
    prompt_lengths: str = "128,512,2048",
    batch_sizes: str = "1,4",
    generated_tokens: int = 128,
    warmup_runs: int = 1,
    repeats: int = 1,
    output_root: str = "/artifacts/inference",
    name: str = "",
    seed_text: str = "SketchAWQ inference benchmark prompt.",
) -> None:
    report = inference_remote.remote(
        _to_remote_config_path(config),
        variant=variant,
        prompt_lengths=tuple(_split_ints(prompt_lengths)),
        batch_sizes=tuple(_split_ints(batch_sizes)),
        generated_tokens=generated_tokens,
        warmup_runs=warmup_runs,
        repeats=repeats,
        output_root=output_root,
        name=name or None,
        seed_text=seed_text,
    )
    print(json.dumps(report, indent=2))


def _split_ints(value: str) -> list[int]:
    normalized = value.replace(",", " ")
    return [int(item) for item in normalized.split() if item.strip()]


def _to_remote_config_path(config_path: str) -> str:
    if config_path.startswith("/"):
        return config_path
    return f"{REMOTE_ROOT}/{config_path}"
