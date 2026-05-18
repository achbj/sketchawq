from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeImage:
    @classmethod
    def debian_slim(cls, **_: object) -> "_FakeImage":
        return cls()

    def apt_install(self, *_: object, **__: object) -> "_FakeImage":
        return self

    def pip_install(self, *_: object, **__: object) -> "_FakeImage":
        return self

    def add_local_dir(self, *_: object, **__: object) -> "_FakeImage":
        return self

    def run_commands(self, *_: object, **__: object) -> "_FakeImage":
        return self


class _FakeApp:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def function(self, *_: object, **__: object) -> object:
        return lambda fn: fn

    def local_entrypoint(self, *_: object, **__: object) -> object:
        return lambda fn: fn


class _FakeSecret:
    @classmethod
    def from_local_environ(cls, *_: object, **__: object) -> "_FakeSecret":
        return cls()

    @classmethod
    def from_name(cls, *_: object, **__: object) -> "_FakeSecret":
        return cls()


class _FakeVolume:
    @classmethod
    def from_name(cls, *_: object, **__: object) -> "_FakeVolume":
        return cls()

    def commit(self) -> None:
        pass


def _fake_modal_module() -> types.ModuleType:
    module = types.ModuleType("modal")
    module.App = _FakeApp
    module.Image = _FakeImage
    module.Secret = _FakeSecret
    module.Volume = _FakeVolume
    return module


def _load_entrypoint(filename: str) -> types.ModuleType:
    path = PROJECT_ROOT / "infra" / "modal" / filename
    module_name = f"_test_{filename.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None

    previous_modal = sys.modules.get("modal")
    sys.modules["modal"] = _fake_modal_module()
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous_modal
    return module


def _make_project_root(path: Path) -> Path:
    (path / "src" / "sketchawq").mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname = 'sketchawq'\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("filename", ["sketchawq_app.py", "baseline_app.py", "inference_app.py"])
def test_modal_entrypoint_resolves_local_project_root(filename: str) -> None:
    module = _load_entrypoint(filename)

    assert module.PROJECT_ROOT == PROJECT_ROOT


@pytest.mark.parametrize("filename", ["sketchawq_app.py", "baseline_app.py", "inference_app.py"])
def test_modal_entrypoint_resolves_project_root_from_nested_file(
    filename: str,
    tmp_path: Path,
) -> None:
    module = _load_entrypoint(filename)
    project_root = _make_project_root(tmp_path / "repo")
    nested_file = project_root / "infra" / "modal" / filename
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("", encoding="utf-8")

    assert module._resolve_project_root(nested_file) == project_root.resolve()


@pytest.mark.parametrize("filename", ["sketchawq_app.py", "baseline_app.py", "inference_app.py"])
def test_modal_entrypoint_resolves_remote_project_root(filename: str, tmp_path: Path) -> None:
    module = _load_entrypoint(filename)
    remote_root = _make_project_root(tmp_path / "root" / "sketchawq")

    resolved = module._resolve_project_root(
        current_file="/root/sketchawq_app.py",
        remote_root=remote_root,
    )

    assert resolved == remote_root.resolve()
