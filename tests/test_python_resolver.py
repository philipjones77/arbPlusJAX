from pathlib import Path

from tools import python_resolver


def test_resolve_python_prefers_explicit_override(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit-python"
    explicit.write_text("", encoding="utf-8")

    monkeypatch.setattr(python_resolver, "_candidates", lambda: [])
    monkeypatch.setattr(python_resolver.sys, "executable", "/usr/bin/python-current")

    assert python_resolver.resolve_python(str(explicit)) == str(explicit)


def test_resolve_python_prefers_linux_jax_env_by_default(monkeypatch):
    jax_python = Path("/home/test/miniforge3/envs/jax/bin/python")

    monkeypatch.setattr(python_resolver, "_system_name", lambda: "linux")
    monkeypatch.setattr(python_resolver, "_candidates", lambda: [jax_python])
    monkeypatch.setattr(Path, "exists", lambda self: self == jax_python)
    monkeypatch.setattr(python_resolver.sys, "executable", "/usr/bin/python-current")
    monkeypatch.delenv("ARBPLUSJAX_PYTHON", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

    assert python_resolver.resolve_python() == str(jax_python)


def test_resolve_python_falls_back_to_current_interpreter_without_jax_env(monkeypatch):
    monkeypatch.setattr(python_resolver, "_system_name", lambda: "linux")
    monkeypatch.setattr(python_resolver, "_candidates", lambda: [])
    monkeypatch.setattr(python_resolver.sys, "executable", "/usr/bin/python-current")
    monkeypatch.delenv("ARBPLUSJAX_PYTHON", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

    assert python_resolver.resolve_python() == "/usr/bin/python-current"
