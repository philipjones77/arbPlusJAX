from pathlib import Path

import jax.numpy as jnp

from arbplusjax import runtime
from tools.runtime_manifest import collect_runtime_manifest
from tools.runtime_manifest import write_runtime_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_collect_runtime_manifest_has_required_keys():
    manifest = collect_runtime_manifest(REPO_ROOT, jax_mode="cpu", python_path="/tmp/python")

    assert manifest["repo_root"] == str(REPO_ROOT)
    assert manifest["environment_kind"] in {"local", "wsl", "colab"}
    assert manifest["python"]["executable"] == "/tmp/python"
    assert manifest["jax"]["requested_mode"] == "cpu"
    assert "system" in manifest["os"]
    assert "commit" in manifest["git"]


def test_write_runtime_manifest_emits_json_file(tmp_path: Path):
    manifest = collect_runtime_manifest(REPO_ROOT, jax_mode="auto")
    out = write_runtime_manifest(tmp_path, manifest)

    assert out.name == "runtime_manifest.json"
    text = out.read_text(encoding="utf-8")
    assert '"repo_root"' in text
    assert '"environment_kind"' in text


def test_runtime_config_builders_capture_dtype_batch_and_env():
    cfg = runtime.cpu_runtime(dtype="float32", dps=80, fixed_batch_size=64, pad_to=128, warmup=False)

    assert cfg.jax_mode == "cpu"
    assert cfg.precision.dtype == "float32"
    assert cfg.precision.prec_bits == runtime.normalize_prec_bits(dps=80)
    assert cfg.batch.fixed_batch_size == 64
    assert cfg.batch.pad_to == 128
    assert cfg.batch.warmup is False

    env = runtime.runtime_env(cfg)
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["JAX_ENABLE_X64"] == "0"


def test_runtime_dtype_resolution_and_manifest_attach_runtime_block():
    assert runtime.resolve_real_dtype("float32") == jnp.dtype(jnp.float32)
    assert runtime.resolve_complex_dtype("float32") == jnp.dtype(jnp.complex64)
    assert runtime.resolve_complex_dtype("complex128") == jnp.dtype(jnp.complex128)

    manifest = runtime.runtime_manifest(REPO_ROOT, config=runtime.gpu_runtime(dtype="float64", prec_bits=200))
    assert manifest["runtime"]["dtype"] == "float64"
    assert manifest["runtime"]["prec_bits"] == 200
    assert manifest["runtime"]["env_overrides"]["JAX_PLATFORMS"] == "cuda"
