from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from arbplusjax import jax_diagnostics


def test_config_from_env_defaults(monkeypatch):
    monkeypatch.delenv("ARBPLUSJAX_JAX_DIAGNOSTICS_ENABLED", raising=False)
    monkeypatch.delenv("ARBPLUSJAX_JAX_DIAGNOSTICS_JAXPR", raising=False)
    monkeypatch.delenv("ARBPLUSJAX_JAX_DIAGNOSTICS_HLO", raising=False)
    cfg = jax_diagnostics.config_from_env()
    assert cfg.enabled is False
    assert cfg.capture_jaxpr is False
    assert cfg.capture_hlo is False


def test_profile_suite_and_report(tmp_path):
    fn = jax.jit(lambda x: x + 1.0)
    cases = [
        {
            "name": "add_one",
            "fn": fn,
            "arg": jnp.ones((4,), dtype=jnp.float64),
            "alt_arg": jnp.ones((6,), dtype=jnp.float64),
        }
    ]
    cfg = jax_diagnostics.JaxDiagnosticsConfig(enabled=True, capture_jaxpr=True, capture_hlo=False)
    profiles = jax_diagnostics.profile_function_suite(cases, repeats=2, config=cfg)
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.name == "add_one"
    assert profile.compile_ms >= 0.0
    assert profile.steady_ms_median >= 0.0
    assert profile.recompile_new_shape_ms >= 0.0
    assert profile.jaxpr is not None or profile.jaxpr_error is not None

    report = jax_diagnostics.write_profile_report(tmp_path / "diag.json", profiles)
    payload = json.loads(report.read_text())
    assert payload[0]["name"] == "add_one"
