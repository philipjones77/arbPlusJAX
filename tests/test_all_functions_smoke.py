from __future__ import annotations

import csv
import importlib
import inspect
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from arbplusjax import acb_core, double_interval as di, fmpzi
from tests._test_checks import _check

TARGETS = Path(__file__).resolve().parent / "targets.csv"


def _interval_scalar() -> jax.Array:
    return di.interval(jnp.float64(0.1), jnp.float64(0.1))


def _interval_vec(n: int = 4) -> jax.Array:
    v = jnp.linspace(jnp.float64(0.1), jnp.float64(0.4), n)
    return di.interval(v, v)


def _box_scalar() -> jax.Array:
    return acb_core.acb_box(di.interval(jnp.float64(0.1), jnp.float64(0.1)), di.interval(jnp.float64(0.0), jnp.float64(0.0)))


def _box_vec(n: int = 4) -> jax.Array:
    z = jnp.linspace(jnp.float64(0.1), jnp.float64(0.4), n) + 0.1j
    re = di.interval(jnp.real(z), jnp.real(z))
    im = di.interval(jnp.imag(z), jnp.imag(z))
    return acb_core.acb_box(re, im)


def _complex_vec(n: int = 8) -> jax.Array:
    return jnp.linspace(jnp.float64(0.1), jnp.float64(0.8), n) + 0.1j


def _int_interval() -> jax.Array:
    return fmpzi.interval(jnp.int64(1), jnp.int64(1))


def _coeffs_acb() -> jax.Array:
    coeffs = jnp.zeros((4, 4), dtype=jnp.float64)
    # fill with small boxes
    return acb_core.acb_box(di.interval(coeffs, coeffs), di.interval(coeffs, coeffs))


def _coeffs_arb() -> jax.Array:
    coeffs = jnp.zeros((4, 2), dtype=jnp.float64)
    return coeffs


def _mat_acb() -> jax.Array:
    zeros = jnp.zeros((2, 2), dtype=jnp.float64)
    return acb_core.acb_box(di.interval(zeros, zeros), di.interval(zeros, zeros))


def _mat_arb() -> jax.Array:
    return jnp.zeros((2, 2, 2), dtype=jnp.float64)


def _bool_mat() -> jax.Array:
    return jnp.zeros((4,), dtype=jnp.uint8)


def _pick_value(func_name: str, param: inspect.Parameter) -> Any:
    pname = param.name

    # Explicit primitives
    if pname in ("prec_bits", "prec"):
        return 53
    if pname in ("dps",):
        return 50
    if pname in ("mode", "impl"):
        return "baseline"
    if pname in ("regularized", "scaled"):
        return False
    if pname in ("n", "m", "k", "terms", "length", "order", "max_terms", "min_terms", "N", "M"):
        return 4

    # Matrix / coeffs
    if "mat" in pname:
        return _mat_acb() if func_name.startswith("acb_") else _mat_arb()
    if "coeff" in pname or "poly" in pname:
        return _coeffs_acb() if func_name.startswith("acb_") else _coeffs_arb()
    if "bool" in pname:
        return _bool_mat()

    # Interval / box values by prefix
    if func_name.startswith("acb_"):
        if pname in ("x", "y", "z", "a", "b", "c", "s", "t", "u", "v", "w", "tau", "beta", "nu"):
            return _box_scalar()
        if pname in ("f", "g", "vec"):
            return _box_vec()
    if func_name.startswith("arb_") or func_name.startswith("dirichlet_"):
        if pname in ("x", "y", "z", "a", "b", "c", "s", "t", "u", "v", "w", "tau", "beta", "nu"):
            return _interval_scalar()
        if pname in ("f", "g", "vec"):
            return _interval_vec()

    if func_name.startswith("fmpzi_"):
        return _int_interval()

    if func_name.startswith("dft"):
        return _complex_vec()
    if func_name.startswith("acb_dft"):
        return _box_vec()

    # Fallback
    return jnp.float64(0.1)


def _call_function(func_name: str, func: Any) -> None:
    sig = inspect.signature(func)
    args = []
    kwargs = {}
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect._empty:
            # skip optional args
            continue
        val = _pick_value(func_name, param)
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            args.append(val)
        else:
            kwargs[param.name] = val

    out = func(*args, **kwargs)
    _check(out is not None)


def test_all_functions_smoke() -> None:
    if not TARGETS.exists():
        _check(False, "targets.csv missing")
        return

    with jax.disable_jit():
        failures = []
        with TARGETS.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["function"]
                module = row["module"]
                try:
                    mod = importlib.import_module(f"arbplusjax.{module}")
                    func = getattr(mod, name, None)
                    if func is None or not callable(func):
                        failures.append((name, module, "not callable"))
                        continue
                    _call_function(name, func)
                except Exception as exc:  # pragma: no cover - used for audit
                    failures.append((name, module, f"{type(exc).__name__}: {exc}"))

        _check(len(failures) == 0, f"failures: {len(failures)}")

        if failures:
            # write a debug report for triage
            report = Path(__file__).resolve().parent.parent / "results" / "test_failures_all_functions.txt"
            report.parent.mkdir(exist_ok=True)
            with report.open("w", encoding="utf-8") as r:
                for name, module, msg in failures:
                    r.write(f"{module}.{name}: {msg}\n")
