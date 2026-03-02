from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp

from arbplusjax import api, double_interval as di, hypgeom, hypgeom_wrappers

from tests._test_checks import _check


_INT_NAMES = {
    "n",
    "m",
    "k",
    "length",
    "n_terms",
    "offset",
    "type",
    "p",
    "q",
    "work_prec",
    "prec_bits",
}
_BOOL_NAMES = {"regularized", "reciprocal", "complementary", "normalized", "scaled"}


def _iv_scalar() -> jax.Array:
    return di.interval(jnp.float64(0.2), jnp.float64(0.25))


def _iv_vec(n: int = 3) -> jax.Array:
    v = jnp.linspace(jnp.float64(0.15), jnp.float64(0.35), n)
    return di.interval(v, v + jnp.float64(0.02))


def _arg_for_param(func_name: str, pname: str):
    is_batch = "_batch" in func_name
    if pname == "mode":
        return "sample"
    if pname == "integrand":
        return "exp"
    if pname in _BOOL_NAMES:
        return False
    if pname in _INT_NAMES:
        return 4
    if pname in {"a", "b", "c", "s", "t", "u", "v", "w", "x", "y", "z", "nu", "eta", "lam", "zinv", "x2sub1"}:
        return _iv_vec() if is_batch else _iv_scalar()
    return _iv_vec() if is_batch else _iv_scalar()


def _build_args_from_prec_name(prec_name: str) -> tuple[list, dict]:
    base_name = prec_name[:-5] if prec_name.endswith("_prec") else prec_name
    fn = getattr(hypgeom, base_name, None)
    if fn is None:
        fn = getattr(hypgeom, prec_name)
    sig = inspect.signature(fn)
    args = []
    kwargs = {}
    for p in sig.parameters.values():
        if p.name == "prec_bits":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        val = _arg_for_param(base_name, p.name)
        if p.default is inspect._empty:
            args.append(val)
        else:
            kwargs[p.name] = val
    return args, kwargs


def test_arb_hypgeom_four_modes_bindings_complete() -> None:
    hyp_names = [n for n in dir(hypgeom) if n.startswith("arb_hypgeom_") and callable(getattr(hypgeom, n))]
    mode_wrappers = {n for n in dir(hypgeom_wrappers) if n.startswith("arb_hypgeom_") and n.endswith("_mode")}

    missing_point = []
    missing_mode = []
    for name in hyp_names:
        try:
            api.bind_point(f"hypgeom.{name}")
        except Exception:
            missing_point.append(name)
        if name.endswith("_prec"):
            mode_name = name.replace("_prec", "_mode")
            if mode_name not in mode_wrappers:
                missing_mode.append(mode_name)

    _check(len(missing_point) == 0)
    _check(len(missing_mode) == 0)


def test_arb_hypgeom_interval_modes_runtime_matrix() -> None:
    # Broad family coverage in all three interval modes.
    matrix = [
        ("arb_hypgeom_gamma_mode", [_iv_scalar()], {}),
        ("arb_hypgeom_rgamma_mode", [_iv_scalar()], {}),
        ("arb_hypgeom_erf_mode", [_iv_scalar()], {}),
        ("arb_hypgeom_erfc_mode", [_iv_scalar()], {}),
        ("arb_hypgeom_erfi_mode", [_iv_scalar()], {}),
        ("arb_hypgeom_0f1_mode", [_iv_scalar(), _iv_scalar()], {}),
        ("arb_hypgeom_1f1_mode", [_iv_scalar(), _iv_scalar(), _iv_scalar()], {}),
        ("arb_hypgeom_2f1_mode", [_iv_scalar(), _iv_scalar(), _iv_scalar(), _iv_scalar()], {}),
        ("arb_hypgeom_u_mode", [_iv_scalar(), _iv_scalar(), _iv_scalar()], {}),
        ("arb_hypgeom_bessel_j_mode", [_iv_scalar(), _iv_scalar()], {}),
        ("arb_hypgeom_bessel_y_mode", [_iv_scalar(), _iv_scalar()], {}),
        ("arb_hypgeom_bessel_i_mode", [_iv_scalar(), _iv_scalar()], {}),
        ("arb_hypgeom_bessel_k_mode", [_iv_scalar(), _iv_scalar()], {}),
        ("arb_hypgeom_li_mode", [_iv_scalar()], {}),
        ("arb_hypgeom_dilog_mode", [_iv_scalar()], {}),
        ("arb_hypgeom_expint_mode", [_iv_scalar(), _iv_scalar()], {}),
    ]
    for mode_name, args, kwargs in matrix:
        fn = getattr(hypgeom_wrappers, mode_name)
        for impl in ("basic", "adaptive", "rigorous"):
            out = fn(*args, impl=impl, dps=40, **kwargs)
            _check(out is not None)
