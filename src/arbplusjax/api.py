from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from typing import Callable

import jax
import jax.numpy as jnp

from . import baseline_wrappers
from . import cubesselk
from . import hypgeom_wrappers
from . import point_wrappers

# Public API for optimized calls.
# - eval_point: point-only kernels (fastest, no bounds)
# - eval_interval: interval kernels (basic/adaptive/rigorous) with optional batching


_POINT_FUNCS = {
    "exp": point_wrappers.arb_exp_point,
    "log": point_wrappers.arb_log_point,
    "sqrt": point_wrappers.arb_sqrt_point,
    "sin": point_wrappers.arb_sin_point,
    "cos": point_wrappers.arb_cos_point,
    "tan": point_wrappers.arb_tan_point,
    "sinh": point_wrappers.arb_sinh_point,
    "cosh": point_wrappers.arb_cosh_point,
    "tanh": point_wrappers.arb_tanh_point,
    "abs": point_wrappers.arb_abs_point,
    "add": point_wrappers.arb_add_point,
    "sub": point_wrappers.arb_sub_point,
    "mul": point_wrappers.arb_mul_point,
    "div": point_wrappers.arb_div_point,
    "inv": point_wrappers.arb_inv_point,
    "fma": point_wrappers.arb_fma_point,
    "log1p": point_wrappers.arb_log1p_point,
    "expm1": point_wrappers.arb_expm1_point,
    "sin_cos": point_wrappers.arb_sin_cos_point,
    "sinh_cosh": point_wrappers.arb_sinh_cosh_point,
    "sin_pi": point_wrappers.arb_sin_pi_point,
    "cos_pi": point_wrappers.arb_cos_pi_point,
    "tan_pi": point_wrappers.arb_tan_pi_point,
    "sinc": point_wrappers.arb_sinc_point,
    "sinc_pi": point_wrappers.arb_sinc_pi_point,
    "asin": point_wrappers.arb_asin_point,
    "acos": point_wrappers.arb_acos_point,
    "atan": point_wrappers.arb_atan_point,
    "asinh": point_wrappers.arb_asinh_point,
    "acosh": point_wrappers.arb_acosh_point,
    "atanh": point_wrappers.arb_atanh_point,
    "sign": point_wrappers.arb_sign_point,
    "pow": point_wrappers.arb_pow_point,
    "pow_ui": point_wrappers.arb_pow_ui_point,
    "root_ui": point_wrappers.arb_root_ui_point,
    "cbrt": point_wrappers.arb_cbrt_point,
    "gamma": point_wrappers.arb_gamma_point,
    "erf": point_wrappers.arb_erf_point,
    "erfc": point_wrappers.arb_erfc_point,
    "besselj": point_wrappers.arb_bessel_j_point,
    "bessely": point_wrappers.arb_bessel_y_point,
    "besseli": point_wrappers.arb_bessel_i_point,
    "besselk": point_wrappers.arb_bessel_k_point,
    "cubesselk": cubesselk.cubesselk_point,
    "CubesselK": cubesselk.cubesselk_point,
}


_INTERVAL_FUNCS = {
    "exp": baseline_wrappers.arb_exp_mp,
    "log": baseline_wrappers.arb_log_mp,
    "sqrt": baseline_wrappers.arb_sqrt_mp,
    "sin": baseline_wrappers.arb_sin_mp,
    "cos": baseline_wrappers.arb_cos_mp,
    "tan": baseline_wrappers.arb_tan_mp,
    "sinh": baseline_wrappers.arb_sinh_mp,
    "cosh": baseline_wrappers.arb_cosh_mp,
    "tanh": baseline_wrappers.arb_tanh_mp,
    "abs": baseline_wrappers.arb_abs_mp,
    "add": baseline_wrappers.arb_add_mp,
    "sub": baseline_wrappers.arb_sub_mp,
    "mul": baseline_wrappers.arb_mul_mp,
    "div": baseline_wrappers.arb_div_mp,
    "inv": baseline_wrappers.arb_inv_mp,
    "fma": baseline_wrappers.arb_fma_mp,
    "log1p": baseline_wrappers.arb_log1p_mp,
    "expm1": baseline_wrappers.arb_expm1_mp,
    "sin_cos": baseline_wrappers.arb_sin_cos_mp,
    "gamma": baseline_wrappers.arb_gamma_mp,
    "erf": baseline_wrappers.arb_erf_mp,
    "erfc": baseline_wrappers.arb_erfc_mp,
    "erfi": baseline_wrappers.arb_erfi_mp,
    "barnesg": baseline_wrappers.arb_barnesg_mp,
    "besselj": baseline_wrappers.arb_bessel_j_mp,
    "bessely": baseline_wrappers.arb_bessel_y_mp,
    "besseli": baseline_wrappers.arb_bessel_i_mp,
    "besselk": baseline_wrappers.arb_bessel_k_mp,
    "cubesselk": cubesselk.arb_cubesselk_mp,
    "CubesselK": cubesselk.arb_cubesselk_mp,
}


_MODULE_NAMES = (
    "acb_calc",
    "acb_core",
    "acb_dirichlet",
    "acb_elliptic",
    "acb_mat",
    "acb_modular",
    "acb_poly",
    "acf",
    "arb_calc",
    "arb_core",
    "arb_fmpz_poly",
    "arb_fpwrap",
    "arb_mat",
    "arb_poly",
    "arf",
    "bernoulli",
    "bool_mat",
    "dft",
    "dlog",
    "boost_hypgeom",
    "cusf_compat",
    "double_interval",
    "fmpz_extras",
    "fmpzi",
    "fmpr",
    "hypgeom",
    "mag",
    "partitions",
)


@lru_cache(maxsize=1)
def _public_registry() -> dict[str, Callable]:
    mapping: dict[str, Callable] = {}
    seen: dict[str, str] = {}
    for mod_name in _MODULE_NAMES:
        mod = importlib.import_module(f".{mod_name}", package=__package__)
        mod_name = mod.__name__.rsplit(".", 1)[-1]
        for name, value in vars(mod).items():
            if name.startswith("_"):
                continue
            if not callable(value):
                continue
            full = f"{mod_name}.{name}"
            mapping[full] = value
            if name not in seen:
                mapping[name] = value
                seen[name] = full
            else:
                # disambiguate by requiring module prefix
                if name in mapping:
                    del mapping[name]
    return mapping


def _require(mapping: dict[str, Callable], name: str) -> Callable:
    if name not in mapping:
        raise KeyError(f"Unknown function '{name}'.")
    return mapping[name]


@lru_cache(maxsize=None)
def _resolve_point_fn(name: str) -> Callable:
    if name in _POINT_FUNCS:
        return _POINT_FUNCS[name]
    return _require(_public_registry(), name)


@lru_cache(maxsize=None)
def _resolve_interval_fn(name: str) -> Callable:
    if name in _INTERVAL_FUNCS:
        return _INTERVAL_FUNCS[name]
    return _require(_public_registry(), name)


@lru_cache(maxsize=None)
def _optional_kwarg_support(fn: Callable) -> tuple[bool, bool, bool]:
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return False, False, False
    return "mode" in params, "prec_bits" in params, "dps" in params


def _call_with_optional_args(fn: Callable, args: tuple, mode: str, prec_bits: int | None, dps: int | None):
    has_mode, has_prec_bits, has_dps = _optional_kwarg_support(fn)
    if not (has_mode or has_prec_bits or has_dps):
        return fn(*args)
    kwargs = {}
    if has_mode:
        kwargs["mode"] = mode
    if has_prec_bits:
        kwargs["prec_bits"] = prec_bits
    if has_dps:
        kwargs["dps"] = dps
    return fn(*args, **kwargs)


def _is_host_point_backend(name: str) -> bool:
    return False


@lru_cache(maxsize=None)
def _resolve_hypgeom_mode_fn(name: str) -> Callable | None:
    fn_name = name
    if "." in name:
        mod, short = name.split(".", 1)
        if mod != "hypgeom":
            return None
        fn_name = short
    if not (fn_name.startswith("arb_hypgeom_") or fn_name.startswith("acb_hypgeom_")):
        return None
    mode_name = f"{fn_name}_mode"
    fn = getattr(hypgeom_wrappers, mode_name, None)
    return fn if callable(fn) else None


@lru_cache(maxsize=None)
def _bound_interval_fn(name: str, mode: str, prec_bits: int | None, dps: int | None) -> Callable:
    mode_fn = _resolve_hypgeom_mode_fn(name)
    if mode_fn is not None:
        if prec_bits is None and dps is None:
            return lambda *args: mode_fn(*args, impl=mode)
        if prec_bits is None:
            return lambda *args: mode_fn(*args, impl=mode, dps=dps)
        if dps is None:
            return lambda *args: mode_fn(*args, impl=mode, prec_bits=prec_bits)
        return lambda *args: mode_fn(*args, impl=mode, prec_bits=prec_bits, dps=dps)

    fn = _resolve_interval_fn(name)
    has_mode, has_prec_bits, has_dps = _optional_kwarg_support(fn)
    if not (has_mode or has_prec_bits or has_dps):
        return fn
    if has_mode and has_prec_bits and has_dps:
        return lambda *args: fn(*args, mode=mode, prec_bits=prec_bits, dps=dps)
    if has_mode and has_prec_bits:
        return lambda *args: fn(*args, mode=mode, prec_bits=prec_bits)
    if has_mode and has_dps:
        return lambda *args: fn(*args, mode=mode, dps=dps)
    if has_prec_bits and has_dps:
        return lambda *args: fn(*args, prec_bits=prec_bits, dps=dps)
    if has_mode:
        return lambda *args: fn(*args, mode=mode)
    if has_prec_bits:
        return lambda *args: fn(*args, prec_bits=prec_bits)
    if has_dps:
        return lambda *args: fn(*args, dps=dps)
    return fn


def eval_point(name: str, *args: jax.Array, jit: bool = False) -> jax.Array:
    fn = _point_jit_fn(name) if jit else _resolve_point_fn(name)
    return fn(*args)


def eval_point_batch(name: str, *args: jax.Array) -> jax.Array:
    batched = _point_batch_fn(name)
    return batched(*args)


def eval_interval(
    name: str,
    *args: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
    jit: bool = False,
) -> jax.Array:
    fn = _interval_jit_fn(name, mode, prec_bits, dps) if jit else _bound_interval_fn(name, mode, prec_bits, dps)
    return fn(*args)


def eval_interval_batch(
    name: str,
    *args: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    batched = _interval_batch_fn(name, mode, prec_bits, dps)
    return batched(*args)


def _chunked_apply(fn: Callable, args: tuple[jax.Array, ...], chunk_size: int) -> jax.Array:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not args:
        raise ValueError("chunked batch call requires at least one argument array")
    for arr in args:
        if arr.ndim == 0:
            raise ValueError("chunked batch call requires arrays with a leading batch dimension")
    n = int(args[0].shape[0])
    for arr in args[1:]:
        if int(arr.shape[0]) != n:
            raise ValueError("all batch arguments must share the same leading dimension")
    if n <= chunk_size:
        return fn(*args)

    outs = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk_args = tuple(arr[start:stop] for arr in args)
        outs.append(fn(*chunk_args))
    return jnp.concatenate(outs, axis=0)


def eval_point_batch_chunked(name: str, *args: jax.Array, chunk_size: int = 1024) -> jax.Array:
    batched = _point_batch_fn(name)
    return _chunked_apply(batched, args, chunk_size)


def eval_interval_batch_chunked(
    name: str,
    *args: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
    chunk_size: int = 1024,
) -> jax.Array:
    batched = _interval_batch_fn(name, mode, prec_bits, dps)
    return _chunked_apply(batched, args, chunk_size)


def bind_point(name: str) -> Callable:
    return _resolve_point_fn(name)


def bind_interval(
    name: str,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> Callable:
    return _bound_interval_fn(name, mode, prec_bits, dps)


def bind_point_jit(name: str) -> Callable:
    return _point_jit_fn(name)


def bind_interval_jit(
    name: str,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> Callable:
    return _interval_jit_fn(name, mode, prec_bits, dps)


def bind_point_ad(name: str, kind: str = "grad", argnums: int | tuple[int, ...] = 0) -> Callable:
    return _point_ad_fn(name, kind, argnums)


def bind_point_batch(name: str) -> Callable:
    return _point_batch_fn(name)


def bind_interval_batch(
    name: str,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> Callable:
    return _interval_batch_fn(name, mode, prec_bits, dps)


@lru_cache(maxsize=None)
def _point_batch_fn(name: str):
    fn = _resolve_point_fn(name)
    if _is_host_point_backend(name):
        return fn

    @jax.jit
    def _batched(*vals):
        return jax.vmap(fn)(*vals)

    return _batched


@lru_cache(maxsize=None)
def _point_jit_fn(name: str):
    fn = _resolve_point_fn(name)
    if _is_host_point_backend(name):
        return fn
    return jax.jit(fn)


@lru_cache(maxsize=None)
def _interval_jit_fn(name: str, mode: str, prec_bits: int | None, dps: int | None):
    fn = _bound_interval_fn(name, mode, prec_bits, dps)
    if _is_host_point_backend(name):
        return fn
    return jax.jit(fn)


@lru_cache(maxsize=None)
def _interval_batch_fn(name: str, mode: str, prec_bits: int | None, dps: int | None):
    fn = _bound_interval_fn(name, mode, prec_bits, dps)
    if _is_host_point_backend(name):
        return fn

    @jax.jit
    def _batched(*vals):
        return jax.vmap(fn)(*vals)

    return _batched


@lru_cache(maxsize=None)
def _point_ad_fn(name: str, kind: str, argnums: int | tuple[int, ...]):
    fn = _point_jit_fn(name)
    if kind == "grad":
        ad_fn = jax.grad(fn, argnums=argnums)
    elif kind == "jacfwd":
        ad_fn = jax.jacfwd(fn, argnums=argnums)
    elif kind == "jacrev":
        ad_fn = jax.jacrev(fn, argnums=argnums)
    else:
        raise ValueError("kind must be one of: grad, jacfwd, jacrev")
    return jax.jit(ad_fn)


__all__ = [
    "eval_point",
    "eval_point_batch",
    "eval_point_batch_chunked",
    "eval_interval",
    "eval_interval_batch",
    "eval_interval_batch_chunked",
    "bind_point",
    "bind_point_batch",
    "bind_interval",
    "bind_point_jit",
    "bind_interval_jit",
    "bind_point_ad",
    "bind_interval_batch",
    "list_public_functions",
    "list_point_functions",
    "list_interval_functions",
]


def list_point_functions() -> list[str]:
    return list(_POINT_FUNC_NAMES)


def list_interval_functions() -> list[str]:
    return list(_INTERVAL_FUNC_NAMES)


def list_public_functions() -> list[str]:
    return list(_PUBLIC_FUNC_NAMES())


_POINT_FUNC_NAMES = tuple(sorted(_POINT_FUNCS.keys()))
_INTERVAL_FUNC_NAMES = tuple(sorted(_INTERVAL_FUNCS.keys()))


@lru_cache(maxsize=1)
def _PUBLIC_FUNC_NAMES() -> tuple[str, ...]:
    return tuple(sorted(_public_registry().keys()))
