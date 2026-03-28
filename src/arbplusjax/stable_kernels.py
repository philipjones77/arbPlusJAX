from __future__ import annotations

from collections.abc import Callable
import time

import jax
import jax.numpy as jnp

from . import acb_core
from . import api
from . import double_gamma
from . import double_interval as di
from . import precision
from . import wrappers_common as wc
from .capability_registry import DOWNSTREAM_KERNELS, lookup_capability


_MODE_SET = ("point", "basic", "adaptive", "rigorous")


def _check_mode(mode: str) -> str:
    if mode not in _MODE_SET:
        raise ValueError(f"unsupported mode {mode!r}; expected one of {_MODE_SET}")
    return mode


def _eval_unary(name: str, x: jax.Array, *, mode: str, dtype: str | jnp.dtype | None, **kwargs) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return api.eval_point(name, x, dtype=dtype, **kwargs)
    return api.eval_interval(name, x, mode=mode, dtype=dtype, **kwargs)


def _cast_dtype(arg: jax.Array, dtype: str | jnp.dtype | None) -> jax.Array:
    arr = jnp.asarray(arg)
    if dtype is None:
        return arr
    return arr.astype(jnp.dtype(dtype))


def _loggamma_name(x: jax.Array) -> str:
    dt = jnp.asarray(x).dtype
    return "acb_lgamma" if jnp.issubdtype(dt, jnp.complexfloating) else "arb_lgamma"


def _gamma_name(x: jax.Array) -> str:
    dt = jnp.asarray(x).dtype
    return "acb_gamma" if jnp.issubdtype(dt, jnp.complexfloating) else "arb_gamma"


def _box_scalar(z: jax.Array) -> jax.Array:
    zz = jnp.asarray(z)
    return acb_core.acb_box(di.interval(jnp.real(zz), jnp.real(zz)), di.interval(jnp.imag(zz), jnp.imag(zz)))


def _bound_input(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if jnp.issubdtype(xx.dtype, jnp.complexfloating):
        return _box_scalar(xx)
    return di.interval(xx, xx)


def _provider_prec_bits(kwargs: dict[str, object]) -> tuple[int, dict[str, object]]:
    local = dict(kwargs)
    dps = local.pop("dps", None)
    prec_bits = local.pop("prec_bits", None)
    pb = precision.get_prec_bits() if dps is None and prec_bits is None else wc.resolve_prec_bits(dps, prec_bits)
    local.pop("max_m_cap", None)
    return pb, local


def _eval_unary_batch(
    name: str,
    x: jax.Array,
    *,
    mode: str,
    dtype: str | jnp.dtype | None,
    pad_to: int | None,
    **kwargs,
) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return api.eval_point_batch(name, x, dtype=dtype, pad_to=pad_to, **kwargs)
    return api.eval_interval_batch(name, x, mode=mode, dtype=dtype, pad_to=pad_to, **kwargs)


def gamma(x: jax.Array, *, mode: str = "point", dtype: str | jnp.dtype | None = None, **kwargs) -> jax.Array:
    cast = _cast_dtype(x, dtype)
    if _check_mode(mode) == "point":
        return api.eval_point("gamma", cast, **kwargs)
    return api.eval_interval(_gamma_name(cast), _bound_input(cast), mode=mode, **kwargs)


def gamma_batch(
    x: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    return _eval_unary_batch("gamma", x, mode=mode, dtype=dtype, pad_to=pad_to, **kwargs)


def loggamma(x: jax.Array, *, mode: str = "point", dtype: str | jnp.dtype | None = None, **kwargs) -> jax.Array:
    cast = _cast_dtype(x, dtype)
    if _check_mode(mode) == "point":
        return _eval_unary(_loggamma_name(cast), cast, mode=mode, dtype=None, **kwargs)
    return api.eval_interval(_loggamma_name(cast), _bound_input(cast), mode=mode, **kwargs)


def loggamma_batch(
    x: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    cast = _cast_dtype(x, dtype)
    return _eval_unary_batch(_loggamma_name(cast), cast, mode=mode, dtype=None, pad_to=pad_to, **kwargs)


def incomplete_gamma_lower(
    s: jax.Array,
    z: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_gamma_lower(_cast_dtype(s, dtype), _cast_dtype(z, dtype), mode=_check_mode(mode), **kwargs)


def incomplete_gamma_lower_batch(
    s: jax.Array,
    z: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    s_cast = _cast_dtype(s, dtype)
    z_cast = _cast_dtype(z, dtype)
    mode = _check_mode(mode)
    if pad_to is None:
        return api.incomplete_gamma_lower_batch(s_cast, z_cast, mode=mode, **kwargs)
    if mode == "point":
        return api.bind_point_batch("incomplete_gamma_lower", dtype=dtype, pad_to=pad_to, **kwargs)(s_cast, z_cast)
    return api.bind_interval_batch("incomplete_gamma_lower", mode=mode, dtype=dtype, pad_to=pad_to, **kwargs)(s_cast, z_cast)


def incomplete_gamma_upper(
    s: jax.Array,
    z: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_gamma_upper(_cast_dtype(s, dtype), _cast_dtype(z, dtype), mode=_check_mode(mode), **kwargs)


def incomplete_gamma_upper_batch(
    s: jax.Array,
    z: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    s_cast = _cast_dtype(s, dtype)
    z_cast = _cast_dtype(z, dtype)
    mode = _check_mode(mode)
    if pad_to is None:
        return api.incomplete_gamma_upper_batch(s_cast, z_cast, mode=mode, **kwargs)
    if mode == "point":
        return api.bind_point_batch("incomplete_gamma_upper", dtype=dtype, pad_to=pad_to, **kwargs)(s_cast, z_cast)
    return api.bind_interval_batch("incomplete_gamma_upper", mode=mode, dtype=dtype, pad_to=pad_to, **kwargs)(s_cast, z_cast)


def incomplete_bessel_i(
    nu: jax.Array,
    z: jax.Array,
    upper_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_bessel_i(
        _cast_dtype(nu, dtype),
        _cast_dtype(z, dtype),
        _cast_dtype(upper_limit, dtype),
        mode=_check_mode(mode),
        **kwargs,
    )


def incomplete_bessel_i_batch(
    nu: jax.Array,
    z: jax.Array,
    upper_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    nu_cast = _cast_dtype(nu, dtype)
    z_cast = _cast_dtype(z, dtype)
    upper_cast = _cast_dtype(upper_limit, dtype)
    mode = _check_mode(mode)
    if pad_to is None:
        return api.incomplete_bessel_i_batch(nu_cast, z_cast, upper_cast, mode=mode, **kwargs)
    if mode == "point":
        return api.bind_point_batch("incomplete_bessel_i", dtype=dtype, pad_to=pad_to, **kwargs)(nu_cast, z_cast, upper_cast)
    return api.bind_interval_batch("incomplete_bessel_i", mode=mode, dtype=dtype, pad_to=pad_to, **kwargs)(nu_cast, z_cast, upper_cast)


def incomplete_bessel_k(
    nu: jax.Array,
    z: jax.Array,
    lower_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_bessel_k(
        _cast_dtype(nu, dtype),
        _cast_dtype(z, dtype),
        _cast_dtype(lower_limit, dtype),
        mode=_check_mode(mode),
        **kwargs,
    )


def incomplete_bessel_k_batch(
    nu: jax.Array,
    z: jax.Array,
    lower_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    nu_cast = _cast_dtype(nu, dtype)
    z_cast = _cast_dtype(z, dtype)
    lower_cast = _cast_dtype(lower_limit, dtype)
    mode = _check_mode(mode)
    if pad_to is None:
        return api.incomplete_bessel_k_batch(nu_cast, z_cast, lower_cast, mode=mode, **kwargs)
    if mode == "point":
        return api.bind_point_batch("incomplete_bessel_k", dtype=dtype, pad_to=pad_to, **kwargs)(nu_cast, z_cast, lower_cast)
    return api.bind_interval_batch("incomplete_bessel_k", mode=mode, dtype=dtype, pad_to=pad_to, **kwargs)(nu_cast, z_cast, lower_cast)


def provider_incomplete_bessel_i(
    nu: jax.Array,
    z: jax.Array,
    upper_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return incomplete_bessel_i(nu, z, upper_limit, mode=mode, dtype=dtype, **kwargs)


def provider_incomplete_bessel_k(
    nu: jax.Array,
    z: jax.Array,
    lower_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return incomplete_bessel_k(nu, z, lower_limit, mode=mode, dtype=dtype, **kwargs)


def barnesdoublegamma(
    z: jax.Array,
    tau: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    z_cast = _cast_dtype(z, dtype)
    tau_cast = _cast_dtype(tau, dtype)
    if jnp.asarray(z_cast).ndim == 0 and jnp.asarray(tau_cast).ndim == 0:
        return double_gamma.ifj_barnesdoublegamma(z_cast, tau_cast, **kwargs)
    return double_gamma.ifj_barnesdoublegamma_batch_fixed_point(z_cast, tau_cast, **kwargs)


def provider_barnesdoublegamma(
    z: jax.Array,
    tau: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    z_cast = _cast_dtype(z, dtype)
    tau_cast = _cast_dtype(tau, dtype)
    prec_bits, local_kwargs = _provider_prec_bits(kwargs)
    if jnp.asarray(z_cast).ndim == 0 and jnp.asarray(tau_cast).ndim == 0:
        return double_gamma.bdg_barnesdoublegamma(z_cast, tau_cast, prec_bits=prec_bits, **local_kwargs)
    return double_gamma.bdg_barnesdoublegamma_batch_fixed_point(z_cast, tau_cast, prec_bits=prec_bits, **local_kwargs)


def barnesdoublegamma_batch(
    z: jax.Array,
    tau: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    z_cast = _cast_dtype(z, dtype)
    tau_cast = _cast_dtype(tau, dtype)
    if pad_to is None:
        return double_gamma.ifj_barnesdoublegamma_batch_fixed_point(z_cast, tau_cast, **kwargs)
    return double_gamma.ifj_barnesdoublegamma_batch_padded_point(z_cast, tau_cast, pad_to=pad_to, **kwargs)


def provider_barnesdoublegamma_batch(
    z: jax.Array,
    tau: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    z_cast = _cast_dtype(z, dtype)
    tau_cast = _cast_dtype(tau, dtype)
    prec_bits, local_kwargs = _provider_prec_bits(kwargs)
    if pad_to is None:
        return double_gamma.bdg_barnesdoublegamma_batch_fixed_point(z_cast, tau_cast, prec_bits=prec_bits, **local_kwargs)
    return double_gamma.bdg_barnesdoublegamma_batch_padded_point(
        z_cast,
        tau_cast,
        pad_to=pad_to,
        prec_bits=prec_bits,
        **local_kwargs,
    )


def log_barnesdoublegamma(
    z: jax.Array,
    tau: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    z_cast = _cast_dtype(z, dtype)
    tau_cast = _cast_dtype(tau, dtype)
    if jnp.asarray(z_cast).ndim == 0 and jnp.asarray(tau_cast).ndim == 0:
        return double_gamma.ifj_log_barnesdoublegamma(z_cast, tau_cast, **kwargs)
    return double_gamma.ifj_log_barnesdoublegamma_batch_fixed_point(z_cast, tau_cast, **kwargs)


def provider_log_barnesdoublegamma(
    z: jax.Array,
    tau: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    z_cast = _cast_dtype(z, dtype)
    tau_cast = _cast_dtype(tau, dtype)
    prec_bits, local_kwargs = _provider_prec_bits(kwargs)
    if jnp.asarray(z_cast).ndim == 0 and jnp.asarray(tau_cast).ndim == 0:
        return double_gamma.bdg_log_barnesdoublegamma(z_cast, tau_cast, prec_bits=prec_bits, **local_kwargs)
    return double_gamma.bdg_log_barnesdoublegamma_batch_fixed_point(z_cast, tau_cast, prec_bits=prec_bits, **local_kwargs)


def log_barnesdoublegamma_batch(
    z: jax.Array,
    tau: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    z_cast = _cast_dtype(z, dtype)
    tau_cast = _cast_dtype(tau, dtype)
    if pad_to is None:
        return double_gamma.ifj_log_barnesdoublegamma_batch_fixed_point(z_cast, tau_cast, **kwargs)
    return double_gamma.ifj_log_barnesdoublegamma_batch_padded_point(z_cast, tau_cast, pad_to=pad_to, **kwargs)


def provider_log_barnesdoublegamma_batch(
    z: jax.Array,
    tau: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    z_cast = _cast_dtype(z, dtype)
    tau_cast = _cast_dtype(tau, dtype)
    prec_bits, local_kwargs = _provider_prec_bits(kwargs)
    if pad_to is None:
        return double_gamma.bdg_log_barnesdoublegamma_batch_fixed_point(z_cast, tau_cast, prec_bits=prec_bits, **local_kwargs)
    return double_gamma.bdg_log_barnesdoublegamma_batch_padded_point(
        z_cast,
        tau_cast,
        pad_to=pad_to,
        prec_bits=prec_bits,
        **local_kwargs,
    )


def prepare_arb_dense_spd_solve_service(
    a: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
) -> Callable[[jax.Array], jax.Array]:
    a_cast = _cast_dtype(a, dtype)
    plan = api.bind_point_batch("arb_mat_dense_spd_solve_plan_prepare", dtype=dtype, pad_to=pad_to)(a_cast)
    apply_fn = api.bind_point_batch("arb_mat_dense_spd_solve_plan_apply", dtype=dtype, pad_to=pad_to)

    def solve(b: jax.Array) -> jax.Array:
        return apply_fn(plan, _cast_dtype(b, dtype))

    return solve


def prepare_acb_dense_matvec_service(
    a: jax.Array,
    *,
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
) -> Callable[[jax.Array], jax.Array]:
    a_cast = _cast_dtype(a, dtype)
    plan = api.bind_point_batch("acb_mat_dense_matvec_plan_prepare", dtype=dtype, pad_to=pad_to)(a_cast)
    apply_fn = api.bind_point_batch("acb_mat_dense_matvec_plan_apply", dtype=dtype, pad_to=pad_to)

    def matvec(x: jax.Array) -> jax.Array:
        return apply_fn(plan, _cast_dtype(x, dtype))

    return matvec


def warm_special_function_point_kernels(
    *,
    dtype: str | jnp.dtype = "float64",
    pad_to: int = 32,
) -> dict[str, float]:
    dt = jnp.dtype(dtype)
    real_dt = jnp.float32 if dt == jnp.float32 else jnp.float64
    complex_dt = jnp.complex64 if real_dt == jnp.float32 else jnp.complex128

    def _timed(name: str, fn: Callable[[], jax.Array]) -> tuple[str, float]:
        started = time.perf_counter()
        jax.block_until_ready(fn())
        return name, time.perf_counter() - started

    real_count = max(4, min(int(pad_to), 32))
    complex_count = max(4, min(int(pad_to), 16))
    s = jnp.linspace(jnp.asarray(1.1, dtype=real_dt), jnp.asarray(2.5, dtype=real_dt), real_count)
    z = jnp.linspace(jnp.asarray(0.2, dtype=real_dt), jnp.asarray(1.6, dtype=real_dt), real_count)
    nu = jnp.linspace(jnp.asarray(0.3, dtype=real_dt), jnp.asarray(1.1, dtype=real_dt), real_count)
    upper = jnp.linspace(jnp.asarray(0.2, dtype=real_dt), jnp.asarray(0.8, dtype=real_dt), real_count)
    lower = jnp.linspace(jnp.asarray(0.1, dtype=real_dt), jnp.asarray(0.5, dtype=real_dt), real_count)
    a = jnp.linspace(jnp.asarray(1.0, dtype=real_dt), jnp.asarray(1.6, dtype=real_dt), real_count)
    b = jnp.linspace(jnp.asarray(2.0, dtype=real_dt), jnp.asarray(2.6, dtype=real_dt), real_count)
    z_hyp = jnp.linspace(jnp.asarray(0.1, dtype=real_dt), jnp.asarray(0.6, dtype=real_dt), real_count)
    z_barnes = jnp.linspace(jnp.asarray(1.1, dtype=real_dt), jnp.asarray(1.6, dtype=real_dt), complex_count).astype(complex_dt) + jnp.asarray(0.05j, dtype=complex_dt)
    tau = jnp.full((complex_count,), jnp.asarray(0.5, dtype=real_dt))

    warmers = (
        _timed(
            "incomplete_gamma_upper",
            lambda: api.bind_point_batch("incomplete_gamma_upper", dtype=real_dt, pad_to=pad_to, regularized=True, method="quadrature")(s, z),
        ),
        _timed(
            "incomplete_bessel_k",
            lambda: api.bind_point_batch("incomplete_bessel_k", dtype=real_dt, pad_to=pad_to, method="quadrature")(nu, z, lower),
        ),
        _timed(
            "arb_hypgeom_1f1",
            lambda: api.bind_point_batch("arb_hypgeom_1f1", dtype=real_dt, pad_to=pad_to)(a, b, z_hyp),
        ),
        _timed(
            "arb_hypgeom_u",
            lambda: api.bind_point_batch("arb_hypgeom_u", dtype=real_dt, pad_to=pad_to)(a, b, z_hyp),
        ),
        _timed(
            "ifj_log_barnesdoublegamma",
            lambda: api.bind_point_batch("ifj_log_barnesdoublegamma", dtype=real_dt, pad_to=pad_to, dps=50, max_m_cap=256)(z_barnes, tau),
        ),
    )
    return dict(warmers)


def list_supported_kernels() -> tuple[str, ...]:
    return tuple(sorted(DOWNSTREAM_KERNELS))


def get_kernel_capability(name: str) -> dict[str, object]:
    return lookup_capability(name)


__all__ = [
    "gamma",
    "gamma_batch",
    "loggamma",
    "loggamma_batch",
    "incomplete_gamma_lower",
    "incomplete_gamma_lower_batch",
    "incomplete_gamma_upper",
    "incomplete_gamma_upper_batch",
    "incomplete_bessel_i",
    "incomplete_bessel_i_batch",
    "incomplete_bessel_k",
    "incomplete_bessel_k_batch",
    "provider_incomplete_bessel_i",
    "provider_incomplete_bessel_k",
    "provider_barnesdoublegamma",
    "provider_barnesdoublegamma_batch",
    "provider_log_barnesdoublegamma",
    "provider_log_barnesdoublegamma_batch",
    "barnesdoublegamma",
    "barnesdoublegamma_batch",
    "log_barnesdoublegamma",
    "log_barnesdoublegamma_batch",
    "prepare_arb_dense_spd_solve_service",
    "prepare_acb_dense_matvec_service",
    "warm_special_function_point_kernels",
    "list_supported_kernels",
    "get_kernel_capability",
]
