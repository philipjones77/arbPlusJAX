from __future__ import annotations

import inspect
from typing import Callable

import jax

from . import acb_mat
from . import arb_mat
from . import kernel_helpers as kh
from . import point_wrappers
from . import wrappers_common as wc



def _kernel_name(name: str) -> str:
    if name.endswith("_batch_prec"):
        return name[:-11] + "_batch"
    if name.endswith("_prec"):
        return name[:-5]
    return name


def _point_name(name: str) -> str:
    kernel = _kernel_name(name)
    return f"{kernel}_point"


def _make_wrapper(name: str, base_fn: Callable[..., jax.Array], kernel_fn: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
    is_acb = name.startswith("acb_")
    point_fn = getattr(point_wrappers, _point_name(name), None)
    exact_rigorous_fn = getattr(acb_mat if is_acb else arb_mat, f"{_kernel_name(name)}_rigorous", None)
    uses_plan_surface = (
        "dense_matvec_plan" in name
        or "dense_lu_solve_plan" in name
        or "dense_spd_solve_plan" in name
        or "dense_hpd_solve_plan" in name
    )

    def rig_fn(*args, prec_bits: int, **kwargs):
        if uses_plan_surface:
            return base_fn(*args, prec_bits=prec_bits, **kwargs)
        if callable(exact_rigorous_fn):
            return exact_rigorous_fn(*args, **kwargs)
        if is_acb:
            if name.startswith("acb_mat_det"):
                return acb_mat.acb_mat_det_rigorous(*args, **kwargs)
            if name.startswith("acb_mat_trace"):
                return acb_mat.acb_mat_trace_rigorous(*args, **kwargs)
            if name.startswith("acb_mat_2x2_det"):
                return acb_mat.acb_mat_2x2_det_rigorous(*args, **kwargs)
            if name.startswith("acb_mat_2x2_trace"):
                return acb_mat.acb_mat_2x2_trace_rigorous(*args, **kwargs)
            return wc.rigorous_acb_kernel(kernel_fn, args, prec_bits, **kwargs)
        if name.startswith("arb_mat_det"):
            return arb_mat.arb_mat_det_rigorous(*args, **kwargs)
        if name.startswith("arb_mat_trace"):
            return arb_mat.arb_mat_trace_rigorous(*args, **kwargs)
        if name.startswith("arb_mat_2x2_det"):
            return arb_mat.arb_mat_2x2_det_rigorous(*args, **kwargs)
        if name.startswith("arb_mat_2x2_trace"):
            return arb_mat.arb_mat_2x2_trace_rigorous(*args, **kwargs)
        return wc.rigorous_interval_kernel(kernel_fn, args, prec_bits, **kwargs)

    def adapt_fn(*args, prec_bits: int, **kwargs):
        if uses_plan_surface:
            return base_fn(*args, prec_bits=prec_bits, **kwargs)
        if is_acb:
            return wc.adaptive_acb_kernel(kernel_fn, args, prec_bits, **kwargs)
        return wc.adaptive_interval_kernel(kernel_fn, args, prec_bits, **kwargs)

    def wrapper(*args, impl: str = "basic", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        pb = wc.resolve_prec_bits(dps, prec_bits)
        if uses_plan_surface:
            if impl == "baseline":
                impl = "basic"
            checks = {"point", "basic", "rigorous", "adaptive"} if point_fn is not None else {"basic", "rigorous", "adaptive"}
            if impl not in checks:
                raise ValueError(f"invalid impl={impl!r} for {name}")
            if impl == "point":
                return point_fn(*args, **kwargs)
            return base_fn(*args, prec_bits=pb, **kwargs)
        return wc.dispatch_mode(impl, point_fn, base_fn, rig_fn, adapt_fn, is_acb, pb, args, kwargs)

    wrapper.__name__ = name.replace("_prec", "_mode")
    wrapper.__doc__ = f"Mode-dispatched wrapper around {name}. impl: point|basic|rigorous|adaptive."
    return wrapper


__all__: list[str] = []

for _mod in (acb_mat, arb_mat):
    for _name in dir(_mod):
        if _name.startswith("_"):
            continue
        if not (_name.endswith("_prec") or _name.endswith("_batch_prec")):
            continue
        _fn = getattr(_mod, _name, None)
        if not callable(_fn):
            continue
        try:
            sig = inspect.signature(_fn)
        except (TypeError, ValueError):
            continue
        if "prec_bits" not in sig.parameters:
            continue
        kernel_name = _kernel_name(_name)
        kernel_fn = getattr(_mod, kernel_name, _fn)
        _wrapper = _make_wrapper(_name, _fn, kernel_fn)
        globals()[_wrapper.__name__] = _wrapper
        __all__.append(_wrapper.__name__)


def _make_batch_mode_fixed(name: str) -> Callable[..., jax.Array]:
    fn = globals()[f"{name}_mode"]

    def wrapped(*args, impl: str = "basic", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        return fn(*args, impl=impl, dps=dps, prec_bits=prec_bits, **kwargs)

    wrapped.__name__ = f"{name}_batch_mode_fixed"
    return wrapped


def _make_batch_mode_padded(name: str) -> Callable[..., jax.Array]:
    fn = globals()[f"{name}_mode"]

    def wrapped(*args, pad_to: int, impl: str = "basic", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        call_args, _ = kh.pad_mixed_batch_args_repeat_last(args, pad_to=pad_to)
        return fn(*call_args, impl=impl, dps=dps, prec_bits=prec_bits, **kwargs)

    wrapped.__name__ = f"{name}_batch_mode_padded"
    return wrapped


for _base in (
    "arb_mat_permutation_matrix",
    "arb_mat_transpose",
    "arb_mat_submatrix",
    "arb_mat_diag",
    "arb_mat_diag_matrix",
    "arb_mat_matmul",
    "arb_mat_matvec",
    "arb_mat_banded_matvec",
    "arb_mat_matvec_cached_prepare",
    "arb_mat_matvec_cached_apply",
    "arb_mat_dense_matvec_plan_prepare",
    "arb_mat_dense_matvec_plan_apply",
    "arb_mat_symmetric_part",
    "arb_mat_is_symmetric",
    "arb_mat_is_spd",
    "arb_mat_cho",
    "arb_mat_ldl",
    "arb_mat_dense_spd_solve_plan_prepare",
    "arb_mat_spd_solve",
    "arb_mat_dense_spd_solve_plan_apply",
    "arb_mat_spd_inv",
    "arb_mat_solve",
    "arb_mat_inv",
    "arb_mat_triangular_solve",
    "arb_mat_lu",
    "arb_mat_dense_lu_solve_plan_prepare",
    "arb_mat_lu_solve",
    "arb_mat_dense_lu_solve_plan_apply",
    "arb_mat_qr",
    "arb_mat_det",
    "arb_mat_trace",
    "arb_mat_sqr",
    "arb_mat_norm_fro",
    "arb_mat_norm_1",
    "arb_mat_norm_inf",
    "acb_mat_permutation_matrix",
    "acb_mat_transpose",
    "acb_mat_conjugate_transpose",
    "acb_mat_submatrix",
    "acb_mat_diag",
    "acb_mat_diag_matrix",
    "acb_mat_matmul",
    "acb_mat_matvec",
    "acb_mat_banded_matvec",
    "acb_mat_matvec_cached_prepare",
    "acb_mat_matvec_cached_apply",
    "acb_mat_dense_matvec_plan_prepare",
    "acb_mat_dense_matvec_plan_apply",
    "acb_mat_hermitian_part",
    "acb_mat_is_hermitian",
    "acb_mat_is_hpd",
    "acb_mat_cho",
    "acb_mat_ldl",
    "acb_mat_dense_hpd_solve_plan_prepare",
    "acb_mat_hpd_solve",
    "acb_mat_dense_hpd_solve_plan_apply",
    "acb_mat_hpd_inv",
    "acb_mat_solve",
    "acb_mat_inv",
    "acb_mat_triangular_solve",
    "acb_mat_lu",
    "acb_mat_dense_lu_solve_plan_prepare",
    "acb_mat_lu_solve",
    "acb_mat_dense_lu_solve_plan_apply",
    "acb_mat_qr",
    "acb_mat_det",
    "acb_mat_trace",
    "acb_mat_sqr",
    "acb_mat_norm_fro",
    "acb_mat_norm_1",
    "acb_mat_norm_inf",
):
    _fixed = _make_batch_mode_fixed(_base)
    _padded = _make_batch_mode_padded(_base)
    globals()[_fixed.__name__] = _fixed
    globals()[_padded.__name__] = _padded
    __all__.append(_fixed.__name__)
    __all__.append(_padded.__name__)
