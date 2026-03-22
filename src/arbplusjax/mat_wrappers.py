from __future__ import annotations

import inspect
from typing import Callable

import jax

from . import acb_mat
from . import arb_mat
from . import kernel_helpers as kh
from . import mat_common
from . import point_wrappers
from . import scb_mat
from . import srb_mat
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


def _make_sparse_wrapper(name: str, base_fn: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
    mod = srb_mat if name.startswith("srb_") else scb_mat
    basic_fn = getattr(mod, f"{name}_basic", None)

    def wrapper(*args, impl: str = "basic", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        del dps, prec_bits
        if impl == "baseline":
            impl = "basic"
        if impl not in {"point", "basic", "rigorous", "adaptive"}:
            raise ValueError(f"invalid impl={impl!r} for {name}")
        if impl == "point":
            return base_fn(*args, **kwargs)
        if callable(basic_fn):
            return basic_fn(*args, **kwargs)
        return base_fn(*args, **kwargs)

    wrapper.__name__ = f"{name}_mode"
    wrapper.__doc__ = (
        f"Mode-dispatched wrapper around {name}. Sparse basic/rigorous/adaptive currently reuse sparse basic "
        "implementations when present, otherwise the point implementation."
    )
    return wrapper


def _make_sparse_batch_wrapper(name: str, batch_fn: Callable[..., jax.Array], *, padded: bool) -> Callable[..., jax.Array]:
    mod = srb_mat if name.startswith("srb_") else scb_mat
    basic_batch_name = f"{name}_basic_batch_padded" if padded else f"{name}_basic_batch_fixed"
    basic_batch_fn = getattr(mod, basic_batch_name, None)

    def wrapped(*args, impl: str = "basic", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        del dps, prec_bits
        if impl == "baseline":
            impl = "basic"
        if impl not in {"point", "basic", "rigorous", "adaptive"}:
            raise ValueError(f"invalid impl={impl!r} for {name}")
        if impl == "point":
            return batch_fn(*args, **kwargs)
        if callable(basic_batch_fn):
            return basic_batch_fn(*args, **kwargs)
        return batch_fn(*args, **kwargs)

    suffix = "batch_mode_padded" if padded else "batch_mode_fixed"
    wrapped.__name__ = f"{name}_{suffix}"
    wrapped.__doc__ = f"Mode-dispatched sparse batch wrapper around {batch_fn.__name__}."
    return wrapped


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


SPARSE_MODE_BASES = (
    "srb_mat_shape",
    "srb_mat_nnz",
    "srb_mat_zero",
    "srb_mat_identity",
    "srb_mat_permutation_matrix",
    "srb_mat_diag",
    "srb_mat_diag_matrix",
    "srb_mat_trace",
    "srb_mat_norm_fro",
    "srb_mat_norm_1",
    "srb_mat_norm_inf",
    "srb_mat_submatrix",
    "srb_mat_to_dense",
    "srb_mat_transpose",
    "srb_mat_symmetric_part",
    "srb_mat_is_symmetric",
    "srb_mat_is_spd",
    "srb_mat_scale",
    "srb_mat_add",
    "srb_mat_sub",
    "srb_mat_matvec",
    "srb_mat_rmatvec",
    "srb_mat_matvec_cached_prepare",
    "srb_mat_matvec_cached_apply",
    "srb_mat_rmatvec_cached_prepare",
    "srb_mat_rmatvec_cached_apply",
    "srb_mat_matmul_dense_rhs",
    "srb_mat_matmul_sparse",
    "srb_mat_triangular_solve",
    "srb_mat_cho",
    "srb_mat_ldl",
    "srb_mat_charpoly",
    "srb_mat_pow_ui",
    "srb_mat_exp",
    "srb_mat_eigvalsh",
    "srb_mat_eigh",
    "srb_mat_eigsh",
    "srb_mat_lu",
    "srb_mat_lu_solve",
    "srb_mat_qr",
    "srb_mat_qr_r",
    "srb_mat_qr_apply_q",
    "srb_mat_qr_explicit_q",
    "srb_mat_qr_solve",
    "srb_mat_solve",
    "srb_mat_det",
    "srb_mat_inv",
    "srb_mat_sqr",
    "srb_mat_lu_solve_plan_prepare",
    "srb_mat_lu_solve_plan_apply",
    "srb_mat_spd_solve_plan_prepare",
    "srb_mat_spd_solve_plan_apply",
    "srb_mat_spd_solve",
    "srb_mat_spd_inv",
    "srb_mat_solve_lu",
    "srb_mat_solve_lu_precomp",
    "srb_mat_solve_transpose",
    "srb_mat_solve_add",
    "srb_mat_solve_transpose_add",
    "srb_mat_mat_solve",
    "srb_mat_mat_solve_transpose",
    "scb_mat_shape",
    "scb_mat_nnz",
    "scb_mat_zero",
    "scb_mat_identity",
    "scb_mat_permutation_matrix",
    "scb_mat_diag",
    "scb_mat_diag_matrix",
    "scb_mat_trace",
    "scb_mat_norm_fro",
    "scb_mat_norm_1",
    "scb_mat_norm_inf",
    "scb_mat_submatrix",
    "scb_mat_to_dense",
    "scb_mat_transpose",
    "scb_mat_conjugate_transpose",
    "scb_mat_hermitian_part",
    "scb_mat_is_hermitian",
    "scb_mat_is_hpd",
    "scb_mat_scale",
    "scb_mat_add",
    "scb_mat_sub",
    "scb_mat_matvec",
    "scb_mat_rmatvec",
    "scb_mat_matvec_cached_prepare",
    "scb_mat_matvec_cached_apply",
    "scb_mat_rmatvec_cached_prepare",
    "scb_mat_rmatvec_cached_apply",
    "scb_mat_matmul_dense_rhs",
    "scb_mat_matmul_sparse",
    "scb_mat_triangular_solve",
    "scb_mat_cho",
    "scb_mat_ldl",
    "scb_mat_charpoly",
    "scb_mat_pow_ui",
    "scb_mat_exp",
    "scb_mat_eigvalsh",
    "scb_mat_eigh",
    "scb_mat_eigsh",
    "scb_mat_lu",
    "scb_mat_lu_solve",
    "scb_mat_qr",
    "scb_mat_qr_r",
    "scb_mat_qr_apply_q",
    "scb_mat_qr_explicit_q",
    "scb_mat_qr_solve",
    "scb_mat_solve",
    "scb_mat_det",
    "scb_mat_inv",
    "scb_mat_sqr",
    "scb_mat_lu_solve_plan_prepare",
    "scb_mat_lu_solve_plan_apply",
    "scb_mat_hpd_solve_plan_prepare",
    "scb_mat_hpd_solve_plan_apply",
    "scb_mat_hpd_solve",
    "scb_mat_hpd_inv",
    "scb_mat_solve_lu",
    "scb_mat_solve_lu_precomp",
    "scb_mat_solve_transpose",
    "scb_mat_solve_add",
    "scb_mat_solve_transpose_add",
    "scb_mat_mat_solve",
    "scb_mat_mat_solve_transpose",
)


SPARSE_BATCH_MODE_BASES = (
    "srb_mat_matvec",
    "srb_mat_rmatvec",
    "srb_mat_matvec_cached_apply",
    "srb_mat_rmatvec_cached_apply",
    "srb_mat_solve",
    "srb_mat_triangular_solve",
    "srb_mat_lu_solve_plan_apply",
    "srb_mat_spd_solve_plan_apply",
    "scb_mat_matvec",
    "scb_mat_rmatvec",
    "scb_mat_matvec_cached_apply",
    "scb_mat_rmatvec_cached_apply",
    "scb_mat_solve",
    "scb_mat_triangular_solve",
    "scb_mat_lu_solve_plan_apply",
    "scb_mat_hpd_solve_plan_apply",
)


for _name in SPARSE_MODE_BASES:
    _mod = srb_mat if _name.startswith("srb_") else scb_mat
    _fn = getattr(_mod, _name, None)
    if callable(_fn):
        _wrapper = _make_sparse_wrapper(_name, _fn)
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

    def _pad_arg(arg, *, pad_to: int):
        if mat_common.is_dense_plan_like(arg):
            return arg
        if not mat_common.is_batch_pad_candidate(arg):
            return arg
        (arg_pad,), _ = kh.pad_mixed_batch_args_repeat_last((arg,), pad_to=pad_to)
        return arg_pad

    def wrapped(*args, pad_to: int, impl: str = "basic", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        call_args = tuple(_pad_arg(arg, pad_to=pad_to) for arg in args)
        return fn(*call_args, impl=impl, dps=dps, prec_bits=prec_bits, **kwargs)

    wrapped.__name__ = f"{name}_batch_mode_padded"
    return wrapped


for _base in (
    "arb_mat_permutation_matrix",
    "arb_mat_transpose",
    "arb_mat_add",
    "arb_mat_sub",
    "arb_mat_neg",
    "arb_mat_submatrix",
    "arb_mat_mul_entrywise",
    "arb_mat_diag",
    "arb_mat_diag_matrix",
    "arb_mat_matmul",
    "arb_mat_matvec",
    "arb_mat_rmatvec",
    "arb_mat_banded_matvec",
    "arb_mat_matvec_cached_prepare",
    "arb_mat_matvec_cached_apply",
    "arb_mat_rmatvec_cached_prepare",
    "arb_mat_rmatvec_cached_apply",
    "arb_mat_dense_matvec_plan_prepare",
    "arb_mat_dense_matvec_plan_apply",
    "arb_mat_symmetric_part",
    "arb_mat_is_symmetric",
    "arb_mat_is_spd",
    "arb_mat_is_diag",
    "arb_mat_is_tril",
    "arb_mat_is_triu",
    "arb_mat_is_zero",
    "arb_mat_is_finite",
    "arb_mat_is_exact",
    "arb_mat_charpoly",
    "arb_mat_pow_ui",
    "arb_mat_exp",
    "arb_mat_cho",
    "arb_mat_ldl",
    "arb_mat_eigvalsh",
    "arb_mat_eigh",
    "arb_mat_dense_spd_solve_plan_prepare",
    "arb_mat_spd_solve",
    "arb_mat_dense_spd_solve_plan_apply",
    "arb_mat_spd_inv",
    "arb_mat_solve",
    "arb_mat_inv",
    "arb_mat_triangular_solve",
    "arb_mat_solve_tril",
    "arb_mat_solve_triu",
    "arb_mat_lu",
    "arb_mat_dense_lu_solve_plan_prepare",
    "arb_mat_lu_solve",
    "arb_mat_solve_lu",
    "arb_mat_solve_transpose",
    "arb_mat_solve_add",
    "arb_mat_solve_transpose_add",
    "arb_mat_mat_solve",
    "arb_mat_mat_solve_transpose",
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
    "acb_mat_add",
    "acb_mat_sub",
    "acb_mat_neg",
    "acb_mat_mul_entrywise",
    "acb_mat_conjugate",
    "acb_mat_submatrix",
    "acb_mat_diag",
    "acb_mat_diag_matrix",
    "acb_mat_matmul",
    "acb_mat_matvec",
    "acb_mat_rmatvec",
    "acb_mat_banded_matvec",
    "acb_mat_matvec_cached_prepare",
    "acb_mat_matvec_cached_apply",
    "acb_mat_rmatvec_cached_prepare",
    "acb_mat_rmatvec_cached_apply",
    "acb_mat_dense_matvec_plan_prepare",
    "acb_mat_dense_matvec_plan_apply",
    "acb_mat_hermitian_part",
    "acb_mat_is_hermitian",
    "acb_mat_is_hpd",
    "acb_mat_is_diag",
    "acb_mat_is_tril",
    "acb_mat_is_triu",
    "acb_mat_is_zero",
    "acb_mat_is_finite",
    "acb_mat_is_exact",
    "acb_mat_is_real",
    "acb_mat_charpoly",
    "acb_mat_pow_ui",
    "acb_mat_exp",
    "acb_mat_cho",
    "acb_mat_ldl",
    "acb_mat_eigvalsh",
    "acb_mat_eigh",
    "acb_mat_dense_hpd_solve_plan_prepare",
    "acb_mat_hpd_solve",
    "acb_mat_dense_hpd_solve_plan_apply",
    "acb_mat_hpd_inv",
    "acb_mat_solve",
    "acb_mat_inv",
    "acb_mat_triangular_solve",
    "acb_mat_solve_tril",
    "acb_mat_solve_triu",
    "acb_mat_lu",
    "acb_mat_dense_lu_solve_plan_prepare",
    "acb_mat_lu_solve",
    "acb_mat_solve_lu",
    "acb_mat_solve_transpose",
    "acb_mat_solve_add",
    "acb_mat_solve_transpose_add",
    "acb_mat_mat_solve",
    "acb_mat_mat_solve_transpose",
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


for _base in SPARSE_BATCH_MODE_BASES:
    _mod = srb_mat if _base.startswith("srb_") else scb_mat
    _fixed = _make_sparse_batch_wrapper(_base, getattr(_mod, f"{_base}_batch_fixed"), padded=False)
    _padded = _make_sparse_batch_wrapper(_base, getattr(_mod, f"{_base}_batch_padded"), padded=True)
    globals()[_fixed.__name__] = _fixed
    globals()[_padded.__name__] = _padded
    __all__.append(_fixed.__name__)
    __all__.append(_padded.__name__)
