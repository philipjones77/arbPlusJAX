from __future__ import annotations

import inspect
from typing import Callable

import jax

from . import dft
from . import wrappers_common as wc


_ACB_DFT_ALIAS_PRECS = {
    "acb_dft_bluestein_prec",
    "acb_dft_bluestein_precomp_prec",
    "acb_dft_convol_prec",
    "acb_dft_convol_dft_prec",
    "acb_dft_convol_mullow_prec",
    "acb_dft_convol_naive_prec",
    "acb_dft_convol_rad2_prec",
    "acb_dft_convol_rad2_precomp_prec",
    "acb_dft_crt_prec",
    "acb_dft_crt_precomp_prec",
    "acb_dft_cyc_prec",
    "acb_dft_cyc_precomp_prec",
    "acb_dft_inverse_prec",
    "acb_dft_inverse_precomp_prec",
    "acb_dft_inverse_rad2_precomp_prec",
    "acb_dft_inverse_rad2_precomp_inplace_prec",
    "acb_dft_naive_precomp_prec",
    "acb_dft_precomp_prec",
    "acb_dft_prod_precomp_prec",
    "acb_dft_rad2_inplace_prec",
    "acb_dft_rad2_inplace_threaded_prec",
    "acb_dft_rad2_precomp_prec",
    "acb_dft_rad2_precomp_inplace_prec",
    "acb_dft_rad2_precomp_inplace_threaded_prec",
    "acb_dft_step_prec",
}


def _kernel_name(name: str) -> str:
    if name.endswith("_batch_prec"):
        return name[:-11] + "_batch"
    if name.endswith("_prec"):
        return name[:-5]
    return name


def _mode_name(name: str) -> str:
    if name.endswith("_batch_prec"):
        return name[:-11] + "_batch_mode"
    if name.endswith("_prec"):
        return name[:-5] + "_mode"
    return name + "_mode"


def _make_wrapper(name: str, base_fn: Callable[..., jax.Array], kernel_fn: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
    is_acb = name.startswith("acb_")

    def rig_fn(*args, prec_bits: int, **kwargs):
        if name in _ACB_DFT_ALIAS_PRECS:
            return base_fn(*args, prec_bits=prec_bits, **kwargs)
        if name.startswith("acb_dft_bluestein_precomp"):
            return dft.acb_dft_bluestein_precomp_prec(*args, prec_bits=prec_bits, **kwargs)
        if name.startswith("acb_dft_bluestein"):
            return dft.acb_dft_bluestein_prec(*args, prec_bits=prec_bits, **kwargs)
        if name.startswith("acb_dft_convol_rad2"):
            return dft.acb_convol_circular_rad2_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_convol_dft"):
            return dft.acb_convol_circular_dft_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_convol_mullow") or name.startswith("acb_dft_convol_naive"):
            return dft.acb_convol_circular_naive_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_convol"):
            return dft.acb_convol_circular_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_crt") or name.startswith("acb_dft_cyc") or name.startswith("acb_dft_prod_precomp"):
            return dft.acb_dft_prod_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_inverse"):
            return dft.acb_idft_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_naive_precomp"):
            return dft.acb_dft_naive_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_precomp"):
            return dft.acb_dft_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_rad2_inplace") or name.startswith("acb_dft_rad2_precomp"):
            return dft.acb_dft_rad2_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_step"):
            return dft.acb_dft_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_naive"):
            return dft.acb_dft_naive_rigorous(*args, **kwargs)
        if name.startswith("acb_idft_naive"):
            return dft.acb_idft_naive_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_rad2"):
            return dft.acb_dft_rad2_rigorous(*args, **kwargs)
        if name.startswith("acb_idft_rad2"):
            return dft.acb_idft_rad2_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_prod"):
            return dft.acb_dft_prod_rigorous(*args, **kwargs)
        if name.startswith("acb_dft"):
            return dft.acb_dft_rigorous(*args, **kwargs)
        if name.startswith("acb_idft"):
            return dft.acb_idft_rigorous(*args, **kwargs)
        if name.startswith("acb_convol_circular_naive"):
            return dft.acb_convol_circular_naive_rigorous(*args, **kwargs)
        if name.startswith("acb_convol_circular_dft"):
            return dft.acb_convol_circular_dft_rigorous(*args, **kwargs)
        if name.startswith("acb_convol_circular_rad2"):
            return dft.acb_convol_circular_rad2_rigorous(*args, **kwargs)
        if name.startswith("acb_convol_circular"):
            return dft.acb_convol_circular_rigorous(*args, **kwargs)
        if name.startswith("acb_mul_vec"):
            return dft.acb_mul_vec_rigorous(*args, **kwargs)
        return wc.rigorous_acb_kernel(kernel_fn, args, prec_bits, **kwargs)

    def adapt_fn(*args, prec_bits: int, **kwargs):
        if name in _ACB_DFT_ALIAS_PRECS:
            return base_fn(*args, prec_bits=prec_bits, **kwargs)
        if name.startswith("acb_dft_bluestein_precomp"):
            return dft.acb_dft_bluestein_precomp_prec(*args, prec_bits=prec_bits, **kwargs)
        if name.startswith("acb_dft_bluestein"):
            return dft.acb_dft_bluestein_prec(*args, prec_bits=prec_bits, **kwargs)
        if name.startswith("acb_dft_convol_rad2"):
            return dft.acb_convol_circular_rad2_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_convol_dft"):
            return dft.acb_convol_circular_dft_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_convol_mullow") or name.startswith("acb_dft_convol_naive"):
            return dft.acb_convol_circular_naive_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_convol"):
            return dft.acb_convol_circular_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_crt") or name.startswith("acb_dft_cyc") or name.startswith("acb_dft_prod_precomp"):
            return dft.acb_dft_prod_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_inverse"):
            return dft.acb_idft_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_naive_precomp"):
            return dft.acb_dft_naive_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_precomp"):
            return dft.acb_dft_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_rad2_inplace") or name.startswith("acb_dft_rad2_precomp"):
            return dft.acb_dft_rad2_rigorous(*args, **kwargs)
        if name.startswith("acb_dft_step"):
            return dft.acb_dft_rigorous(*args, **kwargs)
        return wc.adaptive_acb_kernel(kernel_fn, args, prec_bits, **kwargs)

    def wrapper(*args, impl: str = "basic", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        pb = wc.resolve_prec_bits(dps, prec_bits)
        return wc.dispatch_mode(impl, None, base_fn, rig_fn, adapt_fn, is_acb, pb, args, kwargs)

    wrapper.__name__ = _mode_name(name)
    wrapper.__doc__ = f"Mode-dispatched wrapper around {name}. impl: basic|rigorous|adaptive."
    return wrapper


__all__: list[str] = []

for _name in dir(dft):
    if _name.startswith("_"):
        continue
    if not (_name.endswith("_prec") or _name.endswith("_batch_prec")):
        continue
    _fn = getattr(dft, _name, None)
    if not callable(_fn):
        continue
    try:
        sig = inspect.signature(_fn)
    except (TypeError, ValueError):
        continue
    if "prec_bits" not in sig.parameters:
        continue
    kernel_name = _kernel_name(_name)
    kernel_fn = getattr(dft, kernel_name, _fn)
    _wrapper = _make_wrapper(_name, _fn, kernel_fn)
    globals()[_wrapper.__name__] = _wrapper
    __all__.append(_wrapper.__name__)

