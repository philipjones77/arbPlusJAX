from __future__ import annotations

import inspect
from typing import Callable

import jax

from . import dft
from . import wrappers_common as wc

jax.config.update("jax_enable_x64", True)


def _kernel_name(name: str) -> str:
    if name.endswith("_batch_prec"):
        return name[:-11] + "_batch"
    if name.endswith("_prec"):
        return name[:-5]
    return name


def _make_wrapper(name: str, base_fn: Callable[..., jax.Array], kernel_fn: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
    is_acb = name.startswith("acb_")

    def rig_fn(*args, prec_bits: int, **kwargs):
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
        return wc.adaptive_acb_kernel(kernel_fn, args, prec_bits, **kwargs)

    def wrapper(*args, impl: str = "baseline", dps: int | None = None, prec_bits: int | None = None, **kwargs):
        pb = wc.resolve_prec_bits(dps, prec_bits)
        return wc.dispatch_mode(impl, base_fn, rig_fn, adapt_fn, is_acb, pb, args, kwargs)

    wrapper.__name__ = name.replace("_prec", "_mode")
    wrapper.__doc__ = f"Mode-dispatched wrapper around {name}. impl: baseline|rigorous|adaptive."
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
