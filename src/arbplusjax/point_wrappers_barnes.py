from __future__ import annotations

import jax

from . import barnesg
from .kernel_helpers import pad_mixed_batch_args_repeat_last, scalarize_unary_complex, trim_batch_out


acb_barnes_g_point = scalarize_unary_complex(barnesg.barnesg_complex)
acb_log_barnes_g_point = scalarize_unary_complex(barnesg.log_barnesg)


@jax.jit
def acb_barnes_g_batch_fixed_point(z):
    return acb_barnes_g_point(z)


def acb_barnes_g_batch_padded_point(z, *, pad_to: int):
    call_args, trim_n = pad_mixed_batch_args_repeat_last((z,), pad_to=pad_to)
    out = acb_barnes_g_batch_fixed_point(*call_args)
    return trim_batch_out(out, trim_n)


@jax.jit
def acb_log_barnes_g_batch_fixed_point(z):
    return acb_log_barnes_g_point(z)


def acb_log_barnes_g_batch_padded_point(z, *, pad_to: int):
    call_args, trim_n = pad_mixed_batch_args_repeat_last((z,), pad_to=pad_to)
    out = acb_log_barnes_g_batch_fixed_point(*call_args)
    return trim_batch_out(out, trim_n)


__all__ = [
    "acb_barnes_g_point",
    "acb_log_barnes_g_point",
    "acb_barnes_g_batch_fixed_point",
    "acb_barnes_g_batch_padded_point",
    "acb_log_barnes_g_batch_fixed_point",
    "acb_log_barnes_g_batch_padded_point",
]
