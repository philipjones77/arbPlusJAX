from __future__ import annotations

import jax
import jax.numpy as jnp


_HAS_DEBUG_CHECK = hasattr(jax.debug, "check")


def _debug_check(cond, msg: str, *args) -> None:
    if _HAS_DEBUG_CHECK:
        jax.debug.check(cond, msg, *args)
        return
    try:
        ok = bool(cond)
    except Exception:
        return
    if not ok:
        raise ValueError(msg.format(*args))


def check_last_dim(arr: jax.Array, expected: int, label: str) -> None:
    _debug_check(arr.shape[-1] == expected, "{}: expected last dimension {}, got {}", label, expected, arr.shape)


def check_tail_shape(arr: jax.Array, expected: tuple[int, ...], label: str) -> None:
    _debug_check(
        arr.shape[-len(expected):] == expected, "{}: expected shape (..., {}), got {}", label, expected, arr.shape
    )


def check_ndim(arr: jax.Array, expected: int, label: str) -> None:
    _debug_check(arr.ndim == expected, "{}: expected ndim {}, got shape {}", label, expected, arr.shape)


def check_equal(a: jax.Array, b: jax.Array, label: str) -> None:
    _debug_check(a == b, "{}: expected {} == {}", label, a, b)


def check_in_set(val: str, allowed: tuple[str, ...], label: str) -> None:
    _debug_check(val in allowed, "{}: expected one of {}, got {}", label, allowed, val)


__all__ = ["check_last_dim", "check_tail_shape", "check_ndim", "check_equal", "check_in_set"]
