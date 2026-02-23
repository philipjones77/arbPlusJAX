from __future__ import annotations

import importlib
import inspect
import pkgutil
from functools import wraps
from types import ModuleType
from typing import Callable

from . import precision


def _resolve_prec_bits(dps: int | None, prec_bits: int | None) -> int:
    if prec_bits is not None:
        return int(prec_bits)
    if dps is not None:
        return precision.dps_to_bits(int(dps))
    return precision.get_prec_bits()


def _make_mp_wrapper(fn: Callable[..., object]) -> Callable[..., object]:
    @wraps(fn)
    def wrapper(*args, dps: int | None = None, prec_bits: int | None = None, **kwargs):
        pb = _resolve_prec_bits(dps, prec_bits)
        return fn(*args, prec_bits=pb, **kwargs)

    return wrapper


def _target_name(name: str) -> str | None:
    if name.endswith("_batch_prec"):
        return f"{name[:-11]}_batch_mp"
    if name.endswith("_prec"):
        return f"{name[:-5]}_mp"
    return None


def _supports_prec_bits(fn: Callable[..., object]) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "prec_bits" in sig.parameters


def _register_module(module: ModuleType) -> None:
    for name in dir(module):
        if name.startswith("_"):
            continue
        target = _target_name(name)
        if target is None:
            continue
        fn = getattr(module, name, None)
        if not callable(fn) or not _supports_prec_bits(fn):
            continue
        globals()[target] = _make_mp_wrapper(fn)
        __all__.append(target)


__all__: list[str] = []

_SKIP_MODULES = {
    "mp_mode",
    "precision",
    "runtime",
    "validation",
}

_pkg = importlib.import_module(__package__ or "arbplusjax")
for _info in pkgutil.iter_modules(_pkg.__path__):
    _name = _info.name
    if _name.startswith("_") or _name in _SKIP_MODULES:
        continue
    _mod = importlib.import_module(f"{_pkg.__name__}.{_name}")
    _register_module(_mod)
