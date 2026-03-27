from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Callable


def lazy_attr(module_name: str, attr_name: str) -> tuple[str, str]:
    return (module_name, attr_name)


def lazy_pair(module_name: str, fixed_name: str, padded_name: str) -> tuple[str, str, str]:
    return (module_name, fixed_name, padded_name)


@lru_cache(maxsize=None)
def load_module_attr(module_name: str, attr_name: str, *, package: str) -> Callable:
    mod = importlib.import_module(f".{module_name}", package=package)
    value = getattr(mod, attr_name)
    if not callable(value):
        raise TypeError(f"{module_name}.{attr_name} is not callable")
    return value


def resolve_lazy_callable(entry: object, *, package: str) -> Callable:
    if isinstance(entry, tuple) and len(entry) == 2 and all(isinstance(part, str) for part in entry):
        module_name, attr_name = entry
        return load_module_attr(module_name, attr_name, package=package)
    if not callable(entry):
        raise TypeError(f"expected callable entry, got {type(entry).__name__}")
    return entry


def resolve_lazy_pair(entry: object, *, package: str) -> tuple[Callable, Callable]:
    if isinstance(entry, tuple) and len(entry) == 3 and all(isinstance(part, str) for part in entry):
        module_name, fixed_name, padded_name = entry
        return (
            load_module_attr(module_name, fixed_name, package=package),
            load_module_attr(module_name, padded_name, package=package),
        )
    if isinstance(entry, tuple) and len(entry) == 2:
        fixed_fn, padded_fn = entry
        return (
            resolve_lazy_callable(fixed_fn, package=package),
            resolve_lazy_callable(padded_fn, package=package),
        )
    raise TypeError(f"expected fixed/padded pair, got {entry!r}")


class LazyModuleProxy:
    def __init__(self, module_name: str, *, package: str):
        self._module_name = module_name
        self._package = package
        self._module = None

    def __getattr__(self, attr: str):
        if self._module is None:
            self._module = importlib.import_module(f".{self._module_name}", package=self._package)
        return getattr(self._module, attr)


def lazy_module_proxy(module_name: str, *, package: str) -> LazyModuleProxy:
    return LazyModuleProxy(module_name, package=package)


__all__ = [
    "LazyModuleProxy",
    "lazy_attr",
    "lazy_module_proxy",
    "lazy_pair",
    "load_module_attr",
    "resolve_lazy_callable",
    "resolve_lazy_pair",
]
