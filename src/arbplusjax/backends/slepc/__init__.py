from __future__ import annotations

from .eigs import EigensolveConfig, EigensolveResult, create_eps, solve_eigenproblem
from .native import (
    SlepcObject,
    create_slepc_object,
    native_slepc_module,
    unwrap_slepc_object,
    wrap_slepc_object,
)
from .runtime import SlepcBackendStatus, get_slepc_module, probe_slepc_backend

__all__ = [
    "EigensolveConfig",
    "EigensolveResult",
    "SlepcObject",
    "SlepcBackendStatus",
    "create_eps",
    "create_slepc_object",
    "get_slepc_module",
    "native_slepc_module",
    "probe_slepc_backend",
    "solve_eigenproblem",
    "unwrap_slepc_object",
    "wrap_slepc_object",
]
