from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

from ..petsc.runtime import get_petsc_module


@dataclass(frozen=True)
class SlepcBackendStatus:
    available: bool
    reason: str | None = None
    petsc_module: ModuleType | None = None
    slepc_module: ModuleType | None = None


def get_slepc_module() -> ModuleType:
    if os.environ.get("ARBPLUSJAX_DISABLE_SLEPC_BACKEND", "0") == "1":
        raise RuntimeError("SLEPc backend is disabled by ARBPLUSJAX_DISABLE_SLEPC_BACKEND")
    return import_module("slepc4py.SLEPc")


def probe_slepc_backend() -> SlepcBackendStatus:
    try:
        petsc_module = get_petsc_module()
        slepc_module = get_slepc_module()
    except ImportError:
        return SlepcBackendStatus(available=False, reason="petsc4py/slepc4py are not importable")
    except RuntimeError as error:
        return SlepcBackendStatus(available=False, reason=str(error))
    except Exception as error:
        return SlepcBackendStatus(available=False, reason=f"SLEPc backend import failed: {error}")

    try:
        petsc_module.Sys.getVersion()
        slepc_module.Sys.getVersion()
    except Exception as error:
        return SlepcBackendStatus(
            available=False,
            reason=f"SLEPc backend runtime check failed: {error}",
            petsc_module=petsc_module,
            slepc_module=slepc_module,
        )

    return SlepcBackendStatus(available=True, petsc_module=petsc_module, slepc_module=slepc_module)
