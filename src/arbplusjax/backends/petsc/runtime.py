from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class PetscBackendStatus:
    available: bool
    reason: str | None = None
    module: ModuleType | None = None


def get_petsc_module() -> ModuleType:
    if os.environ.get("ARBPLUSJAX_DISABLE_PETSC_BACKEND", "0") == "1":
        raise RuntimeError("PETSc backend is disabled by ARBPLUSJAX_DISABLE_PETSC_BACKEND")
    return import_module("petsc4py.PETSc")


def probe_petsc_backend() -> PetscBackendStatus:
    try:
        module = get_petsc_module()
    except ImportError:
        return PetscBackendStatus(available=False, reason="petsc4py is not importable")
    except RuntimeError as error:
        return PetscBackendStatus(available=False, reason=str(error))
    except Exception as error:
        return PetscBackendStatus(available=False, reason=f"PETSc backend import failed: {error}")

    try:
        module.Sys.getVersion()
    except Exception as error:
        return PetscBackendStatus(
            available=False,
            reason=f"PETSc backend runtime check failed: {error}",
            module=module,
        )

    return PetscBackendStatus(available=True, module=module)
