from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .lowering import to_petsc_mat, to_petsc_vec
from .runtime import get_petsc_module


@dataclass(frozen=True)
class PetscObject:
    native: Any
    kind: str | None = None

    def __getattr__(self, name: str):
        return getattr(self.native, name)

    def unwrap(self) -> Any:
        return self.native

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self)) | set(dir(self.native)))


def wrap_petsc_object(native_object, *, kind: str | None = None) -> PetscObject:
    if isinstance(native_object, PetscObject):
        return native_object
    return PetscObject(native=native_object, kind=kind)


def unwrap_petsc_object(object_or_wrapper):
    if isinstance(object_or_wrapper, PetscObject):
        return object_or_wrapper.native
    return object_or_wrapper


def native_petsc_module():
    return get_petsc_module()


def create_petsc_object(
    kind: str,
    *,
    petsc=None,
    create: bool = True,
    create_args: tuple[Any, ...] = (),
    create_kwargs: dict[str, Any] | None = None,
    wrap: bool = True,
):
    module = get_petsc_module() if petsc is None else petsc
    factory = getattr(module, kind)
    native_object = factory()
    if create and hasattr(native_object, "create"):
        kwargs = {} if create_kwargs is None else dict(create_kwargs)
        created = native_object.create(*create_args, **kwargs)
        if created is not None:
            native_object = created
    return wrap_petsc_object(native_object, kind=kind) if wrap else native_object


def create_vec(
    values=None,
    *,
    size: int | None = None,
    petsc=None,
    wrap: bool = True,
):
    module = get_petsc_module() if petsc is None else petsc
    if values is not None:
        native_vec = to_petsc_vec(values, petsc=module)
    else:
        native_vec = module.Vec()
        if size is not None and hasattr(native_vec, "createSeq"):
            native_vec = native_vec.createSeq(int(size))
    return wrap_petsc_object(native_vec, kind="Vec") if wrap else native_vec


def create_mat(
    operator=None,
    *,
    shape: tuple[int, int] | None = None,
    petsc=None,
    wrap: bool = True,
):
    module = get_petsc_module() if petsc is None else petsc
    if operator is None:
        native_mat = module.Mat()
        if hasattr(native_mat, "create"):
            created = native_mat.create()
            if created is not None:
                native_mat = created
    else:
        native_mat = to_petsc_mat(operator, shape=shape, petsc=module)
    return wrap_petsc_object(native_mat, kind="Mat") if wrap else native_mat


def create_dmplex_from_cell_list(
    cells,
    coordinates,
    *,
    dim: int | None = None,
    interpolate: bool | None = None,
    petsc=None,
    wrap: bool = True,
    comm=None,
):
    module = get_petsc_module() if petsc is None else petsc
    cell_array = np.asarray(cells, dtype=np.int32)
    coord_array = np.asarray(coordinates)
    plex_factory = getattr(module, "DMPlex", None)
    if plex_factory is None:
        raise AttributeError("Native PETSc module does not expose DMPlex")
    native_plex = plex_factory()
    spatial_dim = int(dim if dim is not None else jnp.asarray(coord_array).shape[-1])
    kwargs = {}
    if comm is not None:
        kwargs["comm"] = comm
    if interpolate is not None:
        kwargs["interpolate"] = bool(interpolate)
    try:
        created = native_plex.createFromCellList(spatial_dim, cell_array, coord_array, **kwargs)
    except TypeError:
        positional = [spatial_dim, cell_array, coord_array]
        if interpolate is not None:
            positional.append(bool(interpolate))
        created = native_plex.createFromCellList(*positional)
    if created is not None:
        native_plex = created
    return wrap_petsc_object(native_plex, kind="DMPlex") if wrap else native_plex
