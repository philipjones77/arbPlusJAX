from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime import get_slepc_module


@dataclass(frozen=True)
class SlepcObject:
    native: Any
    kind: str | None = None

    def __getattr__(self, name: str):
        return getattr(self.native, name)

    def unwrap(self) -> Any:
        return self.native

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self)) | set(dir(self.native)))


def wrap_slepc_object(native_object, *, kind: str | None = None) -> SlepcObject:
    if isinstance(native_object, SlepcObject):
        return native_object
    return SlepcObject(native=native_object, kind=kind)


def unwrap_slepc_object(object_or_wrapper):
    if isinstance(object_or_wrapper, SlepcObject):
        return object_or_wrapper.native
    return object_or_wrapper


def native_slepc_module():
    return get_slepc_module()


def create_slepc_object(
    kind: str,
    *,
    slepc=None,
    create: bool = True,
    create_args: tuple[Any, ...] = (),
    create_kwargs: dict[str, Any] | None = None,
    wrap: bool = True,
):
    module = get_slepc_module() if slepc is None else slepc
    factory = getattr(module, kind)
    native_object = factory()
    if create and hasattr(native_object, "create"):
        kwargs = {} if create_kwargs is None else dict(create_kwargs)
        created = native_object.create(*create_args, **kwargs)
        if created is not None:
            native_object = created
    return wrap_slepc_object(native_object, kind=kind) if wrap else native_object
