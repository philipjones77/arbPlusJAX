from __future__ import annotations

from typing import Callable, TypeVar


F = TypeVar("F", bound=Callable)


def lazy_jit(factory: Callable[[], F]) -> F:
    """Create a lazily initialized compiled wrapper.

    This avoids paying import-time wrapper construction cost for large alias
    surfaces that expose many `*_jit` callables.
    """

    state: dict[str, F] = {}

    def wrapped(*args, **kwargs):
        fn = state.get("fn")
        if fn is None:
            fn = factory()
            state["fn"] = fn
        return fn(*args, **kwargs)

    return wrapped  # type: ignore[return-value]
