from __future__ import annotations

import jax


def _check(cond, msg: str | None = None) -> None:
    label = msg or "test_check"
    try:
        jax.debug.check(cond, "{}: {}", label, cond)
    except Exception:
        pass
    assert bool(cond)


__all__ = ["_check"]
