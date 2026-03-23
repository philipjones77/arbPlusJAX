from __future__ import annotations

import jax.numpy as jnp
import pytest

from arbplusjax import checks


def test_checks_public_helpers_accept_valid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "_HAS_DEBUG_CHECK", False)

    arr = jnp.ones((2, 3, 4))
    checks.check(True, "truthy")
    checks.check_last_dim(arr, 4, "last")
    checks.check_tail_shape(arr, (3, 4), "tail")
    checks.check_ndim(arr, 3, "ndim")
    checks.check_equal(2, 2, "equal")
    checks.check_in_set("cpu", ("cpu", "gpu"), "mode")


def test_checks_public_helpers_raise_on_invalid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "_HAS_DEBUG_CHECK", False)

    arr = jnp.ones((2, 3, 4))
    with pytest.raises(ValueError):
        checks.check(False, "truthy")
    with pytest.raises(ValueError):
        checks.check_last_dim(arr, 5, "last")
    with pytest.raises(ValueError):
        checks.check_tail_shape(arr, (2, 4), "tail")
    with pytest.raises(ValueError):
        checks.check_ndim(arr, 2, "ndim")
    with pytest.raises(ValueError):
        checks.check_equal(2, 3, "equal")
    with pytest.raises(ValueError):
        checks.check_in_set("tpu", ("cpu", "gpu"), "mode")
