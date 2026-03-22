from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from ...autodiff.ad_rules import context_with_fingerprint
from ...autodiff.fingerprints import EvaluationFingerprint, incomplete_gamma_regime_code, method_code
from .derivatives import incomplete_gamma_upper_argument_derivative
from .incomplete_gamma import _incomplete_gamma_upper_point_base, _normalize_method
from .regions import choose_incomplete_gamma_upper_method, incomplete_gamma_upper_regime_metadata


def incomplete_gamma_upper_switched_point_with_fingerprint(
    s,
    z,
    *,
    regularized: bool = False,
    method: str = "auto",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    normalized_method = choose_incomplete_gamma_upper_method(
        s,
        z,
        requested_method=_normalize_method(method, s, z),
    )
    value, diagnostics = _incomplete_gamma_upper_point_base(
        s,
        z,
        regularized=regularized,
        method=normalized_method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=True,
    )
    metadata = incomplete_gamma_upper_regime_metadata(s, z)
    scale = jnp.exp(-jnp.asarray(z, dtype=jnp.float64))
    fingerprint = EvaluationFingerprint(
        regime_code=incomplete_gamma_regime_code(
            near_singularity=metadata.near_singularity,
            cancellation_risk=metadata.cancellation_risk,
        ),
        method_code=method_code(normalized_method),
        work_units=jnp.asarray(
            diagnostics.chunk_count + diagnostics.panel_count + diagnostics.recurrence_steps,
            dtype=jnp.int32,
        ),
        scale=jnp.asarray(scale, dtype=jnp.float64),
        compensated_sum=jnp.asarray(False),
        adjoint_residual=jnp.asarray(0.0, dtype=jnp.float64),
        note="Switched upper incomplete-gamma evaluation with deterministic regime/method fingerprint.",
    )
    return value, fingerprint


def _incomplete_gamma_upper_switched_z_primal(
    s,
    z,
    regularized: bool = False,
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    value, _ = incomplete_gamma_upper_switched_point_with_fingerprint(
        s,
        z,
        regularized=regularized,
        method="auto",
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return value


@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3, 4, 5))
def incomplete_gamma_upper_switched_z_vjp(
    s,
    z,
    regularized: bool = False,
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    return _incomplete_gamma_upper_switched_z_primal(
        s,
        z,
        regularized=regularized,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )


def _incomplete_gamma_upper_switched_z_vjp_fwd(
    s,
    z,
    regularized: bool,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
):
    value, fingerprint = incomplete_gamma_upper_switched_point_with_fingerprint(
        s,
        z,
        regularized=regularized,
        method="auto",
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return value, context_with_fingerprint(
        fingerprint,
        jnp.asarray(s, dtype=jnp.float64),
        jnp.asarray(z, dtype=jnp.float64),
    )


def _incomplete_gamma_upper_switched_z_vjp_bwd(
    s,
    regularized: bool,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    ctx,
    g,
):
    del panel_width, max_panels, samples_per_panel
    s_val, z_val = ctx.saved_values
    derivative = incomplete_gamma_upper_argument_derivative(s_val, z_val, regularized=regularized)
    return (jnp.asarray(g, dtype=jnp.float64) * jnp.asarray(derivative, dtype=jnp.float64),)


incomplete_gamma_upper_switched_z_vjp.defvjp(
    _incomplete_gamma_upper_switched_z_vjp_fwd,
    _incomplete_gamma_upper_switched_z_vjp_bwd,
)


def incomplete_gamma_upper_switched_z_jvp(
    s,
    z,
    tangent_z,
    *,
    regularized: bool = False,
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    value = incomplete_gamma_upper_switched_z_vjp(
        s,
        z,
        regularized=regularized,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    deriv = incomplete_gamma_upper_argument_derivative(s, z, regularized=regularized)
    return value, jnp.asarray(tangent_z, dtype=jnp.float64) * jnp.asarray(deriv, dtype=jnp.float64)


__all__ = [
    "incomplete_gamma_upper_switched_point_with_fingerprint",
    "incomplete_gamma_upper_switched_z_jvp",
    "incomplete_gamma_upper_switched_z_vjp",
]
