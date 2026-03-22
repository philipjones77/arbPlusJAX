from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax

from .fingerprints import EvaluationFingerprint


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdRuleContext:
    fingerprint: EvaluationFingerprint
    saved_values: tuple[object, ...]

    def tree_flatten(self):
        children = (self.fingerprint, *self.saved_values)
        return children, {"saved_count": len(self.saved_values)}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        fingerprint = children[0]
        saved_count = aux_data["saved_count"]
        saved_values = tuple(children[1 : 1 + saved_count])
        return cls(fingerprint=fingerprint, saved_values=saved_values)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ResidualReport:
    primal_residual: object
    adjoint_residual: object
    note: str = ""

    def tree_flatten(self):
        return (self.primal_residual, self.adjoint_residual), {"note": self.note}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        primal_residual, adjoint_residual = children
        return cls(
            primal_residual=primal_residual,
            adjoint_residual=adjoint_residual,
            note=aux_data["note"],
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdAttachment:
    fingerprint: EvaluationFingerprint
    residuals: ResidualReport

    def tree_flatten(self):
        return (self.fingerprint, self.residuals), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        fingerprint, residuals = children
        return cls(fingerprint=fingerprint, residuals=residuals)


def context_with_fingerprint(fingerprint: EvaluationFingerprint, *saved_values) -> AdRuleContext:
    return AdRuleContext(fingerprint=fingerprint, saved_values=tuple(saved_values))


def make_residual_report(*, primal_residual=0.0, adjoint_residual=0.0, note: str = "") -> ResidualReport:
    return ResidualReport(
        primal_residual=primal_residual,
        adjoint_residual=adjoint_residual,
        note=note,
    )


def attach_rule_artifacts(
    fingerprint: EvaluationFingerprint,
    *,
    primal_residual=0.0,
    adjoint_residual=0.0,
    note: str = "",
) -> AdAttachment:
    return AdAttachment(
        fingerprint=fingerprint,
        residuals=make_residual_report(
            primal_residual=primal_residual,
            adjoint_residual=adjoint_residual,
            note=note,
        ),
    )


def context_with_artifacts(attachment: AdAttachment, *saved_values) -> AdRuleContext:
    return AdRuleContext(fingerprint=attachment.fingerprint, saved_values=(attachment.residuals, *saved_values))


def bind_custom_vjp(fn, *, fwd, bwd, nondiff_argnums=()):
    wrapped = partial(jax.custom_vjp, nondiff_argnums=nondiff_argnums)(fn)
    wrapped.defvjp(fwd, bwd)
    return wrapped


def bind_custom_jvp(fn, *, jvp, nondiff_argnums=()):
    wrapped = partial(jax.custom_jvp, nondiff_argnums=nondiff_argnums)(fn)
    wrapped.defjvp(jvp)
    return wrapped


__all__ = [
    "AdAttachment",
    "AdRuleContext",
    "ResidualReport",
    "attach_rule_artifacts",
    "bind_custom_jvp",
    "bind_custom_vjp",
    "context_with_artifacts",
    "context_with_fingerprint",
    "make_residual_report",
]
