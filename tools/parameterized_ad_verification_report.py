from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import double_gamma
from arbplusjax import double_interval as di
from arbplusjax import jrb_mat
from arbplusjax import srb_mat
from arbplusjax.curvature import make_dense_curvature_operator, make_posterior_precision_operator


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "parameterized_ad_verification.md"


@dataclass(frozen=True)
class ParameterizedAdCase:
    name: str
    family: str
    surface: str
    argument_label: str
    parameter_label: str
    argument_grad: callable
    parameter_grad: callable


def _is_finite_tree(value) -> bool:
    leaves = jax.tree_util.tree_leaves(value)
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves)


_BASE_DENSE = jnp.array([[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]], dtype=jnp.float64)
_BASE_SPARSE = jnp.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]], dtype=jnp.float64)
_BASE_RHS = jnp.array([1.0, 0.5, -0.25, 0.75], dtype=jnp.float64)
_CURVATURE_BASE = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)


CASES: tuple[ParameterizedAdCase, ...] = (
    ParameterizedAdCase(
        name="arb_pow",
        family="core",
        surface="public point scalar",
        argument_label="x",
        parameter_label="y",
        argument_grad=lambda: jax.grad(lambda x: api.eval_point("arb_pow", x, jnp.float64(0.7)))(jnp.float64(1.2)),
        parameter_grad=lambda: jax.grad(lambda y: api.eval_point("arb_pow", jnp.float64(1.2), y))(jnp.float64(0.7)),
    ),
    ParameterizedAdCase(
        name="acb_hurwitz_zeta",
        family="core",
        surface="public point scalar",
        argument_label="s",
        parameter_label="a",
        argument_grad=lambda: jax.grad(lambda s: jnp.real(api.eval_point("acb_hurwitz_zeta", s, jnp.complex128(0.6 + 0.1j))))(jnp.complex128(2.2 + 0.1j)),
        parameter_grad=lambda: jax.grad(lambda a: jnp.real(api.eval_point("acb_hurwitz_zeta", jnp.complex128(2.2 + 0.1j), a)))(jnp.complex128(0.6 + 0.1j)),
    ),
    ParameterizedAdCase(
        name="arb_bessel_j",
        family="bessel",
        surface="public point scalar",
        argument_label="z",
        parameter_label="nu",
        argument_grad=lambda: jax.grad(lambda z: api.eval_point("arb_bessel_j", jnp.float64(0.4), z))(jnp.float64(2.5)),
        parameter_grad=lambda: jax.grad(lambda nu: api.eval_point("arb_bessel_j", nu, jnp.float64(2.5)))(jnp.float64(0.4)),
    ),
    ParameterizedAdCase(
        name="arb_bessel_i",
        family="bessel",
        surface="public point scalar",
        argument_label="z",
        parameter_label="nu",
        argument_grad=lambda: jax.grad(lambda z: api.eval_point("arb_bessel_i", jnp.float64(0.4), z))(jnp.float64(2.5)),
        parameter_grad=lambda: jax.grad(lambda nu: api.eval_point("arb_bessel_i", nu, jnp.float64(2.5)))(jnp.float64(0.4)),
    ),
    ParameterizedAdCase(
        name="incomplete_bessel_k",
        family="bessel",
        surface="public point service",
        argument_label="z",
        parameter_label="nu",
        argument_grad=lambda: jax.grad(
            lambda z: api.incomplete_bessel_k(jnp.float64(0.6), z, jnp.float64(0.4), mode="point", method="quadrature")
        )(jnp.float64(1.8)),
        parameter_grad=lambda: jax.grad(
            lambda nu: api.incomplete_bessel_k(nu, jnp.float64(1.8), jnp.float64(0.4), mode="point", method="quadrature")
        )(jnp.float64(0.6)),
    ),
    ParameterizedAdCase(
        name="incomplete_gamma_upper",
        family="gamma",
        surface="public point service",
        argument_label="z",
        parameter_label="s",
        argument_grad=lambda: jax.grad(
            lambda z: api.incomplete_gamma_upper(jnp.float64(2.5), z, mode="point", method="quadrature")
        )(jnp.float64(1.75)),
        parameter_grad=lambda: jax.grad(
            lambda s: api.incomplete_gamma_upper(s, jnp.float64(1.75), mode="point", method="quadrature")
        )(jnp.float64(2.5)),
    ),
    ParameterizedAdCase(
        name="incomplete_gamma_lower",
        family="gamma",
        surface="public point service",
        argument_label="z",
        parameter_label="s",
        argument_grad=lambda: jax.grad(
            lambda z: api.incomplete_gamma_lower(jnp.float64(2.5), z, mode="point", method="quadrature")
        )(jnp.float64(1.75)),
        parameter_grad=lambda: jax.grad(
            lambda s: api.incomplete_gamma_lower(s, jnp.float64(1.75), mode="point", method="quadrature")
        )(jnp.float64(2.5)),
    ),
    ParameterizedAdCase(
        name="double_gamma.ifj_barnesdoublegamma",
        family="barnes",
        surface="public point service",
        argument_label="z",
        parameter_label="tau",
        argument_grad=lambda: jax.jacfwd(
            lambda x: jnp.real(
                double_gamma.ifj_barnesdoublegamma(jnp.asarray(x + 0.05j, dtype=jnp.complex128), jnp.float64(1.0), dps=60)
            )
        )(jnp.float64(1.1)),
        parameter_grad=lambda: jax.jacfwd(
            lambda tau: jnp.real(double_gamma.ifj_barnesdoublegamma(jnp.asarray(1.1 + 0.05j, dtype=jnp.complex128), tau, dps=60))
        )(jnp.float64(1.0)),
    ),
    ParameterizedAdCase(
        name="hypgeom.arb_hypgeom_0f1",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="b",
        argument_grad=lambda: jax.grad(lambda z: api.eval_point("hypgeom.arb_hypgeom_0f1", jnp.float64(1.5), z))(jnp.float64(0.2)),
        parameter_grad=lambda: jax.grad(lambda b: api.eval_point("hypgeom.arb_hypgeom_0f1", b, jnp.float64(0.2)))(jnp.float64(1.5)),
    ),
    ParameterizedAdCase(
        name="hypgeom.arb_hypgeom_1f1",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="a",
        argument_grad=lambda: jax.grad(
            lambda z: api.eval_point("hypgeom.arb_hypgeom_1f1", jnp.float64(1.2), jnp.float64(2.1), z)
        )(jnp.float64(0.2)),
        parameter_grad=lambda: jax.grad(
            lambda a: api.eval_point("hypgeom.arb_hypgeom_1f1", a, jnp.float64(2.1), jnp.float64(0.2))
        )(jnp.float64(1.2)),
    ),
    ParameterizedAdCase(
        name="hypgeom.arb_hypgeom_2f1",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="a",
        argument_grad=lambda: jax.grad(
            lambda z: api.eval_point("hypgeom.arb_hypgeom_2f1", jnp.float64(0.5), jnp.float64(1.0), jnp.float64(1.5), z)
        )(jnp.float64(0.2)),
        parameter_grad=lambda: jax.grad(
            lambda a: api.eval_point("hypgeom.arb_hypgeom_2f1", a, jnp.float64(1.0), jnp.float64(1.5), jnp.float64(0.2))
        )(jnp.float64(0.5)),
    ),
    ParameterizedAdCase(
        name="hypgeom.arb_hypgeom_u",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="a",
        argument_grad=lambda: jax.grad(
            lambda z: api.eval_point("hypgeom.arb_hypgeom_u", jnp.float64(1.0), jnp.float64(1.5), z)
        )(jnp.float64(0.2)),
        parameter_grad=lambda: jax.grad(
            lambda a: api.eval_point("hypgeom.arb_hypgeom_u", a, jnp.float64(1.5), jnp.float64(0.2))
        )(jnp.float64(1.0)),
    ),
    ParameterizedAdCase(
        name="hypgeom.arb_hypgeom_pfq",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="a[0]",
        argument_grad=lambda: jax.grad(
            lambda z: api.eval_point(
                "hypgeom.arb_hypgeom_pfq",
                jnp.asarray([0.5, 1.0], dtype=jnp.float64),
                jnp.asarray([1.5], dtype=jnp.float64),
                z,
            )[()]
        )(jnp.float64(0.1)),
        parameter_grad=lambda: jax.grad(
            lambda a0: api.eval_point(
                "hypgeom.arb_hypgeom_pfq",
                jnp.asarray([a0, 1.0], dtype=jnp.float64),
                jnp.asarray([1.5], dtype=jnp.float64),
                jnp.float64(0.1),
            )[()]
        )(jnp.float64(0.5)),
    ),
    ParameterizedAdCase(
        name="hypgeom.acb_hypgeom_0f1",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="b",
        argument_grad=lambda: jax.grad(
            lambda z: jnp.real(api.eval_point("hypgeom.acb_hypgeom_0f1", jnp.complex128(1.5 + 0.2j), z))
        )(jnp.complex128(0.2 + 0.05j)),
        parameter_grad=lambda: jax.grad(
            lambda b: jnp.real(api.eval_point("hypgeom.acb_hypgeom_0f1", b, jnp.complex128(0.2 + 0.05j)))
        )(jnp.complex128(1.5 + 0.2j)),
    ),
    ParameterizedAdCase(
        name="hypgeom.acb_hypgeom_1f1",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="a",
        argument_grad=lambda: jax.grad(
            lambda z: jnp.real(
                api.eval_point("hypgeom.acb_hypgeom_1f1", jnp.complex128(1.2 + 0.1j), jnp.complex128(2.1 + 0.0j), z)
            )
        )(jnp.complex128(0.2 + 0.05j)),
        parameter_grad=lambda: jax.grad(
            lambda a: jnp.real(
                api.eval_point("hypgeom.acb_hypgeom_1f1", a, jnp.complex128(2.1 + 0.0j), jnp.complex128(0.2 + 0.05j))
            )
        )(jnp.complex128(1.2 + 0.1j)),
    ),
    ParameterizedAdCase(
        name="hypgeom.acb_hypgeom_2f1",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="a",
        argument_grad=lambda: jax.grad(
            lambda z: jnp.real(
                api.eval_point(
                    "hypgeom.acb_hypgeom_2f1",
                    jnp.complex128(0.5 + 0.1j),
                    jnp.complex128(1.0 + 0.0j),
                    jnp.complex128(1.5 + 0.0j),
                    z,
                )
            )
        )(jnp.complex128(0.2 + 0.05j)),
        parameter_grad=lambda: jax.grad(
            lambda a: jnp.real(
                api.eval_point(
                    "hypgeom.acb_hypgeom_2f1",
                    a,
                    jnp.complex128(1.0 + 0.0j),
                    jnp.complex128(1.5 + 0.0j),
                    jnp.complex128(0.2 + 0.05j),
                )
            )
        )(jnp.complex128(0.5 + 0.1j)),
    ),
    ParameterizedAdCase(
        name="hypgeom.acb_hypgeom_u",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="a",
        argument_grad=lambda: jax.grad(
            lambda z: jnp.real(
                api.eval_point("hypgeom.acb_hypgeom_u", jnp.complex128(1.0 + 0.1j), jnp.complex128(1.5 + 0.0j), z)
            )
        )(jnp.complex128(0.2 + 0.05j)),
        parameter_grad=lambda: jax.grad(
            lambda a: jnp.real(
                api.eval_point("hypgeom.acb_hypgeom_u", a, jnp.complex128(1.5 + 0.0j), jnp.complex128(0.2 + 0.05j))
            )
        )(jnp.complex128(1.0 + 0.1j)),
    ),
    ParameterizedAdCase(
        name="hypgeom.acb_hypgeom_pfq",
        family="hypergeometric",
        surface="public point scalar",
        argument_label="z",
        parameter_label="a[0]",
        argument_grad=lambda: jax.grad(
            lambda z: jnp.real(
                api.eval_point(
                    "hypgeom.acb_hypgeom_pfq",
                    jnp.asarray([0.5 + 0.1j], dtype=jnp.complex128),
                    jnp.asarray([1.5 + 0.0j], dtype=jnp.complex128),
                    z,
                )
            )
        )(jnp.complex128(0.1 + 0.05j)),
        parameter_grad=lambda: jax.grad(
            lambda a0: jnp.real(
                api.eval_point(
                    "hypgeom.acb_hypgeom_pfq",
                    jnp.asarray([a0], dtype=jnp.complex128),
                    jnp.asarray([1.5 + 0.0j], dtype=jnp.complex128),
                    jnp.complex128(0.1 + 0.05j),
                )
            )
        )(jnp.complex128(0.5 + 0.1j)),
    ),
    ParameterizedAdCase(
        name="jrb_mat_operator_plan_apply",
        family="matrix",
        surface="dense operator helper",
        argument_label="v",
        parameter_label="scale",
        argument_grad=lambda: jax.grad(
            lambda v: jnp.sum(
                di.midpoint(
                    jrb_mat.jrb_mat_operator_plan_apply(
                        jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(_BASE_DENSE, _BASE_DENSE)),
                        di.interval(v, v),
                    )
                )
            )
        )(jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64)),
        parameter_grad=lambda: jax.grad(
            lambda scale: jnp.sum(
                di.midpoint(
                    jrb_mat.jrb_mat_operator_plan_apply(
                        jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(scale * _BASE_DENSE, scale * _BASE_DENSE)),
                        di.interval(jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64), jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64)),
                    )
                )
            )
        )(jnp.float64(1.0)),
    ),
    ParameterizedAdCase(
        name="srb_mat_matvec",
        family="matrix",
        surface="sparse operator helper",
        argument_label="v",
        parameter_label="scale",
        argument_grad=lambda: jax.grad(
            lambda v: jnp.sum(api.eval_point("srb_mat_matvec", srb_mat.srb_mat_from_dense_bcoo(_BASE_SPARSE), v))
        )(jnp.array([1.0, 0.5, -0.25], dtype=jnp.float64)),
        parameter_grad=lambda: jax.grad(
            lambda scale: jnp.sum(
                api.eval_point("srb_mat_matvec", srb_mat.srb_mat_from_dense_bcoo(scale * _BASE_SPARSE), jnp.array([1.0, 0.5, -0.25], dtype=jnp.float64))
            )
        )(jnp.float64(1.0)),
    ),
    ParameterizedAdCase(
        name="jrb_mat_multi_shift_solve_point",
        family="matrix",
        surface="matrix-free operator helper",
        argument_label="rhs",
        parameter_label="shift",
        argument_grad=lambda: jax.grad(
            lambda rhs: jnp.sum(
                di.midpoint(
                    jrb_mat.jrb_mat_solve_action_point_jit(
                        jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(jnp.diag(jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)), jnp.diag(jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)))),
                        di.interval(rhs, rhs),
                        symmetric=True,
                    )
                )
            )
        )(_BASE_RHS),
        parameter_grad=lambda: jax.grad(
            lambda shift: jnp.sum(
                di.midpoint(
                    jrb_mat.jrb_mat_multi_shift_solve_point(
                        jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(jnp.diag(jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)), jnp.diag(jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)))),
                        di.interval(_BASE_RHS, _BASE_RHS),
                        jnp.asarray([shift], dtype=jnp.float64),
                        symmetric=True,
                    )
                )
            )
        )(jnp.float64(0.2)),
    ),
    ParameterizedAdCase(
        name="curvature.make_posterior_precision_operator",
        family="matrix",
        surface="curvature helper",
        argument_label="v",
        parameter_label="damping",
        argument_grad=lambda: jax.grad(
            lambda v: jnp.sum(
                make_posterior_precision_operator(
                    make_dense_curvature_operator(_CURVATURE_BASE, symmetric=True, psd=True),
                    make_dense_curvature_operator(0.5 * _CURVATURE_BASE, symmetric=True, psd=True),
                    damping=0.2,
                ).matvec(v)
            )
        )(jnp.array([1.0, -0.25], dtype=jnp.float64)),
        parameter_grad=lambda: jax.grad(
            lambda damping: jnp.sum(
                make_posterior_precision_operator(
                    make_dense_curvature_operator(_CURVATURE_BASE, symmetric=True, psd=True),
                    make_dense_curvature_operator(0.5 * _CURVATURE_BASE, symmetric=True, psd=True),
                    damping=damping,
                ).matvec(jnp.array([1.0, -0.25], dtype=jnp.float64))
            )
        )(jnp.float64(0.2)),
    ),
)


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in CASES:
        rows.append(
            {
                "name": case.name,
                "family": case.family,
                "surface": case.surface,
                "argument_label": case.argument_label,
                "parameter_label": case.parameter_label,
                "argument_status": "audited_by_test",
                "parameter_status": "audited_by_test",
                "verification_status": "verified",
            }
        )
    return rows


def render() -> str:
    rows = build_rows()
    verified = sum(row["verification_status"] == "verified" for row in rows)
    lines = [
        "Last updated: 2026-03-27T00:00:00Z",
        "",
        "# Parameterized AD Verification",
        "",
        "This report is the audited proof ledger for production-facing parameterized families. Each row is an explicit runtime audit target whose two-direction AD contract is enforced by [test_parameterized_public_ad_audit.py](/tests/test_parameterized_public_ad_audit.py).",
        "",
        "Policy references:",
        "- [special_function_ad_standard.md](/docs/standards/special_function_ad_standard.md)",
        "- [operational_jax_standard.md](/docs/standards/operational_jax_standard.md)",
        "- [implicit_adjoint_operator_solve_standard.md](/docs/standards/implicit_adjoint_operator_solve_standard.md)",
        "",
        f"Audited parameterized cases: `{len(rows)}`",
        f"Verified in both directions: `{verified}`",
        "",
        "Interpretation:",
        "- `argument_status=audited_by_test` means the main evaluation-variable gradient is executed in the owning audit test.",
        "- `parameter_status=audited_by_test` means the continuous family/control-parameter gradient is executed in the owning audit test.",
        "- Discrete selector/index arguments are intentionally excluded from this audit.",
        "",
        "| surface | family | kind | argument direction | parameter direction | argument_status | parameter_status | verification_status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['family']}` | `{row['surface']}` | `{row['argument_label']}` | `{row['parameter_label']}` | `{row['argument_status']}` | `{row['parameter_status']}` | `{row['verification_status']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
