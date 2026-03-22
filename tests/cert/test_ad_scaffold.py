import jax
import jax.numpy as jnp

from arbplusjax.autodiff import ad_rules
from arbplusjax.autodiff import fingerprints


def _toy_matrix_apply(a, x):
    return jnp.asarray(a, dtype=jnp.float64) @ jnp.asarray(x, dtype=jnp.float64)


def test_autodiff_scaffold_artifact_pytrees():
    fp = fingerprints.make_fingerprint(
        regime_code_value=fingerprints.regime_code("structured"),
        method_code_value=fingerprints.method_code("quadrature"),
        work_units=7,
        scale=2.0,
        compensated_sum=False,
        adjoint_residual=0.0,
        note="toy",
    )
    attachment = ad_rules.attach_rule_artifacts(
        fp,
        primal_residual=jnp.asarray(1e-8, dtype=jnp.float64),
        adjoint_residual=jnp.asarray(2e-8, dtype=jnp.float64),
        note="matrix",
    )
    leaves, treedef = jax.tree_util.tree_flatten(attachment)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert int(rebuilt.fingerprint.work_units) == 7
    assert float(rebuilt.residuals.primal_residual) == float(jnp.asarray(1e-8, dtype=jnp.float64))
    assert float(rebuilt.residuals.adjoint_residual) == float(jnp.asarray(2e-8, dtype=jnp.float64))


def test_autodiff_scaffold_custom_vjp_binding_for_matrix_style_apply():
    def primal(a, x):
        return _toy_matrix_apply(a, x)

    def fwd(a, x):
        fp = fingerprints.make_fingerprint(
            regime_code_value=fingerprints.regime_code("structured"),
            method_code_value=fingerprints.method_code("quadrature"),
            work_units=a.shape[0],
            scale=jnp.linalg.norm(a),
            note="toy-matvec",
        )
        ctx = ad_rules.context_with_artifacts(
            ad_rules.attach_rule_artifacts(fp, primal_residual=0.0, adjoint_residual=0.0),
            jnp.asarray(a, dtype=jnp.float64),
            jnp.asarray(x, dtype=jnp.float64),
        )
        return primal(a, x), ctx

    def bwd(ctx, g):
        _residuals, a, x = ctx.saved_values
        g = jnp.asarray(g, dtype=jnp.float64)
        da = jnp.outer(g, x)
        dx = a.T @ g
        return da, dx

    wrapped = ad_rules.bind_custom_vjp(primal, fwd=fwd, bwd=bwd)

    a = jnp.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=jnp.float64)
    x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    out = wrapped(a, x)
    assert jnp.allclose(out, a @ x)

    loss = lambda aa, xx: jnp.sum(wrapped(aa, xx))
    ga, gx = jax.grad(loss, argnums=(0, 1))(a, x)
    assert ga.shape == a.shape
    assert gx.shape == x.shape
    assert jnp.all(jnp.isfinite(ga))
    assert jnp.all(jnp.isfinite(gx))
