from __future__ import annotations

import jax
import jax.numpy as jnp


def _mfc():
    from arbplusjax import matrix_free_core

    return matrix_free_core


def _enable_x64():
    from arbplusjax import precision

    precision.enable_jax_x64()


def chain_precision(theta: jax.Array, n: int) -> jax.Array:
    """Simple SPD chain precision Q(theta)."""
    scale = jnp.exp(jnp.asarray(theta, dtype=jnp.float64))
    diagonal = (1.0 + 2.0 * scale) * jnp.ones((n,), dtype=jnp.float64)
    off = (-scale) * jnp.ones((n - 1,), dtype=jnp.float64)
    return jnp.diag(diagonal) + jnp.diag(off, 1) + jnp.diag(off, -1)


def bernoulli_neg_log_posterior(x: jax.Array, y: jax.Array, theta: jax.Array) -> jax.Array:
    q = chain_precision(theta, x.shape[0])
    prior = 0.5 * jnp.vdot(x, q @ x).real
    likelihood = jnp.sum(jnp.logaddexp(0.0, x) - y * x)
    return jnp.asarray(prior + likelihood, dtype=jnp.float64)


def posterior_gradient(x: jax.Array, y: jax.Array, theta: jax.Array) -> jax.Array:
    return jax.grad(bernoulli_neg_log_posterior)(x, y, theta)


def posterior_hessian_midpoint(theta: jax.Array, x: jax.Array) -> jax.Array:
    q = chain_precision(theta, x.shape[0])
    p = jax.nn.sigmoid(x)
    w = p * (1.0 - p)
    jitter = jnp.asarray(1e-6, dtype=jnp.float64)
    return q + jnp.diag(w + jitter)


def posterior_hessian_plan(theta: jax.Array, x: jax.Array):
    dense = posterior_hessian_midpoint(theta, x)
    return _mfc().dense_operator_plan(dense, orientation="forward", algebra="jrb")


def posterior_hessian_jacobi(theta: jax.Array, x: jax.Array):
    dense = posterior_hessian_midpoint(theta, x)
    return _mfc().dense_jacobi_preconditioner_plan(dense, algebra="jrb")


def newton_mode(
    y: jax.Array,
    theta: jax.Array,
    *,
    iters: int = 6,
    tol: float = 1e-8,
    damping: float = 1.0,
) -> tuple[jax.Array, dict[str, object]]:
    x = jnp.zeros((y.shape[0],), dtype=jnp.float64)
    last_meta = None
    last_residual = jnp.asarray(jnp.inf, dtype=jnp.float64)

    for _ in range(iters):
        grad = posterior_gradient(x, y, theta)
        plan = posterior_hessian_plan(theta, x)
        preconditioner = posterior_hessian_jacobi(theta, x)
        step, _info, residual, _rhs_norm, meta = _mfc().implicit_krylov_solve_midpoint(
            plan,
            -grad,
            solver="cg",
            structured="spd",
            preconditioner=preconditioner,
            midpoint_vector=jnp.asarray,
            lift_vector=jnp.asarray,
            sparse_bcoo_matvec=lambda *_args, **_kwargs: (_ for _ in ()).throw(NotImplementedError("sparse path unused")),
            dtype=jnp.float64,
        )
        x = x + jnp.asarray(damping, dtype=jnp.float64) * step
        last_meta = meta
        last_residual = residual
        if float(jnp.linalg.norm(step)) < tol:
            break

    return x, {"solve_metadata": last_meta, "residual": last_residual}


def laplace_log_marginal(y: jax.Array, theta: jax.Array) -> tuple[jax.Array, dict[str, object]]:
    x_hat, newton_aux = newton_mode(y, theta)
    hessian_mid = posterior_hessian_midpoint(theta, x_hat)
    sign, logabsdet = jnp.linalg.slogdet(hessian_mid)
    objective = -bernoulli_neg_log_posterior(x_hat, y, theta)
    value = objective - 0.5 * logabsdet
    return value, {
        "mode": x_hat,
        "solve_metadata": newton_aux["solve_metadata"],
        "hessian_midpoint": hessian_mid,
        "spd_safe": bool(jnp.all(jnp.linalg.eigvalsh(hessian_mid) > 0.0)),
        "logdet_sign": sign,
    }


def run_example():
    _enable_x64()
    y = jnp.asarray([1.0, 0.0, 1.0, 1.0], dtype=jnp.float64)
    theta = jnp.asarray(-0.2, dtype=jnp.float64)
    value, aux = laplace_log_marginal(y, theta)
    grad = jax.grad(lambda t: laplace_log_marginal(y, t)[0])(theta)
    print("latent Gaussian Laplace example")
    print(f"theta={float(theta):+.3f}")
    print(f"laplace_log_marginal={float(value):+.6f}")
    print(f"gradient={float(grad):+.6f}")
    print(f"mode={aux['mode']}")
    print(f"spd_safe={aux['spd_safe']}")
    print(f"implicit_adjoint={aux['solve_metadata'].implicit_adjoint}")


if __name__ == "__main__":
    run_example()
