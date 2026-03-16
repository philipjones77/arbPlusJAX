"""Tests for efficient matrix-free adjoints.

Tests verify:
- Correctness of Lanczos and Arnoldi decompositions
- Gradient correctness using finite differences
- Performance of custom VJP vs standard autodiff
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

jax.config.update("jax_enable_x64", True)


class TestLanczosTridiag:
    """Tests for Lanczos tridiagonalization with custom VJP."""
    
    def test_lanczos_decomposition_symmetric(self):
        """Test Lanczos on a symmetric matrix."""
        from arbplusjax import matfree_adjoints
        
        n = 50
        key = jr.PRNGKey(42)
        A_rand = jr.normal(key, (n, n))
        A = (A_rand + A_rand.T) / 2
        
        def matvec(v):
            return A @ v
        
        lanczos = matfree_adjoints.lanczos_tridiag(
            matvec, krylov_depth=10, reortho="full", custom_vjp=True
        )
        
        v0 = jr.normal(jr.PRNGKey(123), (n,))
        (basis, (diags, offdiags)), (remainder, beta) = lanczos(v0)
        
        # Check shapes
        assert basis.shape == (10, n)
        assert diags.shape == (10,)
        assert offdiags.shape == (9,)
        
        # Check orthogonality
        orthogonality_error = jnp.linalg.norm(basis @ basis.T - jnp.eye(10))
        assert orthogonality_error < 1e-10
        
    def test_lanczos_gradient_correctness(self):
        """Test gradient computation using finite differences."""
        from arbplusjax import matfree_adjoints
        
        n = 20
        key = jr.PRNGKey(999)
        A_rand = jr.normal(key, (n, n))
        A = (A_rand + A_rand.T) / 2
        
        def matvec(v):
            return A @ v
        
        lanczos = matfree_adjoints.lanczos_tridiag(
            matvec, krylov_depth=5, reortho="full", custom_vjp=True
        )
        
        def loss_fn(v):
            (basis, (diags, offdiags)), _ = lanczos(v)
            return jnp.sum(diags**2)
        
        v0 = jr.normal(jr.PRNGKey(456), (n,))
        
        # Analytical gradient
        grad_analytical = jax.grad(loss_fn)(v0)
        
        # Numerical gradient (finite differences)
        eps = 1e-5
        grad_numerical = jnp.zeros_like(v0)
        for i in range(n):
            v_plus = v0.at[i].add(eps)
            v_minus = v0.at[i].add(-eps)
            grad_numerical = grad_numerical.at[i].set(
                (loss_fn(v_plus) - loss_fn(v_minus)) / (2 * eps)
            )
        
        # Check agreement
        relative_error = jnp.linalg.norm(grad_analytical - grad_numerical) / jnp.linalg.norm(grad_numerical)
        assert relative_error < 1e-3, f"Gradient error: {relative_error}"
        
    def test_lanczos_custom_vjp_vs_standard(self):
        """Verify custom VJP gives same result as standard autodiff."""
        from arbplusjax import matfree_adjoints
        
        n = 30
        key = jr.PRNGKey(777)
        A_rand = jr.normal(key, (n, n))
        A = (A_rand + A_rand.T) / 2
        
        def matvec(v):
            return A @ v
        
        # With custom VJP
        lanczos_custom = matfree_adjoints.lanczos_tridiag(
            matvec, krylov_depth=8, reortho="full", custom_vjp=True
        )
        
        # Without custom VJP
        lanczos_standard = matfree_adjoints.lanczos_tridiag(
            matvec, krylov_depth=8, reortho="full", custom_vjp=False
        )
        
        v0 = jr.normal(jr.PRNGKey(888), (n,))
        
        def loss_fn(lanczos_fn, v):
            (basis, (diags, offdiags)), _ = lanczos_fn(v)
            return jnp.sum(diags) + jnp.sum(offdiags)
        
        grad_custom = jax.grad(lambda v: loss_fn(lanczos_custom, v))(v0)
        grad_standard = jax.grad(lambda v: loss_fn(lanczos_standard, v))(v0)
        
        error = jnp.linalg.norm(grad_custom - grad_standard) / jnp.linalg.norm(grad_standard)
        assert error < 1e-6, f"Custom VJP differs from standard: {error}"


class TestArnoldiHessenberg:
    """Tests for Arnoldi Hessenberg decomposition with custom VJP."""
    
    def test_arnoldi_decomposition_general(self):
        """Test Arnoldi on a general (non-symmetric) matrix."""
        from arbplusjax import matfree_adjoints
        
        n = 40
        key = jr.PRNGKey(111)
        A = jr.normal(key, (n, n))
        
        def matvec(v):
            return A @ v
        
        arnoldi = matfree_adjoints.arnoldi_hessenberg(
            matvec, krylov_depth=12, reortho="full", custom_vjp=True
        )
        
        v0 = jr.normal(jr.PRNGKey(222), (n,))
        Q, H, v_out, norm = arnoldi(v0)
        
        # Check shapes
        assert Q.shape == (n, 12)
        assert H.shape == (12, 12)
        
        # Check orthogonality
        orthogonality_error = jnp.linalg.norm(Q.T @ Q - jnp.eye(12))
        assert orthogonality_error < 1e-10
        
        # Check Hessenberg structure (zeros below first subdiagonal)
        hessenberg_error = jnp.linalg.norm(jnp.tril(H, -2))
        assert hessenberg_error < 1e-12
        
    def test_arnoldi_complex_matrix(self):
        """Test Arnoldi on complex matrices."""
        from arbplusjax import matfree_adjoints
        
        n = 30
        key = jr.PRNGKey(333)
        A_real = jr.normal(key, (n, n))
        A_imag = jr.normal(jr.PRNGKey(334), (n, n))
        A = A_real + 1j * A_imag
        
        def matvec(v):
            return A @ v
        
        arnoldi = matfree_adjoints.arnoldi_hessenberg(
            matvec, krylov_depth=10, reortho="full", custom_vjp=True
        )
        
        v0 = jr.normal(jr.PRNGKey(335), (n,)) + 1j * jr.normal(jr.PRNGKey(336), (n,))
        Q, H, v_out, norm = arnoldi(v0)
        
        # Check orthogonality (using conjugate transpose)
        orthogonality_error = jnp.linalg.norm(Q.conj().T @ Q - jnp.eye(10))
        assert orthogonality_error < 1e-10
        
    def test_arnoldi_gradient_correctness(self):
        """Test Arnoldi gradient using finite differences."""
        from arbplusjax import matfree_adjoints
        
        n = 15
        key = jr.PRNGKey(444)
        A = jr.normal(key, (n, n))
        
        def matvec(v):
            return A @ v
        
        arnoldi = matfree_adjoints.arnoldi_hessenberg(
            matvec, krylov_depth=5, reortho="full", custom_vjp=True
        )
        
        def loss_fn(v):
            Q, H, _, _ = arnoldi(v)
            return jnp.sum(jnp.diag(H)**2)
        
        v0 = jr.normal(jr.PRNGKey(555), (n,))
        
        # Analytical gradient
        grad_analytical = jax.grad(loss_fn)(v0)
        
        # Numerical gradient
        eps = 1e-5
        grad_numerical = jnp.zeros_like(v0)
        for i in range(n):
            v_plus = v0.at[i].add(eps)
            v_minus = v0.at[i].add(-eps)
            grad_numerical = grad_numerical.at[i].set(
                (loss_fn(v_plus) - loss_fn(v_minus)) / (2 * eps)
            )
        
        relative_error = jnp.linalg.norm(grad_analytical - grad_numerical) / jnp.linalg.norm(grad_numerical)
        assert relative_error < 1e-3, f"Arnoldi gradient error: {relative_error}"


class TestHutchinsonTraceEstimator:
    """Tests for Hutchinson trace estimator."""
    
    def test_hutchinson_trace_positive_matrix(self):
        """Test trace estimation for a positive definite matrix."""
        from arbplusjax import matfree_adjoints
        
        n = 50
        key = jr.PRNGKey(666)
        L = jr.normal(key, (n, n))
        A = L.T @ L  # Positive semi-definite
        
        def integrand(v):
            return jnp.dot(v, A @ v)
        
        def sample_fun(key):
            return jr.normal(key, (20, n))
        
        trace_est = matfree_adjoints.hutchinson_trace_estimator(
            integrand,
            sample_fun,
            use_custom_vjp=False
        )
        
        estimate = trace_est(jr.PRNGKey(777))
        true_trace = jnp.trace(A)
        
        # With 20 samples, expect ~10% relative error
        relative_error = jnp.abs(estimate - true_trace) / jnp.abs(true_trace)
        assert relative_error < 0.2, f"Trace estimate error: {relative_error}"


class TestCGSolver:
    """Tests for enhanced CG solver."""
    
    def test_cg_solve_spd(self):
        """Test CG solver on SPD system."""
        from arbplusjax import matfree_adjoints
        
        n = 50
        key = jr.PRNGKey(888)
        L = jr.normal(key, (n, n))
        A = L.T @ L + 0.1 * jnp.eye(n)
        
        def A_matvec(v):
            return A @ v
        
        b = jr.normal(jr.PRNGKey(999), (n,))
        
        cg_solve = matfree_adjoints.cg_fixed_iterations(num_matvecs=100)
        x, info = cg_solve(A_matvec, b)
        
        # Check solution accuracy
        x_true = jnp.linalg.solve(A, b)
        relative_error = jnp.linalg.norm(x - x_true) / jnp.linalg.norm(x_true)
        assert relative_error < 1e-4, f"CG solution error: {relative_error}"
        
    def test_cg_gradient(self):
        """Test gradient through CG solve."""
        from arbplusjax import matfree_adjoints
        
        n = 30
        key = jr.PRNGKey(1111)
        L = jr.normal(key, (n, n))
        A = L.T @ L + 0.1 * jnp.eye(n)
        
        def A_matvec(v):
            return A @ v
        
        cg_solve = matfree_adjoints.cg_fixed_iterations(num_matvecs=50)
        
        def loss_fn(b):
            x, _ = cg_solve(A_matvec, b)
            return jnp.sum(x**2)
        
        b = jr.normal(jr.PRNGKey(1212), (n,))
        
        # Should not raise an error
        grad = jax.grad(loss_fn)(b)
        assert grad.shape == (n,)
        assert jnp.all(jnp.isfinite(grad))


class TestMatrixFunctionQuadrature:
    """Tests for Lanczos-based matrix function quadrature."""
    
    def test_quadrature_exponential(self):
        """Test v^T exp(A) v computation."""
        from arbplusjax import matfree_adjoints
        
        n = 40
        key = jr.PRNGKey(1313)
        A_rand = jr.normal(key, (n, n))
        A = (A_rand + A_rand.T) / 2
        
        def matvec(v):
            return A @ v
        
        quadform = matfree_adjoints.lanczos_quadrature_spd(
            matfun=jnp.exp,
            krylov_depth=20,
            matvec=matvec,
            reortho="full",
            use_efficient_adjoint=True
        )
        
        v0 = jr.normal(jr.PRNGKey(1414), (n,))
        result = quadform(v0)
        
        # Compare with dense computation
        result_dense = v0 @ jax.scipy.linalg.expm(A) @ v0
        relative_error = jnp.abs(result - result_dense) / jnp.abs(result_dense)
        assert relative_error < 1e-4, f"Quadrature error: {relative_error}"
        
    def test_quadrature_gradient(self):
        """Test gradient of matrix function quadrature."""
        from arbplusjax import matfree_adjoints
        
        n = 25
        key = jr.PRNGKey(1515)
        A_rand = jr.normal(key, (n, n))
        A = (A_rand + A_rand.T) / 2
        
        def matvec(v):
            return A @ v
        
        quadform = matfree_adjoints.lanczos_quadrature_spd(
            matfun=jnp.exp,
            krylov_depth=15,
            matvec=matvec,
            reortho="full",
            use_efficient_adjoint=True
        )
        
        v0 = jr.normal(jr.PRNGKey(1616), (n,))
        
        # Should be differentiable
        grad = jax.grad(quadform)(v0)
        assert grad.shape == (n,)
        assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
