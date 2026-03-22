"""Examples demonstrating efficient matrix-free adjoints.

This module shows how to use the efficient Lanczos and Arnoldi implementations
with custom VJPs for backward differentiation through large-scale matrix operations.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def example_lanczos_basic():
    """Basic Lanczos tridiagonalization with efficient backward pass."""
    from arbplusjax import matfree_adjoints
    
    print("\n" + "="*70)
    print("Example 1: Basic Lanczos Tridiagonalization")
    print("="*70)
    
    # Create a symmetric matrix
    n = 100
    key = jr.PRNGKey(42)
    A_rand = jr.normal(key, (n, n))
    A = (A_rand + A_rand.T) / 2  # Make symmetric
    
    # Define matrix-vector product
    def matvec(v):
        return A @ v
    
    # Create Lanczos decomposition with efficient VJP
    lanczos = matfree_adjoints.lanczos_tridiag(
        matvec,
        krylov_depth=20,
        reortho="full",
        custom_vjp=True
    )
    
    # Initial vector
    v0 = jr.normal(jr.PRNGKey(123), (n,))
    
    # Forward pass
    (basis, (diags, offdiags)), (remainder, beta) = lanczos(v0)
    
    print(f"Matrix size: {n}×{n}")
    print(f"Krylov depth: 20")
    print(f"Basis shape: {basis.shape}")
    print(f"Diagonal values shape: {diags.shape}")
    print(f"Off-diagonal values shape: {offdiags.shape}")
    print(f"Remainder norm: {beta:.6e}")
    
    # Verify orthogonality
    orthogonality_error = jnp.linalg.norm(basis @ basis.T - jnp.eye(20))
    print(f"Basis orthogonality error: {orthogonality_error:.6e}")
    
    # Test backward pass
    def loss_fn(v):
        (basis, (diags, offdiags)), _ = lanczos(v)
        return jnp.sum(diags**2)  # Some differentiable loss
    
    grad = jax.grad(loss_fn)(v0)
    print(f"Gradient computed successfully, norm: {jnp.linalg.norm(grad):.6e}")
    
    return basis, diags, offdiags


def example_arnoldi_complex():
    """Arnoldi iteration for complex, non-symmetric matrices."""
    from arbplusjax import matfree_adjoints
    
    print("\n" + "="*70)
    print("Example 2: Arnoldi for Complex Non-Symmetric Matrix")
    print("="*70)
    
    # Create a complex non-symmetric matrix
    n = 80
    key = jr.PRNGKey(567)
    A_real = jr.normal(key, (n, n))
    A_imag = jr.normal(jr.PRNGKey(568), (n, n))
    A = A_real + 1j * A_imag
    
    def matvec(v):
        return A @ v
    
    # Create Arnoldi decomposition
    arnoldi = matfree_adjoints.arnoldi_hessenberg(
        matvec,
        krylov_depth=30,
        reortho="full",
        custom_vjp=True
    )
    
    # Initial vector
    v0 = jr.normal(jr.PRNGKey(234), (n,)) + 1j * jr.normal(jr.PRNGKey(235), (n,))
    
    # Forward pass
    Q, H, v_out, norm = arnoldi(v0)
    
    print(f"Matrix size: {n}×{n} (complex)")
    print(f"Krylov depth: 30")
    print(f"Q shape: {Q.shape}")
    print(f"H shape: {H.shape}")
    print(f"Q dtype: {Q.dtype}")
    
    # Verify orthogonality
    orthogonality_error = jnp.linalg.norm(Q.conj().T @ Q - jnp.eye(30))
    print(f"Q orthogonality error: {orthogonality_error:.6e}")
    
    # Verify Hessenberg structure (zeros below first subdiagonal)
    hessenberg_error = jnp.linalg.norm(jnp.tril(H, -2))
    print(f"Hessenberg structure error: {hessenberg_error:.6e}")
    
    # Test gradient computation
    def loss_fn(v):
        Q, H, _, _ = arnoldi(v)
        return jnp.sum(jnp.abs(jnp.diag(H))**2)
    
    grad = jax.grad(lambda v: jnp.real(loss_fn(v)))(v0)
    print(f"Gradient norm: {jnp.linalg.norm(grad):.6e}")
    
    return Q, H


def example_matrix_exponential_action():
    """Compute exp(A) @ v using Lanczos with efficient gradients."""
    from arbplusjax import matfree_adjoints
    
    print("\n" + "="*70)
    print("Example 3: Matrix Exponential Action with Gradients")
    print("="*70)
    
    n = 100
    key = jr.PRNGKey(999)
    A_rand = jr.normal(key, (n, n))
    A = (A_rand + A_rand.T) / 2  # Symmetric
    
    def matvec(v):
        return A @ v
    
    def expm_action(v):
        """Compute exp(A) @ v using Lanczos."""
        lanczos = matfree_adjoints.lanczos_tridiag(
            matvec,
            krylov_depth=30,
            reortho="full",
            custom_vjp=True
        )
        
        (basis, (diag, offdiag)), _ = lanczos(v)
        
        # Build tridiagonal matrix
        T = jnp.diag(diag) + jnp.diag(offdiag, 1) + jnp.diag(offdiag, -1)
        
        # Compute exp(T) using dense diagonalization
        eigvals, eigvecs = jnp.linalg.eigh(T)
        exp_T = eigvecs @ jnp.diag(jnp.exp(eigvals)) @ eigvecs.T
        
        # Project back: exp(A) v ≈ β₀ V exp(T) e₁
        e1 = jnp.zeros(len(diag))
        e1 = e1.at[0].set(1.0)
        beta0 = jnp.linalg.norm(v)
        
        return beta0 * (basis.T @ (exp_T @ e1))
    
    # Test vector
    v0 = jr.normal(jr.PRNGKey(111), (n,))
    
    # Compute action
    result = expm_action(v0)
    print(f"exp(A) @ v computed, result norm: {jnp.linalg.norm(result):.6e}")
    
    # Compare with dense computation (for small matrix)
    if n <= 100:
        eigvals, eigvecs = jnp.linalg.eigh(A)
        exp_a = eigvecs @ jnp.diag(jnp.exp(eigvals)) @ eigvecs.T
        result_dense = exp_a @ v0
        error = jnp.linalg.norm(result - result_dense) / jnp.linalg.norm(result_dense)
        print(f"Relative error vs dense: {error:.6e}")
    
    # Compute gradient
    def loss(v):
        return jnp.sum(expm_action(v)**2)
    
    grad = jax.grad(loss)(v0)
    print(f"Gradient of ||exp(A)v||² computed, norm: {jnp.linalg.norm(grad):.6e}")
    
    return result, grad


def example_hutchinson_trace():
    """Estimate tr(exp(A)) using Hutchinson + Lanczos."""
    from arbplusjax import matfree_adjoints
    
    print("\n" + "="*70)
    print("Example 4: Hutchinson Trace Estimation")
    print("="*70)
    
    n = 100
    key = jr.PRNGKey(777)
    A_rand = jr.normal(key, (n, n))
    A = (A_rand + A_rand.T) / 2
    
    def matvec(v):
        return A @ v
    
    # Define quadratic form: v^T exp(A) v
    quadform = matfree_adjoints.lanczos_quadrature_spd(
        matfun=jnp.exp,
        krylov_depth=30,
        matvec=matvec,
        reortho="full",
        use_efficient_adjoint=True
    )
    
    # Sample function (Gaussian random vectors)
    def sample_fun(key):
        num_samples = 10
        return jr.normal(key, (num_samples, n))
    
    # Create Hutchinson estimator
    trace_est = matfree_adjoints.hutchinson_trace_estimator(
        quadform,
        sample_fun,
        use_custom_vjp=True
    )
    
    # Estimate trace
    estimate = trace_est(jr.PRNGKey(888))
    print(f"tr(exp(A)) estimate: {estimate:.6f}")
    
    # Compare with dense computation (for verification)
    if n <= 100:
        eigvals = jnp.linalg.eigvalsh(A)
        trace_true = jnp.sum(jnp.exp(eigvals))
        error = jnp.abs(estimate - trace_true) / jnp.abs(trace_true)
        print(f"True trace: {trace_true:.6f}")
        print(f"Relative error: {error:.6e}")
    
    # The estimator is differentiable!
    # (though gradient w.r.t. random key is not meaningful in practice)
    
    return estimate


def example_cg_with_gradients():
    """CG solver with custom_linear_solve for efficient autodiff."""
    from arbplusjax import matfree_adjoints
    
    print("\n" + "="*70)
    print("Example 5: CG Solver with Efficient Gradients")
    print("="*70)
    
    n = 100
    key = jr.PRNGKey(333)
    
    # Create SPD matrix
    L = jr.normal(key, (n, n))
    A = L.T @ L + 0.1 * jnp.eye(n)  # Ensure positive definite
    
    def A_matvec(v):
        return A @ v
    
    # Right-hand side
    b = jr.normal(jr.PRNGKey(444), (n,))
    
    # Create CG solver
    cg_solve = matfree_adjoints.cg_fixed_iterations(num_matvecs=50)
    
    # Solve
    x, info = cg_solve(A_matvec, b)
    
    residual_norm = jnp.linalg.norm(A @ x - b)
    print(f"Solution computed in 50 iterations")
    print(f"Residual norm: {residual_norm:.6e}")
    print(f"Relative residual: {info['residual_rel'].mean():.6e}")
    
    # Compute gradient through the solve
    def loss_fn(b_val):
        x_val, _ = cg_solve(A_matvec, b_val)
        return jnp.sum(x_val**2)
    
    grad = jax.grad(loss_fn)(b)
    print(f"Gradient w.r.t. b computed, norm: {jnp.linalg.norm(grad):.6e}")
    
    # Verify against dense solve
    x_dense = jnp.linalg.solve(A, b)
    error = jnp.linalg.norm(x - x_dense) / jnp.linalg.norm(x_dense)
    print(f"Error vs dense solve: {error:.6e}")
    
    return x, info


def example_comparison_with_without_custom_vjp():
    """Compare performance with and without custom VJP."""
    from arbplusjax import matfree_adjoints
    import time
    
    print("\n" + "="*70)
    print("Example 6: Performance Comparison")
    print("="*70)
    
    n = 200
    krylov_depth = 50
    
    key = jr.PRNGKey(555)
    A_rand = jr.normal(key, (n, n))
    A = (A_rand + A_rand.T) / 2
    
    def matvec(v):
        return A @ v
    
    # With custom VJP
    lanczos_efficient = matfree_adjoints.lanczos_tridiag(
        matvec,
        krylov_depth=krylov_depth,
        reortho="full",
        custom_vjp=True
    )
    
    # Without custom VJP (standard autodiff)
    lanczos_standard = matfree_adjoints.lanczos_tridiag(
        matvec,
        krylov_depth=krylov_depth,
        reortho="full",
        custom_vjp=False
    )
    
    v0 = jr.normal(jr.PRNGKey(666), (n,))
    
    def loss_fn(lanczos_fn, v):
        (basis, (diags, offdiags)), _ = lanczos_fn(v)
        return jnp.sum(diags**2) + jnp.sum(offdiags**2)
    
    # JIT compile both versions
    grad_efficient = jax.jit(jax.grad(lambda v: loss_fn(lanczos_efficient, v)))
    grad_standard = jax.jit(jax.grad(lambda v: loss_fn(lanczos_standard, v)))
    
    # Warm-up
    _ = grad_efficient(v0)
    _ = grad_standard(v0)
    
    # Time efficient version
    start = time.time()
    for _ in range(10):
        g_eff = grad_efficient(v0).block_until_ready()
    time_efficient = (time.time() - start) / 10
    
    # Time standard version
    start = time.time()
    for _ in range(10):
        g_std = grad_standard(v0).block_until_ready()
    time_standard = (time.time() - start) / 10
    
    print(f"Matrix size: {n}×{n}")
    print(f"Krylov depth: {krylov_depth}")
    print(f"\nTiming (averaged over 10 runs):")
    print(f"  Efficient custom VJP: {time_efficient*1000:.2f} ms")
    print(f"  Standard autodiff:    {time_standard*1000:.2f} ms")
    print(f"  Speedup: {time_standard/time_efficient:.2f}x")
    
    # Verify they give the same result
    error = jnp.linalg.norm(g_eff - g_std) / jnp.linalg.norm(g_std)
    print(f"\nGradient agreement: {error:.6e} (relative error)")
    
    return time_efficient, time_standard


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print("MATRIX-FREE ADJOINTS EXAMPLES")
    print("Efficient Backward Differentiation for Krylov Methods")
    print("="*70)
    
    example_lanczos_basic()
    example_arnoldi_complex()
    example_matrix_exponential_action()
    example_hutchinson_trace()
    example_cg_with_gradients()
    example_comparison_with_without_custom_vjp()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_examples()
