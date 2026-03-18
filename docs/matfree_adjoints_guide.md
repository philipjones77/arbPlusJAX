# Matrix-Free Adjoints for Efficient Backward Differentiation

This module implements efficient custom VJPs (Vector-Jacobian Products) for Lanczos and Arnoldi iterations based on the research paper:

> **Gradients of functions of large matrices.**  
> _Nicholas Krämer, Pablo Moreno-Muñoz, Hrittik Roy, Søren Hauberg._  
> 2024. arXiv:2405.17277.  
> https://arxiv.org/abs/2405.17277

## Key Innovation

Traditional approaches to differentiating through Krylov methods either:
- Recompute the entire decomposition in the backward pass (expensive)
- Use automatic differentiation through the loop (memory-intensive)

**This implementation reuses the Krylov basis from the forward pass** to compute gradients efficiently, making backward differentiation through large-scale matrix operations practical.

## Features

### 1. Efficient Lanczos Tridiagonalization
For **symmetric/Hermitian matrices** A:
```python
from arbplusjax import matfree_adjoints

# Define your matrix-vector product
def matvec(v, *params):
    return A @ v  # or any implicit operator

# Create Lanczos decomposition function
lanczos = matfree_adjoints.lanczos_tridiag(
    matvec, 
    krylov_depth=50, 
    reortho="full",
    custom_vjp=True  # Enable efficient backward pass
)

# Compute decomposition
(basis, (diags, offdiags)), remainder = lanczos(v0)

# Now you can differentiate through this!
grad_fn = jax.grad(lambda v: lanczos(v)[0][1][0].sum())
gradient = grad_fn(v0)
```

### 2. Efficient Arnoldi Hessenberg Decomposition
For **general (non-symmetric) matrices** A:
```python
from arbplusjax import matfree_adjoints

def matvec(v, *params):
    return A @ v

arnoldi = matfree_adjoints.arnoldi_hessenberg(
    matvec,
    krylov_depth=50,
    reortho="full",
    custom_vjp=True
)

Q, H, v_out, norm_out = arnoldi(v0)

# Works with complex matrices too!
def complex_matvec(v, *params):
    return complex_A @ v

arnoldi_complex = matfree_adjoints.arnoldi_hessenberg(
    complex_matvec,
    krylov_depth=50,
    reortho="full",
    custom_vjp=True
)
```

### 3. Hutchinson Trace Estimator
Estimate tr(f(A)) using Monte Carlo with efficient variance reduction:
```python
from arbplusjax import matfree_adjoints
import jax.random as jr

# Define integrand: v -> v^T f(A) v
def integrand(v, *params):
    # Your matrix function application
    return jnp.dot(v, f_A @ v)

# Sample function (typically Rademacher or Gaussian)
def sample_fun(key):
    return jr.normal(key, shape=(num_samples, n))

# Create estimator
trace_est = matfree_adjoints.hutchinson_trace_estimator(
    integrand,
    sample_fun,
    use_custom_vjp=True  # Different samples for forward/backward
)

# Estimate trace
key = jr.PRNGKey(0)
estimate = trace_est(key)

# Differentiate through the estimator!
grad = jax.grad(lambda k: trace_est(k))(key)
```

### 4. Lanczos Quadrature for Matrix Functions
Compute v^T f(A) v efficiently:
```python
from arbplusjax import matfree_adjoints

# Define your matrix function (applied to eigenvalues)
def f(x):
    return jnp.exp(x)  # or jnp.log, sqrt, etc.

quadform = matfree_adjoints.lanczos_quadrature_spd(
    matfun=f,
    krylov_depth=50,
    matvec=your_matvec,
    reortho="full",
    use_efficient_adjoint=True
)

# Compute quadratic form
result = quadform(v0)

# Combine with Hutchinson for trace estimation
def integrand(v):
    return quadform(v)

trace_estimator = matfree_adjoints.hutchinson_trace_estimator(
    integrand,
    sample_fun
)

# Now tr(f(A)) ≈ E[v^T f(A) v] where v ~ N(0, I)
trace_estimate = trace_estimator(key)
```

### 5. Enhanced CG Solver
Conjugate gradient with `custom_linear_solve` for better autodiff:
```python
from arbplusjax import matfree_adjoints

cg_solve = matfree_adjoints.cg_fixed_iterations(num_matvecs=100)

def A(x):
    return your_matrix @ x

b = jnp.ones(n)
x, info = cg_solve(A, b)

# Differentiate through the solve!
grad = jax.grad(lambda b_val: cg_solve(A, b_val)[0].sum())(b)
```

## Integration with arbplusJAX Matrix-Free Modules

The new efficient adjoints can be used with existing `jrb_mat` and `jcb_mat` modules:

### Real Symmetric Matrices (`jrb_mat`)
```python
from arbplusjax import jrb_mat, matfree_adjoints
import jax.numpy as jnp

# Create interval matrix
A_interval = jrb_mat.jrb_mat_as_interval_matrix(A)

# Define matvec using jrb_mat operations
def matvec(v):
    v_interval = jrb_mat.jrb_mat_as_interval_vector(v)
    result = jrb_mat.jrb_mat_matvec_point(A_interval, v_interval)
    return jrb_mat._jrb_mid_vector(result)

# Use efficient Lanczos
lanczos = matfree_adjoints.lanczos_tridiag(
    matvec,
    50,
    reortho="full",
    custom_vjp=True
)

# Compute matrix exponential action with efficient backward pass
def expm_action(v):
    (basis, (diag, offdiag)), _ = lanczos(v)
    # Build tridiagonal matrix
    T = jnp.diag(diag) + jnp.diag(offdiag, 1) + jnp.diag(offdiag, -1)
    # Compute exp(T) via dense eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(T)
    exp_T = eigvecs @ jnp.diag(jnp.exp(eigvals)) @ eigvecs.T
    # Project back
    e1 = jnp.zeros(len(diag))
    e1 = e1.at[0].set(1.0)
    return jnp.linalg.norm(v) * (basis.T @ (exp_T @ e1))

result = expm_action(v0)
grad = jax.grad(lambda v: jnp.sum(expm_action(v)))(v0)
```

### Complex General Matrices (`jcb_mat`)
```python
from arbplusjax import jcb_mat, matfree_adjoints

# Define matvec for complex matrices
def matvec(v):
    v_box = jcb_mat.jcb_mat_as_box_vector(v)
    result = jcb_mat.jcb_mat_matvec_point(A_box, v_box)
    return jcb_mat._jcb_mid_vector(result)

# Use efficient Arnoldi
arnoldi = matfree_adjoints.arnoldi_hessenberg(
    matvec,
    50,
    reortho="full",
    custom_vjp=True
)

Q, H, v_out, norm = arnoldi(v0)
```

## Reorthogonalization Strategies

### `reortho="none"`
- Faster iteration
- May lose orthogonality for ill-conditioned matrices
- Use for well-conditioned problems or when speed is critical

### `reortho="full"`
- Maintains orthogonality (Gram-Schmidt on entire basis)
- More stable for ill-conditioned matrices
- Slightly more expensive per iteration
- **Recommended default**

## Performance Considerations

### Backward Pass Efficiency
With traditional autodiff through a 50-iteration Lanczos:
- **Memory**: O(50 × n) to store all intermediate states
- **Computation**: Recompute 50 iterations or use stored checkpoints

With efficient custom VJP:
- **Memory**: O(krylov_depth × n) for the basis (same as forward)
- **Computation**: Single backward sweep using the stored basis
- **Speedup**: ~2-5× for typical problems

### When to Use
**Use efficient adjoints when:**
- Krylov depth > 10
- Differentiating through matrix functions (exp, log, sqrt, etc.)
- Training models that use large-scale linear algebra
- Computing gradients of quadratic forms or traces

**Use standard autodiff when:**
- Very small Krylov depth (< 10)
- No need for gradients
- Debugging (simpler to trace)

## Mathematical Background

### Lanczos Adjoint Formula
Given the forward pass:
```
A V = V T + β v_{k+1} e_k^T
```

The adjoint efficiently computes gradients by observing that the cotangent vectors satisfy a backward recurrence that reuses V and T, avoiding recomputation of the decomposition.

### Computational Complexity
- **Forward pass**: O(k × matvec_cost)
- **Backward pass (traditional)**: O(k × matvec_cost) + O(k² n)
- **Backward pass (efficient)**: O(k × matvec_cost) only

where k = krylov_depth and n = problem dimension.

## Citation

If you use these efficient adjoints in your research, please cite the original paper:

```bibtex
@article{kraemer2024gradients,
    title={Gradients of functions of large matrices},
    author={Kr\"amer, Nicholas and Moreno-Mu\~noz, Pablo and Roy, Hrittik and Hauberg S\o{}ren},
    journal={arXiv preprint arXiv:2405.17277},
    year={2024}
}
```

## See Also

- `jrb_mat`: Real interval matrix-free operations
- `jcb_mat`: Complex box matrix-free operations
- `iterative_solvers`: Pure JAX iterative linear solvers
- Paper: https://arxiv.org/abs/2405.17277
- matfree library: https://github.com/pnkraemer/matfree
