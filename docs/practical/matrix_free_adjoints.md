# Matrix-Free Adjoints for Efficient Backward Differentiation

This guide covers the practical use of the matrix-free custom-VJP adjoint
surfaces for Lanczos, Arnoldi, quadrature, and trace estimation.

It documents efficient backward passes built around Krylov bases reused from the
forward pass, following:

> **Gradients of functions of large matrices.**  
> _Nicholas Krämer, Pablo Moreno-Muñoz, Hrittik Roy, Søren Hauberg._  
> 2024. arXiv:2405.17277.  
> https://arxiv.org/abs/2405.17277

## Why This Exists

Traditional differentiation through Krylov methods either recomputes the full
decomposition in the backward pass or relies on loop-level autodiff with high
memory cost. The matrix-free adjoint surfaces reuse the Krylov basis from the
forward pass so backward differentiation remains practical for large operators.

## Main Surfaces

### Lanczos Tridiagonalization

For symmetric/Hermitian operators:

```python
from arbplusjax import matfree_adjoints

def matvec(v, *params):
    return A @ v

lanczos = matfree_adjoints.lanczos_tridiag(
    matvec,
    krylov_depth=50,
    reortho="full",
    custom_vjp=True,
)

(basis, (diags, offdiags)), remainder = lanczos(v0)
grad_fn = jax.grad(lambda v: lanczos(v)[0][1][0].sum())
gradient = grad_fn(v0)
```

### Arnoldi Hessenberg

For general real or complex operators:

```python
from arbplusjax import matfree_adjoints

def matvec(v, *params):
    return A @ v

arnoldi = matfree_adjoints.arnoldi_hessenberg(
    matvec,
    krylov_depth=50,
    reortho="full",
    custom_vjp=True,
)

Q, H, v_out, norm_out = arnoldi(v0)
```

### Hutchinson Trace Estimation

```python
from arbplusjax import matfree_adjoints
import jax.random as jr

def integrand(v, *params):
    return jnp.dot(v, f_A @ v)

def sample_fun(key):
    return jr.normal(key, shape=(num_samples, n))

trace_est = matfree_adjoints.hutchinson_trace_estimator(
    integrand,
    sample_fun,
    use_custom_vjp=True,
)

estimate = trace_est(jr.PRNGKey(0))
```

### Lanczos Quadrature

```python
from arbplusjax import matfree_adjoints

def f(x):
    return jnp.exp(x)

quadform = matfree_adjoints.lanczos_quadrature_spd(
    matfun=f,
    krylov_depth=50,
    matvec=your_matvec,
    reortho="full",
    use_efficient_adjoint=True,
)

result = quadform(v0)
```

### Fixed-Iteration CG

```python
from arbplusjax import matfree_adjoints

cg_solve = matfree_adjoints.cg_fixed_iterations(num_matvecs=100)
x, info = cg_solve(A, b)
```

## Integration With `jrb_mat` And `jcb_mat`

These adjoint helpers are designed to sit alongside the matrix-free operator
stack rather than replace it. The usual pattern is:

1. define a midpoint/operator `matvec`
2. build a Lanczos or Arnoldi decomposition from `matfree_adjoints`
3. use that decomposition inside a matrix function, quadrature, or estimator
4. differentiate through the assembled operator pipeline

## Reorthogonalization

- `reortho="none"`: lower per-iteration cost, weaker stability
- `reortho="full"`: stronger orthogonality and the recommended default

## Performance Notes

- custom VJPs avoid backward recomputation of full Krylov loops
- memory still scales with the stored basis
- full reorthogonalization is more stable but more expensive
- large-scale use should be benchmarked with the matrix-free benchmark suite

## Related Docs

- [matrix_free.md](/home/phili/projects/arbplusJAX/docs/practical/matrix_free.md)
- [matrix_stack.md](/home/phili/projects/arbplusJAX/docs/implementation/matrix_stack.md)
- [slepc_inspired_jax.md](/home/phili/projects/arbplusJAX/docs/implementation/slepc_inspired_jax.md)
