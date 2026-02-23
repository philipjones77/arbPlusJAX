# JAX Setup and Implementation Conventions

This document describes how the JAX implementation is structured and the patterns used to keep the API consistent.

## Package layout

- `src/arbplusjax/`: primary JAX implementation.
- `tests/`: chassis and parity tests.
- `tools/`: parity/benchmark scripts.
- `docs/`: theory + implementation notes.

## Core design rules

1. **JAX-first**: The JAX implementation is the source of truth. C code is used only for parity/benchmarking.
2. **Interval semantics**: Real intervals are stored as `[lo, hi]` arrays. Complex boxes are `[re_lo, re_hi, im_lo, im_hi]`.
3. **Outward rounding**: Precision-aware APIs accept `prec_bits` and apply outward rounding after computation.
4. **Safe fallbacks**: If a computation produces non-finite values, return a full interval/box.
5. **Consistency**: Each function has scalar, batch, precision, and jit variants where applicable.

## Data types

- Real interval: `double_interval.interval(lo, hi)` → shape `(2,)`.
- Complex box: `acb_box(real_interval, imag_interval)` → shape `(4,)`.
- All kernels operate on `float64` (JAX x64 enabled).

## Naming conventions

For a kernel `foo`:

- Scalar: `arb_foo(...)` or `acb_foo(...)`
- Batch: `arb_foo_batch(...)` or `acb_foo_batch(...)`
- Precision: `arb_foo_prec(...)` or `acb_foo_prec(...)`
- Batch + precision: `arb_foo_batch_prec(...)`
- JIT batch: `arb_foo_batch_jit(...)`
- JIT batch + precision: `arb_foo_batch_prec_jit(...)`

This keeps call sites consistent and predictable.

## Mode selection (baseline vs rigorous vs adaptive)

For `exp`, `log`, `sin`, `gamma` there are **mode-selectable** wrappers:

- Real: `baseline_wrappers.arb_exp_mp`, `arb_log_mp`, `arb_sin_mp`, `arb_gamma_mp`
- Complex: `baseline_wrappers.acb_exp_mp`, `acb_log_mp`, `acb_sin_mp`, `acb_gamma_mp`

Each accepts:

- `mode="baseline" | "rigorous" | "adaptive"`
- `prec_bits` or `dps`

Example:

```python
from arbplusjax import baseline_wrappers as bw

y = bw.arb_exp_mp(x, mode="baseline", dps=50)
y = bw.arb_exp_mp(x, mode="rigorous", prec_bits=80)
y = bw.arb_exp_mp(x, mode="adaptive", dps=80)
```

## Implementation pattern (real interval)

Example structure:

```python
@jax.jit
def arb_example(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    m = di.midpoint(x)
    v = some_jax_fn(m)
    out = di.interval(di._below(v), di._above(v))
    return jnp.where(jnp.isfinite(v), out, di.interval(-jnp.inf, jnp.inf))
```

Batch:

```python
def arb_example_batch(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(arb_example)(x)
```

Precision:

```python
def arb_example_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_example(x), prec_bits)
```

## Implementation pattern (complex box)

Example structure:

```python
@jax.jit
def acb_example(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    z = acb_midpoint(box)  # complex midpoint
    v = some_complex_fn(z)
    out = _acb_from_complex(v)
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    return jnp.where(finite[..., None], out, _full_box_like(box))
```

## Autodiff

- Gradients are evaluated on midpoint-based kernels.
- Batch and jit functions are written to remain differentiable where the analytic function is smooth.

## Error handling

- Non-finite results force full intervals/boxes.
- When function domains are restricted, invalid inputs are mapped to full intervals.

## Parity and benchmarks

- Parity tests are gated by `arbplusjax_RUN_PARITY=1`.
- C libs are built in the Arb workspace and loaded by tools in `tools/`.
- Benchmarks are JAX-only unless explicitly comparing to C.

## Where to look

- Real interval primitives: `src/arbplusjax/double_interval.py`
- Complex box primitives: `src/arbplusjax/acb_core.py`
- Hypergeometric scaffold: `src/arbplusjax/hypgeom.py`
- Runtime facade: `src/arbplusjax/runtime.py`
