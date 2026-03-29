Last updated: 2026-03-26T00:00:00Z

# API Cold-Path Audit

## Scope
This audit classifies the imports reachable from
[`src/arbplusjax/api.py`](/src/arbplusjax/api.py)
into four buckets:

- required on cold start
- point-only
- interval/mode-only
- benchmark/docs-only

The measurement basis was a clean subprocess import of:

```python
from arbplusjax import api
```

and clean subprocess imports of the major direct `api.py` dependencies.

## Current direct `api.py` imports

`api.py` directly imports:

- [`acb_core`](/src/arbplusjax/acb_core.py)
- [`double_interval`](/src/arbplusjax/double_interval.py)
- [`kernel_helpers`](/src/arbplusjax/kernel_helpers.py)
- [`point_wrappers`](/src/arbplusjax/point_wrappers.py)
- [`public_metadata`](/src/arbplusjax/public_metadata.py)
- [`special.tail_acceleration`](/src/arbplusjax/special/tail_acceleration/__init__.py)
- [`special.bessel`](/src/arbplusjax/special/bessel/__init__.py)
- [`special.gamma`](/src/arbplusjax/special/gamma/__init__.py)
- [`special.laplace_bessel`](/src/arbplusjax/special/laplace_bessel/__init__.py)

## Observed cold-start import set

Observed during `from arbplusjax import api`:

- `arbplusjax.acb_core`
- `arbplusjax.acb_dirichlet`
- `arbplusjax.acb_elliptic`
- `arbplusjax.acb_modular`
- `arbplusjax.api`
- `arbplusjax.arb_core`
- `arbplusjax.ball_wrappers`
- `arbplusjax.barnesg`
- `arbplusjax.bessel_kernels`
- `arbplusjax.checks`
- `arbplusjax.coeffs`
- `arbplusjax.core_wrappers`
- `arbplusjax.double_gamma`
- `arbplusjax.double_interval`
- `arbplusjax.elementary`
- `arbplusjax.hypgeom`
- `arbplusjax.jax_precision`
- `arbplusjax.kernel_helpers`
- `arbplusjax.lazy_jit`
- `arbplusjax.mat_common`
- `arbplusjax.point_wrappers`
- `arbplusjax.precision`
- `arbplusjax.public_metadata`
- `arbplusjax.sampling_helpers`
- `arbplusjax.series_missing_impl`
- `arbplusjax.series_utils`
- `arbplusjax.special`
- `arbplusjax.special.bessel`
- `arbplusjax.special.bessel.*`
- `arbplusjax.special.gamma`
- `arbplusjax.special.gamma.barnes_double_gamma_ifj`
- `arbplusjax.special.laplace_bessel`
- `arbplusjax.special.laplace_bessel.*`
- `arbplusjax.special.tail_acceleration`
- `arbplusjax.special.tail_acceleration.*`
- `arbplusjax.wrappers_common`

## Classification

### Required On Cold Start

These are legitimate cold-start dependencies for the current `api.py` design.

- [`acb_core.py`](/src/arbplusjax/acb_core.py)
- [`double_interval.py`](/src/arbplusjax/double_interval.py)
- [`kernel_helpers.py`](/src/arbplusjax/kernel_helpers.py)
- [`checks.py`](/src/arbplusjax/checks.py)
- [`arb_core.py`](/src/arbplusjax/arb_core.py)
- [`elementary.py`](/src/arbplusjax/elementary.py)
- [`lazy_jit.py`](/src/arbplusjax/lazy_jit.py)
- [`precision.py`](/src/arbplusjax/precision.py)
- [`jax_precision.py`](/src/arbplusjax/jax_precision.py)

These are also currently cold-start imports because `api.py` exposes their public functions directly:

- [`special/bessel/__init__.py`](/src/arbplusjax/special/bessel/__init__.py)
- [`special/gamma/__init__.py`](/src/arbplusjax/special/gamma/__init__.py)
- [`special/laplace_bessel/__init__.py`](/src/arbplusjax/special/laplace_bessel/__init__.py)
- [`special/tail_acceleration/__init__.py`](/src/arbplusjax/special/tail_acceleration/__init__.py)

These may still deserve later lazy-boundary work, but under the current API contract they are part of the cold path.

### Point-Only

These modules are semantically point-only or point-heavy. They should not all be on the cold path just to import `api`.

- [`point_wrappers.py`](/src/arbplusjax/point_wrappers.py)
- [`hypgeom.py`](/src/arbplusjax/hypgeom.py)
- [`double_gamma.py`](/src/arbplusjax/double_gamma.py)
- [`acb_dirichlet.py`](/src/arbplusjax/acb_dirichlet.py)
- [`acb_elliptic.py`](/src/arbplusjax/acb_elliptic.py)
- [`acb_modular.py`](/src/arbplusjax/acb_modular.py)
- [`barnesg.py`](/src/arbplusjax/barnesg.py)
- [`bessel_kernels.py`](/src/arbplusjax/bessel_kernels.py)
- [`coeffs.py`](/src/arbplusjax/coeffs.py)
- [`core_wrappers.py`](/src/arbplusjax/core_wrappers.py)
- [`mat_common.py`](/src/arbplusjax/mat_common.py)
- [`sampling_helpers.py`](/src/arbplusjax/sampling_helpers.py)
- [`series_missing_impl.py`](/src/arbplusjax/series_missing_impl.py)
- [`series_utils.py`](/src/arbplusjax/series_utils.py)
- [`wrappers_common.py`](/src/arbplusjax/wrappers_common.py)
- [`special/gamma/barnes_double_gamma_ifj.py`](/src/arbplusjax/special/gamma/barnes_double_gamma_ifj.py)

Observed current leak:

- importing [`point_wrappers.py`](/src/arbplusjax/point_wrappers.py) alone imports `hypgeom`, `double_gamma`, `acb_dirichlet`, `acb_elliptic`, `acb_modular`, and the IFJ Barnes path.
- importing [`special/laplace_bessel/__init__.py`](/src/arbplusjax/special/laplace_bessel/__init__.py) also reaches `point_wrappers`, which reaches `hypgeom`.

This is the main remaining cold-path violation.

### Interval/Mode-Only

These families are currently staying off the `api` cold path and are already behind lazy resolution in `api.py`.

- [`baseline_wrappers.py`](/src/arbplusjax/baseline_wrappers.py)
- [`hypgeom_wrappers.py`](/src/arbplusjax/hypgeom_wrappers.py)
- [`boost_hypgeom.py`](/src/arbplusjax/boost_hypgeom.py)
- [`arb_calc.py`](/src/arbplusjax/arb_calc.py)
- [`acb_calc.py`](/src/arbplusjax/acb_calc.py)
- [`arb_mat.py`](/src/arbplusjax/arb_mat.py)
- [`acb_mat.py`](/src/arbplusjax/acb_mat.py)
- [`mat_wrappers.py`](/src/arbplusjax/mat_wrappers.py)
- [`srb_mat.py`](/src/arbplusjax/srb_mat.py)
- [`srb_block_mat.py`](/src/arbplusjax/srb_block_mat.py)
- [`srb_vblock_mat.py`](/src/arbplusjax/srb_vblock_mat.py)
- [`scb_mat.py`](/src/arbplusjax/scb_mat.py)
- [`scb_block_mat.py`](/src/arbplusjax/scb_block_mat.py)
- [`scb_vblock_mat.py`](/src/arbplusjax/scb_vblock_mat.py)

Verification snapshot from a clean `api` import:

- `arbplusjax.baseline_wrappers`: not imported
- `arbplusjax.hypgeom_wrappers`: not imported
- `arbplusjax.boost_hypgeom`: not imported
- `arbplusjax.arb_calc`: not imported
- `arbplusjax.acb_calc`: not imported
- `arbplusjax.arb_mat`: not imported
- `arbplusjax.acb_mat`: not imported
- `arbplusjax.mat_wrappers`: not imported
- `arbplusjax.srb_mat`: not imported
- `arbplusjax.scb_mat`: not imported
- `arbplusjax.cubesselk`: not imported

### Benchmark/Docs-Only

There are no benchmark modules imported during `api` cold start.

However, one docs/introspection-oriented dependency is still on the cold path:

- [`public_metadata.py`](/src/arbplusjax/public_metadata.py)

This is not benchmark code, but it is metadata/rendering support rather than numeric runtime support. It belongs in a future “docs/introspection-only” split so that metadata reporting does not widen runtime startup.

## Findings

### 1. `point_wrappers` is still a mixed surface

`api.py` imports [`point_wrappers.py`](/src/arbplusjax/point_wrappers.py) directly, and that single import still pulls in:

- hypergeom point kernels
- Barnes/double-gamma point kernels
- dirichlet/elliptic/modular point families
- auxiliary point support modules

This means the current point surface is not yet minimal-load. The next split should separate:

- core point wrappers needed for generic cold-start API
- family-specific point wrappers loaded on demand

### 2. The remaining `hypgeom` cold-start edge is indirect

The current evidence does not point to `hypgeom_wrappers` or `boost_hypgeom` anymore. Those stay lazy.

The remaining `arbplusjax.hypgeom` import is reaching cold start through:

- [`point_wrappers.py`](/src/arbplusjax/point_wrappers.py)
- and secondarily through [`special/laplace_bessel/__init__.py`](/src/arbplusjax/special/laplace_bessel/__init__.py), which imports point wrappers

### 3. `double_gamma` is also still on the cold path

Even after lazy registration in `api.py`, [`double_gamma.py`](/src/arbplusjax/double_gamma.py) still arrives through the current point-wrapper graph.

This should be treated as the same architectural problem as hypergeom: the point surface still mixes too many specialized families into one import boundary.

### 4. Interval/mode lazy loading is materially better than before

The main interval/mode families now stay out of the cold path:

- interval wrappers
- matrix interval wrappers
- calc wrappers
- sparse matrix interval layers
- boost/hypgeom mode wrappers

That means the next tranche should focus on point-surface decomposition, not interval-surface cleanup.

## Recommended Next Tranche

1. Split [`point_wrappers.py`](/src/arbplusjax/point_wrappers.py) into:
   - core point wrappers
   - hypergeom point wrappers
   - double-gamma/Barnes point wrappers
   - modular/dirichlet/elliptic point wrappers

2. Change [`api.py`](/src/arbplusjax/api.py) so the cold path imports only the core point wrapper subset.

3. Keep family-specific point registries behind lazy module descriptors, the same way interval/mode registries are already handled.

4. Move [`public_metadata.py`](/src/arbplusjax/public_metadata.py) toward a docs/introspection boundary so metadata rendering is not part of the default numeric startup path.

5. Add a follow-up import-boundary probe specifically for:
   - `api import -> point_wrappers core only`
   - `first hypergeom point call`
   - `first double_gamma point call`
   - `first dirichlet/modular/elliptic point call`
