Last updated: 2026-03-18T01:30:00Z

# NUFFT

This module provides a small JAX-native 1D/2D/3D NUFFT layer inspired by DUCC's functionality, but implemented entirely inside arbPlusJAX without any DUCC runtime dependency.

## Current subset

Implemented:

- `nufft_type1_direct(points, values, n_modes)`
- `nufft_type2_direct(points, modes)`
- `nufft_type1(..., method="direct" | "lanczos" | "auto")`
- `nufft_type2(..., method="direct" | "lanczos" | "auto")`
- `nufft_type1_nd_direct(points, values, mode_shape)`
- `nufft_type2_nd_direct(points, modes)`
- `nufft_type1_nd(...)`
- `nufft_type2_nd(...)`
- `nufft_type1_2d(...)`, `nufft_type2_2d(...)`
- `nufft_type1_3d(...)`, `nufft_type2_3d(...)`
- `nufft_type1_with_diagnostics(...)`
- `nufft_type2_with_diagnostics(...)`
- `nufft_type1_nd_with_diagnostics(...)`
- `nufft_type2_nd_with_diagnostics(...)`
- `nufft_type1_cached_prepare(...)`, `nufft_type2_cached_prepare(...)`
- `nufft_type1_nd_cached_prepare(...)`, `nufft_type2_nd_cached_prepare(...)`
- `nufft_type1_cached_apply(...)`, `nufft_type2_cached_apply(...)`
- `nufft_type1_cached_apply_batch_fixed(...)`, `nufft_type2_cached_apply_batch_fixed(...)`
- `nufft_type1_batch_fixed(...)`, `nufft_type2_batch_fixed(...)`
- `nufft_good_size()`
- `nufft_good_shape()`

The current accelerated path is an oversampled Lanczos-grid interpolation scheme in 1D, 2D, and 3D. It is an approximate NUFFT. The direct path remains the exact reference implementation and the default fallback for small problems.

## Conventions

Points are interpreted on the unit torus:
\[
x_j \in [0,1)^{d} \pmod 1.
\]

Type-1 computes the positive-index Fourier modes
\[
F_k = \sum_j c_j e^{-2\pi i \langle k, x_j \rangle}, \qquad k \in \prod_{\ell=1}^d \{0,\dots,N_\ell-1\}.
\]

Type-2 evaluates
\[
f(x_j) = \sum_k F_k e^{2\pi i \langle k, x_j \rangle}.
\]

This matches the repo's DFT sign convention and makes the direct NUFFT agree with `dft.dft()`, `dft.dft2()`, and `dft.dft3()` on uniform tensor grids.

## Accelerated path

The accelerated path:

1. chooses an oversampled smooth grid shape via `nufft_good_shape()`
2. pads low modes onto that grid
3. uses FFTs for the uniform-grid stage
4. interpolates or spreads with a truncated Lanczos stencil

For fixed-point workloads, the expensive point normalization and Lanczos stencil construction can now be hoisted into a prepared plan. The cached apply path is the same numerical kernel as the ordinary runtime path; it just reuses the precomputed stencils or direct mode grid.

The type-1 and type-2 Lanczos paths are implemented as adjoint pairs in 1D, 2D, and 3D, so the approximate forward and adjoint operators satisfy the discrete inner-product identity exactly up to floating-point roundoff.

## Diagnostics

`*_with_diagnostics()` reports:

- chosen method (`direct` or `lanczos`)
- `n_points`
- `mode_shape`
- `ndim`
- when applicable: `grid_shape`, `oversamp`, `kernel_width`
- 1D wrappers also report `n_modes` and `grid_size` for compatibility with the older 1D-only diagnostics shape

This is intended to make downstream routing decisions inspectable without pulling in any experimental JAX surface.

The shared conversions, dtype constants, smooth-size helper, and Lanczos stencil builders now live in `transform_common`, so DFT and NUFFT follow the same JAX array conversion rules.

## Current limits

Not yet implemented:

- type-3 NUFFT
- min-max or Gaussian kernel variants
- rigorous interval NUFFT
- centered-frequency conventions

So this is a stable starter subset, not a replacement for a full DUCC-class NUFFT stack.
