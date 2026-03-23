Last updated: 2026-03-18T01:30:00Z

# DFT and Convolution

This module implements discrete Fourier transforms and circular convolution over complex numbers and complex interval boxes. The basic kernels operate on midpoints (`complex128`). Rigorous mode uses explicit interval twiddle factors and interval arithmetic to propagate bounds.

## Current implementation status

The current module now contains a real 1D/2D/3D complex FFT subset, but it is still narrower than the full FLINT `acb_dft` surface.

Current behavior:

- midpoint power-of-two transforms use `jnp.fft.fft` and `jnp.fft.ifft`
- arbitrary-length midpoint transforms use a repo-owned Bluestein kernel built on top of public JAX FFTs
- multidimensional midpoint transforms (`dft_nd`, `dft2`, `dft3`) apply the 1D kernels axis-by-axis, so awkward per-axis lengths inherit the same 1D behavior
- `make_dft_precomp(length, inverse=...)` constructs the reusable chirp/kernel payload used by the `*_precomp` entry points
- `dft_matvec_cached_prepare_point(length, inverse=...)` and `dft_matvec_cached_apply_point(plan, x)` expose that reusable payload as a stable point-operator plan
- `dft_matvec_cached_prepare_basic(length, inverse=...)` and `dft_matvec_cached_apply_basic(plan, x)` do the same for box-valued basic mode
- `dft_matvec_batch_fixed_*` and `dft_matvec_cached_apply_batch_fixed_*` provide explicit batch operator surfaces for point and basic vectors
- `dft_good_size()` provides a small smooth-size helper for Bluestein padding
- basic `acb_dft_*` paths now evaluate the midpoint FFT first and then propagate box half-widths deterministically using absolute twiddle bounds
- basic `acb_dft_nd`, `acb_dft2`, and `acb_dft3` are available for box-valued 2D/3D arrays
- rigorous interval transforms still use explicit `O(n^2)` twiddle accumulation
- shared casting, axis canonicalization, and smooth-size helpers now live in `transform_common` so DFT and NUFFT obey the same JAX conversion rules

Examples:

- `acb_dft_bluestein()` is now a real midpoint Bluestein path for non-power-of-two sizes
- `acb_dft_bluestein_precomp()` consumes the repo-owned precomp payload from `make_dft_precomp()`
- `acb_dft_precomp()` and `acb_dft_rad2_precomp()` use the same payload for awkward lengths
- `acb_dft_convol_mullow()` currently aliases the naive convolution path

So the module currently offers:

- public naming parity for much of the `acb_dft` surface
- a real midpoint fast path for both power-of-two and arbitrary 1D sizes
- 2D and 3D complex FFT/IDFT entry points built from the 1D kernels
- reusable precomp support for the Bluestein-based subset
- prepared matvec and batched matvec entry points for point/basic vector transforms
- rigorous interval containment via explicit twiddle bounds

It does not yet offer:

- real/complex `r2c` / `c2r` transforms
- DCT / DST / Hartley transforms
- axis-wise convolution helpers like DUCC's `convolve_axis`
- a high-performance rigorous FFT algorithm

## Core formulas

For \(x \in \mathbb{C}^n\), the DFT and inverse DFT are:
\[
X_k = \sum_{t=0}^{n-1} x_t \, \omega^{kt},\qquad \omega = e^{-2\pi i / n}
\]
\[
x_t = \frac{1}{n}\sum_{k=0}^{n-1} X_k \, \omega^{-kt}
\]

Circular convolution:
\[
(f * g)_k = \sum_{t=0}^{n-1} f_t \, g_{k-t \bmod n}
\]

DFT-based convolution:
\[
f * g = \mathrm{IDFT}(\mathrm{DFT}(f)\odot \mathrm{DFT}(g))
\]

## Interval twiddle factors

Rigorous mode constructs interval twiddle factors:
\[
\omega^{kt} = \cos(\theta_{kt}) + i \sin(\theta_{kt}),\quad \theta_{kt} = \pm 2\pi kt/n
\]
and replaces each scalar with an interval:
\[
\cos(\theta) \mapsto [\mathrm{below}(\cos\theta),\mathrm{above}(\cos\theta)],
\]
\[
\sin(\theta) \mapsto [\mathrm{below}(\sin\theta),\mathrm{above}(\sin\theta)].
\]

The DFT sum is then evaluated using interval arithmetic for all adds and multiplies:
```
acb_dft_naive_rigorous
```
This guarantees containment in the returned box under the `double_interval` outward rounding model.

## Rigorous convolution

Rigorous convolution uses:
- interval DFT (`acb_dft_*_rigorous`),
- interval elementwise multiply (`acb_mul_vec_rigorous`),
- interval inverse DFT (`acb_idft_*_rigorous`).

This preserves containment for convolution outputs and aligns with Arb’s philosophy of interval propagation.

## Basic `acb` width propagation

For the basic `acb_dft_*` path, the midpoint is transformed by the midpoint FFT kernel and the real/imaginary half-widths are propagated by
\[
r^{(\Re)}_k = \sum_t \left(|\cos \theta_{kt}|\, e^{(\Re)}_t + |\sin \theta_{kt}|\, e^{(\Im)}_t\right),
\]
\[
r^{(\Im)}_k = \sum_t \left(|\sin \theta_{kt}|\, e^{(\Re)}_t + |\cos \theta_{kt}|\, e^{(\Im)}_t\right),
\]
with \(\theta_{kt} = 2\pi kt/n\).

This keeps exact point boxes exact while still giving a deterministic, JAX-native enclosure rule for nonzero-width boxes.

## Dispatch

`dft_wrappers` exposes mode wrappers (`*_mode`) that select:
- `basic`: midpoint computation + outward rounding.
- `rigorous`: analytic interval DFT/convolution kernels.
- `adaptive`: Jacobian/sampling bounds (fallback).

The rigorous path is purely JAX and static‑shape/JIT‑friendly, enabling GPU execution.
