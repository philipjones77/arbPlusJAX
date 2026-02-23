# DFT and Convolution

This module implements discrete Fourier transforms and circular convolution over complex numbers and complex interval boxes. The baseline kernels operate on midpoints (complex128). Rigorous mode uses explicit interval twiddle factors and interval arithmetic to propagate bounds.

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

## Dispatch

`dft_wrappers` exposes mode wrappers (`*_mode`) that select:
- `baseline`: midpoint computation + outward rounding.
- `rigorous`: analytic interval DFT/convolution kernels.
- `adaptive`: Jacobian/sampling bounds (fallback).

The rigorous path is purely JAX and static‑shape/JIT‑friendly, enabling GPU execution.
