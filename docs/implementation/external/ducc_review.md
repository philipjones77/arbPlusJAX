Last updated: 2026-03-18T00:10:00Z

# DUCC Review

This note records the initial audit of the DUCC repository as an external implementation lineage relevant to arbPlusJAX.

## Local checkout

- cloned to `/home/phili/projects/ducc`
- audited branch: `ducc0`
- audited commit: `2f3ba02ff717b116e9bfe65bd1032b5941217787`

## What DUCC provides

From the upstream repository and public Python bindings, the relevant functionality is:

- `ducc0.fft`
  - multidimensional `c2c`, `r2c`, `c2r`
  - real halfcomplex transforms in FFTPACK and FFTW storage conventions
  - DCT and DST
  - separable and genuine Hartley transforms
  - `good_size` padding helper
  - `convolve_axis`
  - long-double support
  - internal plan caching and multithreading
- `ducc0.nufft`
  - type-1 `nu2u`
  - type-2 `u2nu`
  - planned repeated transforms
  - incremental execution helpers
  - experimental type-3 `nu2nu`
- additional modules outside the immediate arbPlusJAX need
  - `sht`
  - `healpix`
  - `totalconvolve`
  - `wgridder`
  - `misc` Gauss-Legendre quadrature support

## Fit For arbPlusJAX

DUCC is too broad to be copied wholesale into arbPlusJAX.

The main constraints are:

- arbPlusJAX is MIT-licensed
- DUCC is GPLv2-or-later overall
- DUCC's README states that only the FFT component and its internal dependencies are additionally available under the 3-clause BSD license

That means:

- the full DUCC repository should not be vendored into arbPlusJAX
- the FFT-shaped subset may be portable, but only after a file-level license review
- the safer near-term path is to reimplement the needed ideas against public JAX APIs rather than pulling DUCC code directly

## What arbPlusJAX has today

arbPlusJAX now has a real 1D/2D/3D FFT/NUFFT starter subset, but it is still not a mature DUCC-class transform subsystem.

Current state:

- `dft.py` implements 1D midpoint DFT/IDFT and circular convolution
- `dft.py` also exposes 2D/3D complex FFT entry points built axis-by-axis from the 1D kernels
- power-of-two midpoint transforms delegate to `jnp.fft`
- arbitrary-length midpoint transforms use a repo-owned Bluestein kernel
- `make_dft_precomp()` and `dft_good_size()` support that FFT subset
- the rigorous interval paths still use explicit twiddle construction and `O(n^2)` accumulation
- `nufft.py` now provides JAX-native 1D/2D/3D type-1 and type-2 NUFFT subsets with exact direct reference kernels and an oversampled Lanczos acceleration path

Examples:

- `acb_dft_bluestein()` is now a real midpoint Bluestein path
- `acb_dft_bluestein_precomp()` consumes the repo-owned precomp payload
- `acb_dft_convol_mullow()` currently delegates to the naive circular convolution path

So the repo does have an FFT-shaped public surface, but it does not yet have:

- axis-wise convolution kernels like DUCC's `convolve_axis`
- a genuine high-performance rigorous FFT backend
- type-3 NUFFT

## Arb / FLINT comparison

FLINT does provide an `acb_dft` module. arbPlusJAX mirrors much of that naming, but the current implementation is not yet algorithmically equivalent to the mature FLINT/Arb surface.

The practical distinction is:

- naming parity: partial
- functional placeholder coverage: broad
- mature FFT backend parity: not yet

## Boost comparison

I did not find a general-purpose official Boost FFT library analogous to DUCC, pocketfft, FFTW, or FLINT's `acb_dft`.

Boost does have related but different functionality:

- Fourier-integral quadrature in `boost/math/quadrature/ooura_fourier_integrals.hpp`
- wavelet/Fourier helper functions in `boost/math/special_functions/fourier_transform_daubechies.hpp`

Those are not replacements for a general multidimensional FFT backend.

## Recommended scope

If we want DUCC-inspired functionality in arbPlusJAX, the right staged target is:

1. continue hardening the current `dft.py` surface so more of the public names correspond to real algorithms
2. extend the current midpoint FFT layer with:
   - `r2c` and `c2r`
   - normalization and axis controls
   - `good_size`
3. keep interval and rigorous wrappers as a separate layer
4. extend the current 1D/2D/3D NUFFT subset to richer kernels and plan caching only if downstream work actually needs it

The `sht`, `healpix`, `totalconvolve`, and `wgridder` parts of DUCC should be treated as separate future work rather than part of a first arbPlusJAX FFT push.

## Sources

- DUCC repository: <https://github.com/mreineck/ducc>
- DUCC README and license notes in the local checkout: `/home/phili/projects/ducc/README.md`
- FLINT `acb_dft` docs: <https://flintlib.org/doc/acb_dft.html>
