# Arb Incremental C -> JAX Migration Chassis

This workspace is a sidecar migration scaffold that keeps Arb source untouched while we port folder-by-folder.

## What this gives us

- `C` chassis: a standalone reference implementation for each migrated folder.
- `JAX` chassis: matching kernels designed for `jit`, `vmap`, autograd, and GPU.
- Accuracy harness: compare JAX outputs against C reference outputs.
- Runtime split:
  - Pure JAX runtime API: `arbjax.runtime`
  - Validation mode (C parity): parity tests and `migration/tools/compare_*.py`

## First migrated folder

- `double_interval` (completed in this scaffold)

## Second migrated folder

- `acb_hypgeom` + `arb_hypgeom`
- Completed kernels:
  - `rising_ui` (real + complex interval-box)
  - `lgamma` (real + complex interval-box)
  - `gamma` / `rgamma` (real + complex interval-box)
  - `erf` / `erfc` / `erfi` (real + complex interval-box)
  - `0f1`, `m`, `1f1`, `2f1` (real + complex interval-box)
  - Regularized variants for `0f1`, `m`, `1f1`, `2f1`
- `bessel_*` (real + complex interval-box)
  - Note: series-based approximations; accuracy degrades for large order/argument

## Third migrated folder

- `arb` core elementary real interval kernels
- Completed kernels:
  - `exp` / `log` / `sqrt`
  - `sin` / `cos` / `tan`
  - `sinh` / `cosh` / `tanh`

## Fourth migrated folder

## Fourth migrated folder

- `acb` core elementary complex interval kernels
- Completed kernels:
  - `exp` / `log` / `sqrt`
  - `sin` / `cos` / `tan`
  - `sinh` / `cosh` / `tanh`

## Fifth migrated folder

- `acb_dft` scaffold (complex floating migration chassis)
- Completed kernels:
  - Main DFT: `dft_naive`, `dft`, inverse variants
  - DFT on products: `dft_prod`
  - Convolution: circular `naive`, `dft`, default dispatch
  - FFT algorithms: radix-2 `dft_rad2`, inverse and convolution path
  - Interval-box variants: `acb_*` APIs for DFT/product/convolution/radix-2

## Layout

- `migration/c_chassis`: C reference library and build files.
- `migration/jax_chassis`: Python/JAX package and tests.
- `migration/tools`: parity scripts that compare C and JAX numerics.

### C support libraries now included

- `double_interval_ref`: outward-rounded interval primitives.
- `hypgeom_ref`: real/complex interval-box rising-factorial kernels built on `double_interval_ref`.
- `arb_core_ref`: real interval elementary kernels.
- `acb_core_ref`: complex interval elementary kernels.
- `dft_ref`: complex DFT/convolution reference kernels.

### Arb-like precision semantics in JAX

- Precision-aware APIs accept `prec_bits` and apply outward rounding after operations.
- Implemented in:
  - `double_interval`: `*_prec` and `*_prec_jit` batch variants
  - `hypgeom`: `arb/acb` `rising_ui_prec` and `lgamma_prec` (scalar + batch)

## Build and test

1. Build C reference library:

```powershell
cmake -S migration/c_chassis -B migration/c_chassis/build
cmake --build migration/c_chassis/build --config Release
```

2. Install JAX chassis (editable):

```powershell
python -m pip install -e migration/jax_chassis
```

3. Point tests to the built C library (Windows example):

```powershell
$env:DI_REF_LIB = "migration/c_chassis/build/libdouble_interval_ref.dll"
```

Linux/macOS examples:

- `migration/c_chassis/build/libdouble_interval_ref.so`
- `migration/c_chassis/build/libdouble_interval_ref.dylib`

4. Run tests:

```powershell
python -m pytest migration/jax_chassis/tests -q -m "not parity"

# parity mode (requires C libs)
$env:ARBJAX_RUN_PARITY = "1"
python -m pytest migration/jax_chassis/tests -q -m parity
python migration/tools/compare_double_interval.py --samples 20000
python migration/tools/compare_hypgeom.py --samples-real 10000 --samples-complex 5000
python migration/tools/compare_arb_core.py --samples 12000
python migration/tools/compare_acb_core.py --samples 12000
python migration/tools/compare_dft.py --n 8 --n-prod 12
python migration/tools/benchmark_hypgeom.py --samples-real 3000 --samples-complex 1200
python migration/tools/compare_acb_calc.py --samples 6000 --steps 48
python migration/tools/benchmark_acb_calc.py --samples 20000 --steps 64 --integrand exp
python migration/tools/compare_acb_dirichlet.py --samples 6000 --terms 64
python migration/tools/benchmark_acb_dirichlet.py --samples 20000 --terms 64 --which zeta
python migration/tools/compare_acb_elliptic.py --samples 6000
python migration/tools/benchmark_acb_elliptic.py --samples 20000 --which k
python migration/tools/compare_acb_mat.py --samples 6000
python migration/tools/benchmark_acb_mat.py --samples 20000 --which det
python migration/tools/compare_acb_modular.py --samples 6000
python migration/tools/benchmark_acb_modular.py --samples 20000
python migration/tools/compare_acb_poly.py --samples 6000
python migration/tools/benchmark_acb_poly.py --samples 20000
python migration/tools/compare_acf.py --samples 20000
python migration/tools/benchmark_acf.py --samples 200000 --which mul
python migration/tools/compare_arb_calc.py --samples 6000 --steps 48
python migration/tools/benchmark_arb_calc.py --samples 20000 --steps 64 --integrand exp
python migration/tools/compare_arb_fmpz_poly.py --samples 6000
python migration/tools/benchmark_arb_fmpz_poly.py --samples 20000
python migration/tools/compare_arb_fpwrap.py --samples 6000
python migration/tools/benchmark_arb_fpwrap.py --samples 200000 --which exp
python migration/tools/compare_arb_mat.py --samples 6000
python migration/tools/benchmark_arb_mat.py --samples 20000 --which det
python migration/tools/compare_arb_poly.py --samples 6000
python migration/tools/benchmark_arb_poly.py --samples 20000
python migration/tools/compare_arf.py --samples 20000
python migration/tools/benchmark_arf.py --samples 200000 --which mul
python migration/tools/compare_bernoulli.py --samples 20000
python migration/tools/benchmark_bernoulli.py --samples 200000
python migration/tools/compare_bool_mat.py --samples 10000
python migration/tools/benchmark_bool_mat.py --samples 200000 --which det
python migration/tools/compare_dirichlet.py --samples 6000 --terms 32 --which zeta
python migration/tools/benchmark_dirichlet.py --samples 20000 --terms 32 --which zeta
python migration/tools/compare_dlog.py --samples 20000
python migration/tools/benchmark_dlog.py --samples 200000
python migration/tools/compare_fmpr.py --samples 20000
python migration/tools/benchmark_fmpr.py --samples 200000 --which mul
python migration/tools/compare_fmpz_extras.py --samples 20000
python migration/tools/benchmark_fmpz_extras.py --samples 200000 --which mul
python migration/tools/compare_fmpzi.py --samples 6000
python migration/tools/benchmark_fmpzi.py --samples 200000 --which add
python migration/tools/compare_mag.py --samples 20000
python migration/tools/benchmark_mag.py --samples 200000 --which mul
python migration/tools/compare_partitions.py --max-n 20
python migration/tools/benchmark_partitions.py --samples 20000 --max-n 20
```

## Contract for next folders

For each additional folder migration:

1. Add C reference kernels to `migration/c_chassis` with batch APIs.
2. Add matching JAX kernels in `migration/jax_chassis/arbjax`.
3. Add parity tests and update the compare script.
4. Require:
   - `jit` compile pass
   - vectorized batch path
   - gradient path for smooth subdomain
   - C-vs-JAX numerical parity checks

## Scaffolding for remaining folders

For all remaining folders, the following scaffolding is now present:

- C stub headers and sources in `migration/c_chassis/include/*_ref.h` and `migration/c_chassis/src/*_ref.c`
- JAX stub modules in `migration/jax_chassis/arbjax/*.py`
- Placeholder tests in `migration/jax_chassis/tests/test_*_{chassis,parity}.py`
- Placeholder compare and benchmark tools in `migration/tools/compare_*.py` and `migration/tools/benchmark_*.py`
- Theory placeholders in `migration/theory/*.md`
- Benchmark results registry in `migration/results/`

These are stubs only; each folder still needs real function implementations and parity/benchmark logic.

## In progress

- `acb_calc` has initial line-integration chassis for `exp`, `sin`, and `cos`.
- `acb_dirichlet` has naive zeta/eta series chassis.
- `acb_elliptic` has AGM-based K/E chassis.
- `acb_mat` has 2x2 det/trace chassis.
- `acb_modular` has a truncated j-invariant chassis.
- `acb_poly` has cubic evaluation chassis.
- `acf` has complex add/mul chassis.
- `arb_calc` has line-integration chassis for `exp`, `sin`, and `cos`.
- `arb_fmpz_poly` has cubic evaluation chassis.
- `arb_fpwrap` has exp/log wrappers.
- `arb_mat` has 2x2 det/trace chassis.
- `arb_poly` has cubic evaluation chassis.
- `arf` has float add/mul chassis.
- `bernoulli` has low-order Bernoulli number chassis.
- `bool_mat` has 2x2 det/trace chassis over GF(2).
- `dirichlet` has naive zeta/eta series chassis (real interval).
- `dlog` has `log1p` numeric placeholder chassis.
- `fmpr` has float add/mul chassis.
- `fmpz_extras` has int64 add/mul chassis.
- `fmpzi` has int64 interval add/sub chassis.
- `mag` has float add/mul chassis.
- `partitions` has Euler-pentagonal partition chassis.
