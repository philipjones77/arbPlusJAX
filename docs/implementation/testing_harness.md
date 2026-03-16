Last updated: 2026-03-16T18:32:00Z

# Testing Harness

## Purpose

The benchmark and validation harness exists to compare arbPlusJAX against a fixed stack of external systems. The important point is that these backends do **not** all serve the same role:

- some are interval/enclosure references,
- some are high-precision point references,
- some are engineering parity baselines,
- some are optional fallbacks depending on what is installed locally.

Where possible below, backend names are tied to bibliography keys from `/mnt/c/dev/references/bibliography/library/software.bib`.

The canonical harness entry point is:

```bash
python benchmarks/bench_harness.py
```

The standard runner is:

```bash
python tools/run_benchmarks.py --profile quick
```

## Backend stack

The harness should be thought of as supporting these comparison layers.

### 1. Arb / FLINT

This is the primary interval reference whenever available.

Role:

- interval containment reference,
- midpoint/radius-style Arb-family comparison,
- closest external baseline for rigorous enclosure semantics.

Resolution:

- explicit `--c-ref-dir <path>`, or
- auto-detection through `ARB_C_REF_DIR`, `stuff/migration/c_chassis/build`, `../flint/build`, or `../arb/build`.

Harness backend name:

- `c_arb`

Use this whenever the function exists in the C reference layer. For interval semantics, this is the most important external comparator.

Bibliography keys:

- `[@Johansson2017]`
- `[@flint2026]`

### 2. Boost

Boost is an optional point-valued reference backend, mainly for special functions where Boost.Math has a mature implementation and where an additional non-Arb baseline is useful.

Role:

- independent point backend,
- cross-check for special-function formulas,
- useful when Arb and mpmath disagree or when we want another industrial-strength implementation in the comparison set.

Resolution:

- pass `--boost-ref-cmd "<command>"`, or
- set `BOOST_REF_CMD`.

The command reads JSON from stdin and returns a JSON numeric array on stdout. The repo also includes native and Python adapter paths.

Bibliography key:

- `[@boost2026]`

### 3. mpmath

mpmath is the high-precision point reference.

Role:

- arbitrary-precision point comparison,
- branch and continuation sanity checks,
- useful for functions not covered by SciPy or when we want higher precision than float64.

Resolution:

- installed `mpmath`, or
- `MPMATH_REPO`.

Important limitation:

- mpmath is **not** the interval reference in this harness unless one explicitly designs an `mp.iv` path. The current harness uses mpmath as a point backend.

Bibliography key:

- `[@mpmath2023]`

### 4. SciPy

SciPy is the standard float64 point baseline on CPU.

Role:

- mainstream scientific Python point comparison,
- useful for standard special functions and elementary functions,
- engineering baseline for expected user-facing float64 behavior.

Resolution:

- installed `scipy`, or
- `SCIPY_REPO`.

Harness backend name:

- `scipy`

Bibliography key:

- `[@scipy2020]`

### 5. JAX SciPy

JAX SciPy is the nearest “same framework, different implementation” point baseline.

Role:

- compares arbPlusJAX against standard JAX-native special-function behavior,
- separates interval-wrapper issues from raw JAX math issues,
- useful for tracking JIT/batch parity on the same runtime substrate.

Resolution:

- installed JAX/JAXlib, or
- `JAX_REPO`.

Harness backend name:

- `jax_scipy`

This is especially important because it isolates whether a discrepancy comes from:

- our interval/mode layer, or
- the underlying JAX point computation itself.

Bibliography key:

- `[@jaxscipyspecial2026]`

### 6. Mathematica

Mathematica is an optional high-precision point backend.

Role:

- independent symbolic/numeric point reference,
- useful for difficult special-function branch checks,
- useful as an extra arbiter when mpmath and Boost are inconclusive.

Resolution order:

1. local Wolfram installation if available,
2. Wolfram Cloud endpoint otherwise.

Local configuration:

- `WOLFRAM_WINDOWS_DIR`
- `WOLFRAM_LINUX_DIR`

Cloud configuration:

- `WOLFRAM_CLOUD_URL`
- optional `WOLFRAM_CLOUD_API_KEY`

Harness backend names:

- `mathematica_local`
- `mathematica_cloud`

Practical policy:

- prefer local `wolframscript` when present,
- otherwise use cloud if a maintained endpoint exists,
- otherwise skip Mathematica rather than blocking the run.

Bibliography key:

- `[@mathematica2026]`

## What each backend is for

The shortest honest summary is:

- Arb / FLINT: interval truth source.
- Boost: extra industrial point reference.
- mpmath: arbitrary-precision point reference.
- SciPy: mainstream float64 point reference.
- JAX SciPy: same-runtime point reference.
- Mathematica: optional independent high-precision point reference.

That distinction matters because not every mismatch means the same thing.

## Recommended comparison policy

When a function has support across the full stack, the preferred comparison order is:

1. Arb / FLINT for interval containment.
2. mpmath and Mathematica for high-precision point agreement.
3. SciPy and JAX SciPy for mainstream float64 parity.
4. Boost as an additional point backend when relevant.

In other words:

- use Arb to judge enclosure quality,
- use mpmath / Mathematica / Boost to judge hard point-value correctness,
- use SciPy / JAX SciPy to judge user-visible float64 parity.

## Current harness behavior

The harness currently resolves backends from `benchmarks/bench_registry.py`, where each function spec may define:

- `jax_basic`
- `jax_adaptive`
- `jax_rigorous`
- `jax_point`
- `scipy`
- `jax_scipy`
- `mpmath`
- `c_lib`
- `c_fn`

So the harness is already structured to compare:

- arbPlusJAX interval modes,
- arbPlusJAX point mode,
- external point references,
- the Arb C reference where present.

Boost and Mathematica are optional overlays on top of that core stack.

## Local source installs

The standard local source-install prefix in this workspace is:

```bash
~/.local/opt/arbplusjax_refs
```

Current subpaths:

- `~/.local/opt/arbplusjax_refs/flint/current`
- `~/.local/opt/arbplusjax_refs/boost/current`

The repository bootstrap for tests, benchmarks, and notebooks now auto-discovers that prefix and exports:

- `ARBPLUSJAX_REF_PREFIX`
- `FLINT_ROOT`
- `BOOST_ROOT`
- `BOOST_INCLUDEDIR`
- `BOOST_LIBRARYDIR`
- `BOOST_REF_CMD`
- `WOLFRAM_LINUX_DIR`

Manual shell bootstrap remains available:

```bash
source tools/source_reference_env.sh
```

Important distinction:

- `flint_install` means the FLINT/Arb source install is present.
- `arb_flint_c_ref` means the repo's separate C reference adapter layer is present.

Those are related but not identical. The elementary/core notebook reports both so that an installed FLINT tree is visible even if the older C adapter path is absent.

## Local vs cloud Mathematica

The repo should treat Mathematica availability as:

- local first,
- cloud second,
- optional always.

That is consistent with the current implementation in `benchmarks/bench_harness.py`, which first searches for `wolframscript` and then falls back to the configured Wolfram Cloud URL.

## Local implementation pointers

- [bench_harness.py](/home/phili/projects/arbplusJAX/benchmarks/bench_harness.py)
- [bench_registry.py](/home/phili/projects/arbplusJAX/benchmarks/bench_registry.py)
- [benchmarks.md](/home/phili/projects/arbplusJAX/docs/implementation/benchmarks.md)
- [benchmark_process.md](/home/phili/projects/arbplusJAX/docs/implementation/benchmark_process.md)
