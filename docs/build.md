Last updated: 2026-03-04T00:00:00Z

# Arb C build plan (Windows + Ubuntu)

This project runs parity tests against the Arb C library. Below are reproducible build recipes for Windows and Ubuntu.

## Windows (recommended: MSVC + vcpkg)

### 1. Toolchain
- Install **Visual Studio Build Tools 2022** with the **"Desktop development with C++"** workload.
- Install **CMake** (either standalone or via Visual Studio).

### 2. Install vcpkg
- Clone vcpkg and bootstrap it.

### 3. Install dependencies
- `vcpkg install mpir mpfr flint`

If `flint` is not available in your vcpkg build, build FLINT from source and set `CMAKE_PREFIX_PATH` to its install prefix.

### 4. Build Arb
From `C:\Users\phili\OneDrive\Documents\GitHub\arb`:

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=PATH_TO_VCPKG_INSTALLED
cmake --build build --config Release
```

### 5. Export library path for tests
```
setx ARB_LIB_DIR "C:\Users\phili\OneDrive\Documents\GitHub\arb\build\Release"
```
If headers are required, also set `ARB_INCLUDE_DIR` to `C:\Users\phili\OneDrive\Documents\GitHub\arb`.

## Windows (alternative: MSYS2 / MinGW64)

### 1. Install MSYS2
- Install MSYS2 and open the **MinGW64** shell.

### 2. Install dependencies
```
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-mpir mingw-w64-x86_64-mpfr mingw-w64-x86_64-flint cmake make
```

### 3. Build Arb
```
cd /c/Users/phili/OneDrive/Documents/GitHub/arb
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Ubuntu 22.04/24.04

### 1. Install dependencies
```
sudo apt update
sudo apt install -y build-essential cmake git libgmp-dev libmpfr-dev libflint-dev
```

### 2. Build Arb
```
cd ~/path/to/arb
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3. Export library path
```
export ARB_LIB_DIR=~/path/to/arb/build
export LD_LIBRARY_PATH=$ARB_LIB_DIR:$LD_LIBRARY_PATH
```

## Running parity tests (Windows + Linux)
From `arbPlusJAX`.

Windows PowerShell:
```powershell
$env:ARB_C_REF_DIR = "C:\path\to\arbPlusJAX\stuff\migration\c_chassis\build"
$env:ARBPLUSJAX_RUN_PARITY = "1"
python -m pytest tests -q -m parity
```

Linux/macOS (bash/zsh):
```bash
export ARB_C_REF_DIR=/path/to/arbPlusJAX/stuff/migration/c_chassis/build
export LD_LIBRARY_PATH="$ARB_C_REF_DIR:$LD_LIBRARY_PATH"
export ARBPLUSJAX_RUN_PARITY=1
python -m pytest tests -q -m parity
```

## Running benchmark smoke checks (optional, in CI)
From `arbPlusJAX`.

Windows PowerShell:
```powershell
$env:ARBPLUSJAX_RUN_BENCHMARKS = "1"
python -m pytest benchmarks -q -m benchmark
```

Linux/macOS (bash/zsh):
```bash
export ARBPLUSJAX_RUN_BENCHMARKS=1
python -m pytest benchmarks -q -m benchmark
```

For benchmark sweeps (speed/memory/accuracy):

```
python tools/run_benchmarks.py --profile quick
```

Optional Boost baseline:

```
python tools/run_benchmarks.py --profile quick --with-boost --boost-ref-cmd "<command>"
```

Generate a markdown benchmark summary:

```
python tools/bench_report.py --run results/benchmarks/<run_dir> --out results/benchmarks/<run_dir>/report.md
```

## Notes
- If `cl` or `gcc` is not found, install the toolchain before attempting the build.
- For GPU runs, install the correct CUDA/ROCm-enabled `jaxlib` wheel on Linux.

## Source packaging
Use:
```
python tools/package_repo.py
```
The archive is written to `_bundles/` by default.
The generated archive name is enforced as:
`<repo>_source_YYYY-MM-DD.zip`
Example: `arbPlusJAX_source_2026-03-01.zip`

## cuda_besselk backend
`cuda_besselk` is implemented in pure JAX (no CUDA/shared-library dependency).
Use API names `cuda_besselk` or `cuda_besselk`:
- point: `api.eval_point("cuda_besselk", nu, z)`
- interval: `api.eval_interval("cuda_besselk", nu_iv, z_iv, mode="basic|rigorous|adaptive")`

