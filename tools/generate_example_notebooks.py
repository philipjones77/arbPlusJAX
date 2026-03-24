from __future__ import annotations

import json
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:  # pragma: no cover - environment fallback
    class _FallbackV4:
        @staticmethod
        def new_notebook() -> dict:
            return {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }

        @staticmethod
        def new_markdown_cell(source: str) -> dict:
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": source,
            }

        @staticmethod
        def new_code_cell(source: str) -> dict:
            return {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }

    class _FallbackNBF:
        v4 = _FallbackV4()

        @staticmethod
        def write(nb: dict, path: Path) -> None:
            path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

    nbf = _FallbackNBF()


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def _write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nbf.write(nb, path)


def _common_setup_cells(example_name: str):
    return [
        nbf.v4.new_markdown_cell(
            "## Scope\n\n"
            f"This notebook is the canonical example surface for `{example_name}`. "
            "It runs against the repo source tree through `/src`, shows direct public API usage, "
            "summarizes validation and benchmark status, and includes visual summaries."
        ),
        nbf.v4.new_code_cell(
            "import io\n"
            "import json\n"
            "import os\n"
            "import re\n"
            "import subprocess\n"
            "import sys\n"
            "import textwrap\n"
            "from pathlib import Path\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "\n"
            "def find_repo_root(start: Path) -> Path:\n"
            "    cur = start.resolve()\n"
            "    for p in [cur, *cur.parents]:\n"
            "        if (p / 'pyproject.toml').exists() and (p / 'src' / 'arbplusjax').exists():\n"
            "            return p\n"
            "    raise RuntimeError(f'Could not locate repo root from: {start}')\n"
            "\n"
            "REPO_ROOT = find_repo_root(Path.cwd())\n"
            "if str(REPO_ROOT / 'src') not in sys.path:\n"
            "    sys.path.insert(0, str(REPO_ROOT / 'src'))\n"
            "os.chdir(REPO_ROOT)\n"
            "\n"
            "PYTHON = os.getenv('ARBPLUSJAX_PYTHON', sys.executable)\n"
            "JAX_MODE = os.getenv('JAX_MODE', 'cpu').strip().lower()\n"
            "JAX_DTYPE = os.getenv('JAX_DTYPE', 'float64').strip().lower()\n"
            "RUN_ENV = os.environ.copy()\n"
            "RUN_ENV['PYTHONPATH'] = str(REPO_ROOT / 'src') + os.pathsep + RUN_ENV.get('PYTHONPATH', '')\n"
            "if JAX_MODE == 'cpu':\n"
            "    RUN_ENV['JAX_PLATFORMS'] = 'cpu'\n"
            "elif JAX_MODE == 'gpu':\n"
            "    RUN_ENV['JAX_PLATFORMS'] = 'cuda'\n"
            "RUN_ENV['JAX_ENABLE_X64'] = '1' if JAX_DTYPE == 'float64' else '0'\n"
            f"EXAMPLE_INPUT_ROOT = REPO_ROOT / 'examples' / 'inputs' / '{example_name}'\n"
            f"EXAMPLE_OUTPUT_ROOT = REPO_ROOT / 'examples' / 'outputs' / '{example_name}'\n"
            "EXAMPLE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "def run(cmd: list[str], *, capture: bool = False):\n"
            "    print('[cmd]', ' '.join(cmd))\n"
            "    return subprocess.run(cmd, cwd=REPO_ROOT, env=RUN_ENV, text=True, capture_output=capture, check=True)\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Environment\n\n"
            "The notebook reports interpreter, selected JAX mode, and the active backend/device view. "
            "Canonical retained execution in this repo state is CPU-oriented, but the notebook calling pattern remains CPU/GPU portable and explicitly parameterized for `float32` and `float64`."
        ),
        nbf.v4.new_code_cell(
            "SUPPORTED_JAX_MODES = ('cpu', 'gpu')\n"
            "SUPPORTED_JAX_DTYPES = ('float32', 'float64')\n"
            "if JAX_MODE not in SUPPORTED_JAX_MODES:\n"
            "    raise ValueError(f'Unsupported JAX_MODE: {JAX_MODE}')\n"
            "if JAX_DTYPE not in SUPPORTED_JAX_DTYPES:\n"
            "    raise ValueError(f'Unsupported JAX_DTYPE: {JAX_DTYPE}')\n"
            "print('python:', PYTHON)\n"
            "print('jax_mode:', JAX_MODE)\n"
            "print('jax_dtype:', JAX_DTYPE)\n"
            "print('supported_jax_modes:', SUPPORTED_JAX_MODES)\n"
            "print('supported_jax_dtypes:', SUPPORTED_JAX_DTYPES)\n"
            "print('validation_slice:', 'cpu_current__gpu_portable_contract')\n"
            "runtime = run([PYTHON, 'tools/check_jax_runtime.py'], capture=True)\n"
            "print(runtime.stdout)\n"
            "runtime_payload = json.loads(runtime.stdout)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'runtime_{JAX_MODE}.json').write_text(json.dumps(runtime_payload, indent=2) + '\\n', encoding='utf-8')"
        ),
    ]


def _production_pattern_cells(title: str, guidance: str, code: str, benchmark_notes: str):
    return [
        nbf.v4.new_markdown_cell(
            f"## {title}\n\n"
            f"{guidance}"
        ),
        nbf.v4.new_code_cell(code),
        nbf.v4.new_markdown_cell(
            "## Extending Benchmarks\n\n"
            f"{benchmark_notes}"
        ),
    ]


def _ad_pattern_cells(guidance: str, code: str):
    return [
        nbf.v4.new_markdown_cell(
            "## AD Product Pattern\n\n"
            f"{guidance}"
        ),
        nbf.v4.new_code_cell(code),
    ]


def _fast_jax_pattern_cells(guidance: str, code: str):
    return [
        nbf.v4.new_markdown_cell(
            "## Fast JAX Point Pattern\n\n"
            f"{guidance}"
        ),
        nbf.v4.new_code_cell(code),
    ]


def _core_scalar_notebook() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Example Core Scalar Surface\n\n"
            "Canonical core-scalar example notebook for the `arb_core`, `acb_core`, `arf`, `acf`, `fmpr`, "
            "`fmpzi`, and `arb_fpwrap` public surfaces."
        ),
        *_common_setup_cells("example_core_scalar_surface"),
        nbf.v4.new_markdown_cell(
            "## Object/Input Construction\n\n"
            "Construct representative real intervals, complex boxes, floating arrays, integer-interval arrays, "
            "and fpwrap point inputs."
        ),
        nbf.v4.new_code_cell(
            "import jax.numpy as jnp\n"
            "from arbplusjax import acb_core, api, double_interval as di\n"
            "\n"
            "real_interval = di.interval(jnp.array([0.2, 0.5, 1.0], dtype=jnp.float64), jnp.array([0.25, 0.6, 1.1], dtype=jnp.float64))\n"
            "complex_box = acb_core.acb_box(\n"
            "    di.interval(jnp.array([0.2, 0.4], dtype=jnp.float64), jnp.array([0.25, 0.5], dtype=jnp.float64)),\n"
            "    di.interval(jnp.array([-0.2, 0.1], dtype=jnp.float64), jnp.array([-0.1, 0.2], dtype=jnp.float64)),\n"
            ")\n"
            "float_a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)\n"
            "float_b = jnp.array([0.5, 1.5, 2.5], dtype=jnp.float32)\n"
            "complex_a = jnp.array([1.0 + 0.5j, 2.0 - 0.25j], dtype=jnp.complex64)\n"
            "int_interval_a = jnp.array([[1, 2], [3, 5], [8, 13]], dtype=jnp.int64)\n"
            "int_interval_b = jnp.array([[2, 3], [5, 8], [13, 21]], dtype=jnp.int64)\n"
            "fpwrap_real = jnp.array([0.1, 0.4, 0.9], dtype=jnp.float32)\n"
            "display({'real_interval_shape': real_interval.shape, 'complex_box_shape': complex_box.shape})"
        ),
        nbf.v4.new_markdown_cell(
            "## Direct Usage\n\n"
            "Run the public API directly on representative core scalar families."
        ),
        nbf.v4.new_code_cell(
            "results = {\n"
            "    'arb_exp_basic': api.eval_interval('arb_exp', real_interval, mode='basic', dtype='float64'),\n"
            "    'acb_sin_basic': api.eval_interval('acb_sin', complex_box, mode='basic', dtype='float64'),\n"
            "    'arf_add': api.eval_point('arf_add', float_a, float_b),\n"
            "    'acf_mul': api.eval_point('acf_mul', complex_a, complex_a),\n"
            "    'fmpr_mul': api.eval_point('fmpr_mul', float_a, float_b),\n"
            "    'fmpzi_add': api.eval_point('fmpzi_add', int_interval_a, int_interval_b),\n"
            "    'arb_fpwrap_double_exp': api.eval_point('arb_fpwrap_double_exp', fpwrap_real),\n"
            "}\n"
            "display(results)"
        ),
        *_production_pattern_cells(
            "Production Pattern",
            "Production use should bind once, keep dtype stable, and pad or chunk batches when repeated calls would otherwise trigger shape churn. "
            "For helper scalar families, `bind_point_batch()` is the main service-style entrypoint.",
            "service_real = jnp.linspace(0.1, 1.0, 7, dtype=jnp.float32)\n"
            "service_complex = jnp.asarray([0.1 + 0.2j, 0.3 - 0.1j, 0.5 + 0.4j], dtype=jnp.complex64)\n"
            "arf_bound = api.bind_point_batch('arf_add', dtype='float32', pad_to=16, chunk_size=4)\n"
            "acf_bound = api.bind_point_batch('acf_mul', dtype='float32', pad_to=16, chunk_size=4)\n"
            "fpwrap_bound = api.bind_point_batch('arb_fpwrap_double_exp', dtype='float32', pad_to=16)\n"
            "service_results = {\n"
            "    'arf_bound': arf_bound(service_real, service_real),\n"
            "    'acf_bound': acf_bound(service_complex, service_complex),\n"
            "    'fpwrap_bound': fpwrap_bound(service_real),\n"
            "}\n"
            "display(service_results)",
            "To benchmark another scalar family, either add another existing benchmark entrypoint to the `bench_dir` loop below or extend the representative operations in "
            "`benchmark_core_scalar_service_api.py` / `benchmark_core_scalar_batch_padding.py` and rerun this notebook."
        ),
        *_fast_jax_pattern_cells(
            "For the true compiled point path, use `api.bind_point_batch_jit(...)` with stable `dtype` and `pad_to`. "
            "This is the API-level fast-JAX surface for repeated point evaluation.",
            "import jax\n"
            "fast_real = jnp.linspace(0.1, 1.2, 8, dtype=jnp.float32)\n"
            "fast_bound = api.bind_point_batch_jit('arb_fpwrap_double_exp', dtype='float32', pad_to=8)\n"
            "fast_vals = fast_bound(fast_real)\n"
            "fast_vmap = jax.vmap(lambda t: api.eval_point('arb_fpwrap_double_exp', t, dtype='float32'))(fast_real)\n"
            "display({'jit_shape': fast_vals.shape, 'jit_dtype': fast_vals.dtype, 'jit_matches_vmap': bool(jnp.allclose(fast_vals, fast_vmap))})"
        ),
        *_ad_pattern_cells(
            "AD should be shown on the production-facing bound callables, not only on local toy functions. "
            "This section validates a scalar helper surface and plots primal versus gradient values over a representative sweep.",
            "import jax\n"
            "sweep = jnp.linspace(0.1, 1.2, 32, dtype=jnp.float32)\n"
            "exp_bound = api.bind_point_batch('arb_fpwrap_double_exp', dtype='float32', pad_to=32)\n"
            "def scalar_loss(x):\n"
            "    return jnp.sum(exp_bound(x))\n"
            "grad_vals = jax.grad(scalar_loss)(sweep)\n"
            "primal_vals = exp_bound(sweep)\n"
            "ad_df = pd.DataFrame({'x': np.asarray(sweep), 'primal': np.asarray(primal_vals), 'grad': np.asarray(grad_vals)})\n"
            "display(ad_df.head())\n"
            "ax = ad_df.plot(x='x', y=['primal', 'grad'], figsize=(8, 4), title='Core Scalar AD Validation')\n"
            "ax.set_ylabel('value')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'ad_validation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Parameter/Value Sweeps\n\n"
            "Sweep representative scalar families over values and modes using the existing harness profile."
        ),
        nbf.v4.new_code_cell(
            "profile_dir = EXAMPLE_OUTPUT_ROOT / ('cpu_profile' if JAX_MODE == 'cpu' else 'gpu_profile')\n"
            "run([\n"
            "    PYTHON, 'benchmarks/run_harness_profile.py',\n"
            "    '--name', f'example_core_scalar_{JAX_MODE}',\n"
            "    '--outdir', str(profile_dir),\n"
            "    '--functions', 'exp,log,sqrt,sin,cos,tan,sinh,cosh,tanh',\n"
            "    '--samples', '64,128',\n"
            "    '--seeds', '0,1',\n"
            "    '--jax-mode', JAX_MODE,\n"
            "    '--jax-dtype', JAX_DTYPE,\n"
            "    '--prec-bits', '128',\n"
            "])\n"
            "profile_csv = profile_dir / 'profile_summary.csv'\n"
            "profile_df = pd.read_csv(profile_csv)\n"
            "display(profile_df.head(20))"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Summary\n\n"
            "Run the existing scalar chassis/API tests and summarize the result."
        ),
        nbf.v4.new_code_cell(
            "tests = run([\n"
            "    PYTHON, '-m', 'pytest', '-q',\n"
            "    'tests/test_core_scalar_api_contracts.py',\n"
            "    'tests/test_arb_core_chassis.py',\n"
            "    'tests/test_acb_core_chassis.py',\n"
            "    'tests/test_arf_chassis.py',\n"
            "    'tests/test_acf_chassis.py',\n"
            "    'tests/test_fmpr_chassis.py',\n"
            "    'tests/test_fmpzi_chassis.py',\n"
            "    'tests/test_arb_fpwrap_chassis.py',\n"
            "], capture=True)\n"
            "print(tests.stdout)\n"
            "if tests.stderr:\n"
            "    print(tests.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'pytest_{JAX_MODE}.txt').write_text(tests.stdout + ('\\n' + tests.stderr if tests.stderr else ''), encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark Summary\n\n"
            "Run representative scalar benchmark entrypoints and summarize cold/warm/recompile behavior."
        ),
        nbf.v4.new_code_cell(
            "bench_dir = EXAMPLE_OUTPUT_ROOT / 'benchmark_artifacts'\n"
            "bench_dir.mkdir(parents=True, exist_ok=True)\n"
            "run([PYTHON, 'benchmarks/benchmark_arf.py', '--samples', '4096', '--which', 'add', '--runs', '3', '--output', str(bench_dir / 'benchmark_arf.json')])\n"
            "run([PYTHON, 'benchmarks/benchmark_arb_fpwrap.py', '--samples', '4096', '--which', 'exp', '--runs', '3', '--output', str(bench_dir / 'benchmark_arb_fpwrap.json')])\n"
            "run([PYTHON, 'benchmarks/benchmark_acf.py', '--samples', '4096', '--which', 'mul', '--runs', '3', '--output', str(bench_dir / 'benchmark_acf.json')])\n"
            "run([PYTHON, 'benchmarks/benchmark_fmpr.py', '--samples', '4096', '--which', 'mul', '--runs', '3', '--output', str(bench_dir / 'benchmark_fmpr.json')])\n"
            "run([PYTHON, 'benchmarks/benchmark_fmpzi.py', '--samples', '4096', '--which', 'add', '--runs', '3', '--output', str(bench_dir / 'benchmark_fmpzi.json')])\n"
            "run([PYTHON, 'benchmarks/benchmark_core_scalar_batch_padding.py', '--samples', '4099', '--pad-multiple', '128', '--runs', '3', '--output', str(bench_dir / 'benchmark_core_scalar_batch_padding.json')])\n"
            "bench_payloads = []\n"
            "for path in sorted(bench_dir.glob('*.json')):\n"
            "    payload = json.loads(path.read_text())\n"
            "    for row in payload['records']:\n"
                "        bench_payloads.append(row)\n"
            "bench_df = pd.DataFrame(bench_payloads)\n"
            "bench_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'benchmark_summary_{JAX_MODE}.csv', index=False)\n"
            "display(bench_df[['operation', 'cold_time_s', 'warm_time_s', 'recompile_time_s']])"
        ),
        nbf.v4.new_markdown_cell(
            "## Batch Padding Speed\n\n"
            "Measure padded versus unpadded API batch execution separately. "
            "This is distinct from the correctness or containment sweeps."
        ),
        nbf.v4.new_code_cell(
            "batch_padding_df = bench_df[bench_df['implementation'].isin(['api_batch_unpadded', 'api_batch_padded'])].copy()\n"
            "batch_padding_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'batch_padding_summary_{JAX_MODE}.csv', index=False)\n"
            "display(batch_padding_df[['operation', 'implementation', 'warm_time_s', 'recompile_time_s']])"
        ),
        nbf.v4.new_markdown_cell(
            "## Comparison Summary\n\n"
            "Where reference software is available, use the existing compare scripts. "
            "If local C reference libraries are absent, note that explicitly."
        ),
        nbf.v4.new_code_cell(
            "compare_cmds = [\n"
            "    [PYTHON, 'benchmarks/compare_arb_core.py', '--samples', '256', '--output', str(EXAMPLE_OUTPUT_ROOT / 'compare_arb_core.json')],\n"
            "    [PYTHON, 'benchmarks/compare_acb_core.py', '--samples', '256', '--output', str(EXAMPLE_OUTPUT_ROOT / 'compare_acb_core.json')],\n"
            "]\n"
            "comparison_rows = []\n"
            "for cmd in compare_cmds:\n"
            "    try:\n"
            "        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=RUN_ENV, text=True, capture_output=True, check=True)\n"
            "        print(completed.stdout)\n"
            "        comparison_rows.append({'script': cmd[1], 'status': 'ok', 'stdout': completed.stdout[-2000:], 'stderr': completed.stderr[-2000:]})\n"
            "    except subprocess.CalledProcessError as exc:\n"
            "        print('comparison unavailable or failed for', cmd[1])\n"
            "        print(exc.stdout)\n"
            "        print(exc.stderr)\n"
            "        comparison_rows.append({'script': cmd[1], 'status': 'failed_or_unavailable', 'stdout': (exc.stdout or '')[-2000:], 'stderr': (exc.stderr or '')[-2000:]})\n"
            "(EXAMPLE_OUTPUT_ROOT / f'comparison_status_{JAX_MODE}.json').write_text(json.dumps(comparison_rows, indent=2) + '\\n', encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Plots\n\n"
            "Plot backend timing and containment summaries from the harness profile."
        ),
        nbf.v4.new_code_cell(
            "plot_df = profile_df.copy()\n"
            "for col in ['time_ms', 'containment_rate', 'mean_abs_err']:\n"
            "    if col in plot_df.columns:\n"
            "        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')\n"
            "summary = plot_df.groupby('backend', dropna=False)[['time_ms', 'containment_rate']].mean(numeric_only=True).sort_values('time_ms')\n"
            "summary.to_csv(EXAMPLE_OUTPUT_ROOT / f'profile_backend_summary_{JAX_MODE}.csv')\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "summary['time_ms'].plot(kind='bar', ax=axes[0], title='Mean Time (ms)', color='#b85c38')\n"
            "summary['containment_rate'].plot(kind='bar', ax=axes[1], title='Mean Containment', color='#41535d')\n"
            "fig.tight_layout()\n"
            "fig.savefig(EXAMPLE_OUTPUT_ROOT / f'profile_backend_summary_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "if not batch_padding_df.empty:\n"
            "    pad_pivot = batch_padding_df.pivot(index='operation', columns='implementation', values='warm_time_s')\n"
            "    ax = pad_pivot.plot(kind='bar', figsize=(10, 4), title='Batch Padding Warm Time')\n"
            "    ax.set_ylabel('warm_time_s')\n"
            "    plt.tight_layout()\n"
            "    plt.savefig(EXAMPLE_OUTPUT_ROOT / f'batch_padding_warm_time_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "    plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Optional Diagnostics\n\n"
            "Use the matrix/compile diagnostics tools only when compile traces or memory deltas are needed. "
            "The scalar tranche here keeps those optional."
        ),
        nbf.v4.new_code_cell(
            "summary_lines = [\n"
            "    f'# Example Core Scalar Surface Summary ({JAX_MODE})',\n"
            "    '',\n"
            "    f'- python: `{PYTHON}`',\n"
            "    f'- backend: `{runtime_payload[\"platform\"]}`',\n"
            "    f'- devices: `{runtime_payload[\"devices\"]}`',\n"
            "    f'- benchmark_rows: `{len(bench_df)}`',\n"
            "    f'- profile_rows: `{len(profile_df)}`',\n"
            "    f'- comparison_rows: `{len(comparison_rows)}`',\n"
            "    f'- batch_padding_rows: `{len(batch_padding_df)}`',\n"
            "    '',\n"
            "    '## Benchmark Operations',\n"
            "    '',\n"
            "]\n"
            "for op in sorted(set(bench_df['operation'].tolist())):\n"
            "    summary_lines.append(f'- `{op}`')\n"
            "summary_lines.extend(['', '## Backend Summary', ''])\n"
            "for row in summary.reset_index().to_dict(orient='records'):\n"
            "    summary_lines.append(f\"- `{row['backend']}`: mean_time_ms={row['time_ms']:.6g}, mean_containment={row['containment_rate']:.6g}\")\n"
            "if not batch_padding_df.empty:\n"
            "    summary_lines.extend(['', '## Batch Padding Speed', ''])\n"
            "    for row in batch_padding_df.to_dict(orient='records'):\n"
            "        summary_lines.append(f\"- `{row['operation']}` / `{row['implementation']}`: warm={row['warm_time_s']:.6g}s, recompile={row['recompile_time_s']:.6g}s\")\n"
            "(EXAMPLE_OUTPUT_ROOT / f'summary_{JAX_MODE}.md').write_text('\\n'.join(summary_lines) + '\\n', encoding='utf-8')\n"
            "display('\\n'.join(summary_lines[:16]))"
        ),
    ]
    return cells


def _api_surface_notebook() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Example API Surface\n\n"
            "Canonical API/runtime routing example notebook for the public `api` facade."
        ),
        *_common_setup_cells("example_api_surface"),
        nbf.v4.new_markdown_cell(
            "## Object/Input Construction\n\n"
            "Build representative scalar, interval, and matrix inputs that exercise the routed API."
        ),
        nbf.v4.new_code_cell(
            "import jax.numpy as jnp\n"
            "from arbplusjax import acb_core, api, double_interval as di\n"
            "\n"
            "x = jnp.asarray(0.5, dtype=jnp.float64)\n"
            "y = jnp.asarray(2.0, dtype=jnp.float64)\n"
            "s = jnp.asarray(2.5, dtype=jnp.float64)\n"
            "z = jnp.asarray(1.0, dtype=jnp.float64)\n"
            "a_mid = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)\n"
            "rhs_mid = jnp.array([[1.0], [2.0]], dtype=jnp.float64)\n"
            "a = di.interval(a_mid, a_mid)\n"
            "rhs = di.interval(rhs_mid, rhs_mid)\n"
            "c_mid = jnp.array([[4.0 + 0.0j, 1.0 + 1.0j], [1.0 - 1.0j, 5.0 + 0.0j]], dtype=jnp.complex128)\n"
            "c_rhs_mid = jnp.array([[1.0 + 0.5j], [2.0 - 0.25j]], dtype=jnp.complex128)\n"
            "c_a = acb_core.acb_box(di.interval(jnp.real(c_mid), jnp.real(c_mid)), di.interval(jnp.imag(c_mid), jnp.imag(c_mid)))\n"
            "c_rhs = acb_core.acb_box(di.interval(jnp.real(c_rhs_mid), jnp.real(c_rhs_mid)), di.interval(jnp.imag(c_rhs_mid), jnp.imag(c_rhs_mid)))"
        ),
        nbf.v4.new_markdown_cell(
            "## Direct Usage\n\n"
            "Compare direct routed `evaluate()` calls against the explicit public entrypoints."
        ),
        nbf.v4.new_code_cell(
            "api_results = {\n"
            "    'besselk_routed': api.evaluate('besselk', x, y, implementation='cuda_besselk', value_kind='real'),\n"
            "    'incgamma_direct': api.incomplete_gamma_upper(s, z, method='quadrature', samples_per_panel=8, max_panels=16),\n"
            "    'incgamma_routed': api.evaluate('incomplete_gamma_upper', s, z, method='quadrature', method_params={'samples_per_panel': 8, 'max_panels': 16}),\n"
            "    'arb_mat_solve_routed': api.evaluate('arb_mat_solve', a, rhs, mode='basic', value_kind='real_matrix', dtype='float64'),\n"
            "    'acb_mat_solve_routed': api.evaluate('acb_mat_solve', c_a, c_rhs, mode='basic', value_kind='complex_matrix', dtype='float64'),\n"
            "}\n"
            "display(api_results)"
        ),
        *_production_pattern_cells(
            "Production Pattern",
            "For routed library use, favor explicit `evaluate()` or pre-bound batch entrypoints with fixed `dtype`, `mode`, and `pad_to` so service calls do not constantly recompile on batch-length drift. "
            "Keep matrix plans cached when an operation supports prepare/apply separation.",
            "real_batch = jnp.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=jnp.float64)\n"
            "gamma_bound = api.bind_point_batch('incomplete_gamma_upper', dtype='float64', pad_to=8, method='quadrature', regularized=True)\n"
            "solve_bound = api.bind_interval_batch('arb_mat_solve', mode='basic', dtype='float64', pad_to=4, prec_bits=53)\n"
            "cached_plan = api.eval_point('arb_mat_matvec_cached_prepare', a)\n"
            "api_service_results = {\n"
            "    'gamma_bound': gamma_bound(real_batch, jnp.asarray([1.0, 1.1, 1.2, 1.3, 1.4], dtype=jnp.float64)),\n"
            "    'solve_bound': solve_bound(a, rhs),\n"
            "    'cached_matvec': api.eval_point('arb_mat_matvec_cached_apply', cached_plan, rhs),\n"
            "}\n"
            "display(api_service_results)",
            "To extend routed benchmarks, add the target operation or implementation branch in `benchmark_api_surface.py`, `benchmark_special_function_service_api.py`, or `benchmark_matrix_service_api.py`, depending on whether the concern is generic API routing, special-function services, or matrix services."
        ),
        *_fast_jax_pattern_cells(
            "The routed API should still show the compiled point-batch path explicitly. "
            "Use `bind_point_batch_jit()` when a routed function is going to be called repeatedly in a point-mode service loop.",
            "import jax\n"
            "jit_gamma = api.bind_point_batch_jit('incomplete_gamma_upper', dtype='float64', pad_to=8, method='quadrature', regularized=True)\n"
            "jit_out = jit_gamma(real_batch, jnp.asarray([1.0, 1.1, 1.2, 1.3, 1.4], dtype=jnp.float64))\n"
            "vmap_out = jax.vmap(lambda s_i, z_i: api.evaluate('incomplete_gamma_upper', s_i, z_i, method='quadrature', regularized=True))(real_batch, jnp.asarray([1.0, 1.1, 1.2, 1.3, 1.4], dtype=jnp.float64))\n"
            "display({'jit_shape': jit_out.shape, 'jit_matches_vmap': bool(jnp.allclose(jit_out, vmap_out, rtol=1e-6, atol=1e-6))})"
        ),
        *_ad_pattern_cells(
            "The routed API should demonstrate AD through the same public metadata-aware entrypoints used in product code. "
            "This section differentiates a routed special-function call and plots primal and gradient behavior together.",
            "import jax\n"
            "sweep = jnp.linspace(0.25, 1.5, 24, dtype=jnp.float64)\n"
            "def routed_loss(zv):\n"
            "    return jnp.sum(api.evaluate('incomplete_gamma_upper', jnp.asarray(2.5, dtype=jnp.float64), zv, method='quadrature', method_params={'samples_per_panel': 8, 'max_panels': 16}))\n"
            "grad_vals = jax.grad(routed_loss)(sweep)\n"
            "primal_vals = api.bind_point_batch('incomplete_gamma_upper', dtype='float64', pad_to=32, method='quadrature')(jnp.full_like(sweep, 2.5), sweep)\n"
            "ad_df = pd.DataFrame({'z': np.asarray(sweep), 'primal': np.asarray(primal_vals), 'grad': np.asarray(grad_vals)})\n"
            "display(ad_df.head())\n"
            "ax = ad_df.plot(x='z', y=['primal', 'grad'], figsize=(8, 4), title='API Routed AD Validation')\n"
            "ax.set_ylabel('value')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'ad_validation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Parameter/Value Sweeps\n\n"
            "Sweep the official API benchmark over representative routed cases."
        ),
        nbf.v4.new_code_cell(
            "api_report = EXAMPLE_OUTPUT_ROOT / f'api_surface_{JAX_MODE}.json'\n"
            "run([PYTHON, 'benchmarks/benchmark_api_surface.py', '--warmup', '1', '--runs', '3', '--output', str(api_report)])\n"
            "api_payload = json.loads(api_report.read_text())\n"
            "api_df = pd.DataFrame(api_payload['records'])\n"
            "display(api_df[['operation', 'implementation', 'cold_time_s', 'warm_time_s', 'recompile_time_s']])"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Summary\n\n"
            "Run the API metadata and selection contract tests that back the routed public surface."
        ),
        nbf.v4.new_code_cell(
            "tests = run([\n"
            "    PYTHON, '-m', 'pytest', '-q',\n"
            "    'tests/test_api_metadata.py',\n"
            "    'tests/test_api_selection_contracts.py',\n"
            "    'tests/test_core_scalar_api_contracts.py',\n"
            "], capture=True)\n"
            "print(tests.stdout)\n"
            "if tests.stderr:\n"
            "    print(tests.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'pytest_{JAX_MODE}.txt').write_text(tests.stdout + ('\\n' + tests.stderr if tests.stderr else ''), encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark Summary\n\n"
            "Summarize the official API benchmark artifacts emitted by the routed benchmark script."
        ),
        nbf.v4.new_code_cell(
            "summary = api_df.groupby(['operation', 'implementation'])[['cold_time_s', 'warm_time_s', 'recompile_time_s']].mean(numeric_only=True)\n"
            "summary.reset_index().to_csv(EXAMPLE_OUTPUT_ROOT / f'api_benchmark_summary_{JAX_MODE}.csv', index=False)\n"
            "display(summary)"
        ),
        nbf.v4.new_markdown_cell(
            "## Comparison Summary\n\n"
            "The API layer is a routing surface rather than a separate numerical backend. "
            "Its comparison story is direct-vs-routed overhead, plus the downstream scalar/matrix comparison layers."
        ),
        nbf.v4.new_code_cell(
            "print('direct-vs-routed overhead rows:')\n"
            "display(api_df[['operation', 'implementation', 'warm_time_s']].sort_values(['operation', 'implementation']))"
        ),
        nbf.v4.new_markdown_cell(
            "## Diagnostics\n\n"
            "Run the existing matrix diagnostics entrypoint to capture compile, steady-state, and recompile behavior for representative routed matrix surfaces."
        ),
        nbf.v4.new_code_cell(
            "diag_report = EXAMPLE_OUTPUT_ROOT / f'api_matrix_diagnostics_{JAX_MODE}.json'\n"
            "run([PYTHON, 'benchmarks/benchmark_matrix_stack_diagnostics.py', '--n', '4', '--repeats', '2', '--output', str(diag_report)])\n"
            "diag_df = pd.DataFrame(json.loads(diag_report.read_text()))\n"
            "diag_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'api_matrix_diagnostics_{JAX_MODE}.csv', index=False)\n"
            "display(diag_df[['name', 'compile_ms', 'steady_ms_median', 'recompile_new_shape_ms', 'peak_rss_delta_mb']])"
        ),
        nbf.v4.new_markdown_cell(
            "## Plots\n\n"
            "Plot cold/warm/recompile timing by operation and implementation."
        ),
        nbf.v4.new_code_cell(
            "pivot = api_df.pivot(index='operation', columns='implementation', values='warm_time_s')\n"
            "ax = pivot.plot(kind='bar', figsize=(10, 4), title='API Warm Time by Operation')\n"
            "ax.set_ylabel('warm_time_s')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'api_warm_time_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Optional Diagnostics\n\n"
            "For compile/memory diagnostics beyond the API benchmark, use `benchmark_matrix_stack_diagnostics.py` "
            "or the JAX diagnostics helpers explicitly."
        ),
        nbf.v4.new_code_cell(
            "summary_lines = [\n"
            "    f'# Example API Surface Summary ({JAX_MODE})',\n"
            "    '',\n"
            "    f'- python: `{PYTHON}`',\n"
            "    f'- backend: `{runtime_payload[\"platform\"]}`',\n"
            "    f'- api_rows: `{len(api_df)}`',\n"
            "    f'- diagnostics_rows: `{len(diag_df)}`',\n"
            "    '',\n"
            "    '## Routed Operations',\n"
            "    '',\n"
            "]\n"
            "for row in summary.reset_index().to_dict(orient='records'):\n"
            "    summary_lines.append(f\"- `{row['operation']}` / `{row['implementation']}`: warm={row['warm_time_s']:.6g}s, cold={row['cold_time_s']:.6g}s, recompile={row['recompile_time_s']:.6g}s\")\n"
            "summary_lines.extend(['', '## Diagnostics Cases', ''])\n"
            "for row in diag_df.to_dict(orient='records'):\n"
            "    summary_lines.append(f\"- `{row['name']}`: compile_ms={row['compile_ms']:.6g}, steady_ms_median={row['steady_ms_median']:.6g}, recompile_new_shape_ms={row['recompile_new_shape_ms']:.6g}\")\n"
            "(EXAMPLE_OUTPUT_ROOT / f'summary_{JAX_MODE}.md').write_text('\\n'.join(summary_lines) + '\\n', encoding='utf-8')\n"
            "display('\\n'.join(summary_lines[:16]))"
        ),
    ]
    return cells


def _dense_matrix_surface_notebook() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Example Dense Matrix Surface\n\n"
            "Canonical dense matrix notebook for direct solve, cached matvec/rmatvec reuse, and dense operator-plan usage."
        ),
        *_common_setup_cells("example_dense_matrix_surface"),
        nbf.v4.new_markdown_cell(
            "## Direct Usage\n\n"
            "Construct representative real and complex dense matrices and exercise solve, matvec, cached matvec, and cached rmatvec surfaces."
        ),
        nbf.v4.new_code_cell(
            "import jax.numpy as jnp\n"
            "from arbplusjax import acb_core, acb_mat, arb_mat, double_interval as di, jcb_mat, jrb_mat\n"
            "\n"
            "a_mid = jnp.array([[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]], dtype=jnp.float64)\n"
            "x_mid = jnp.array([[1.0, 0.0], [2.0, 1.0], [-1.0, 3.0]], dtype=jnp.float64)\n"
            "vec_mid = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64)\n"
            "a = di.interval(a_mid, a_mid)\n"
            "x = di.interval(x_mid, x_mid)\n"
            "vec = di.interval(vec_mid, vec_mid)\n"
            "rhs = arb_mat.arb_mat_matmul(a, x)\n"
            "cache = arb_mat.arb_mat_dense_matvec_plan_prepare(a)\n"
            "rcache = arb_mat.arb_mat_rmatvec_cached_prepare(a)\n"
            "dense_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)\n"
            "\n"
            "a_c_mid = a_mid + 1j * jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.5], [0.0, -0.5, 0.0]], dtype=jnp.float64)\n"
            "vec_c_mid = vec_mid + 1j * jnp.array([0.25, -0.1, 0.3], dtype=jnp.float64)\n"
            "a_c = acb_core.acb_box(di.interval(jnp.real(a_c_mid), jnp.real(a_c_mid)), di.interval(jnp.imag(a_c_mid), jnp.imag(a_c_mid)))\n"
            "vec_c = acb_core.acb_box(di.interval(jnp.real(vec_c_mid), jnp.real(vec_c_mid)), di.interval(jnp.imag(vec_c_mid), jnp.imag(vec_c_mid)))\n"
            "cache_c = acb_mat.acb_mat_dense_matvec_plan_prepare(a_c)\n"
            "rcache_c = acb_mat.acb_mat_rmatvec_cached_prepare(a_c)\n"
            "dense_plan_c = jcb_mat.jcb_mat_dense_operator_plan_prepare(a_c)\n"
            "\n"
            "dense_results = {\n"
            "    'solve_basic': arb_mat.arb_mat_solve(a, rhs),\n"
            "    'cached_matvec': arb_mat.arb_mat_dense_matvec_plan_apply(cache, vec),\n"
            "    'cached_rmatvec': arb_mat.arb_mat_rmatvec_cached_apply(rcache, vec),\n"
            "    'operator_apply': jrb_mat.jrb_mat_operator_plan_apply(dense_plan, vec),\n"
            "    'complex_cached_matvec': acb_mat.acb_mat_dense_matvec_plan_apply(cache_c, vec_c),\n"
            "    'complex_cached_rmatvec': acb_mat.acb_mat_rmatvec_cached_apply(rcache_c, vec_c),\n"
            "    'complex_operator_apply': jcb_mat.jcb_mat_operator_plan_apply(dense_plan_c, vec_c),\n"
            "}\n"
            "display(dense_results)"
        ),
        *_production_pattern_cells(
            "Production Pattern",
            "Dense production use should prepare solve and matvec/rmatvec plans once, reuse them across repeated calls, and keep dtype and batch shape stable. "
            "Dense operator plans should be the bridge into matrix-free workflows when callers later want Krylov-style execution without rewriting the model surface.",
            "rhs_batch = jnp.stack([vec, vec], axis=0)\n"
            "dense_service = {\n"
            "    'solve_reuse': arb_mat.arb_mat_dense_lu_solve_plan_apply(arb_mat.arb_mat_dense_lu_solve_plan_prepare(a), rhs),\n"
            "    'cached_matvec': arb_mat.arb_mat_dense_matvec_plan_apply(cache, vec),\n"
            "    'cached_rmatvec': arb_mat.arb_mat_rmatvec_cached_apply(rcache, vec),\n"
            "    'cached_matvec_padded': arb_mat.arb_mat_dense_matvec_plan_apply_batch_padded(cache, rhs_batch, pad_to=8),\n"
            "    'operator_apply': jrb_mat.jrb_mat_operator_plan_apply(dense_plan, vec),\n"
            "}\n"
            "display(dense_service)",
            "To extend dense benchmarks, add a stable metric in `benchmark_dense_matrix_surface.py`; use that same metric key in the matrix workbook so dense, sparse, and matrix-free surfaces remain comparable."
        ),
        *_fast_jax_pattern_cells(
            "Dense point-mode fast JAX should run through the compiled public batch surface with cached-apply style operations where available.",
            "import jax\n"
            "dense_batch = di.interval(jnp.stack([a_mid, a_mid], axis=0), jnp.stack([a_mid, a_mid], axis=0))\n"
            "rhs_batch_fast = di.interval(jnp.stack([vec_mid, vec_mid], axis=0), jnp.stack([vec_mid, vec_mid], axis=0))\n"
            "dense_fast = api.bind_point_batch_jit('arb_mat_matvec', dtype='float64', pad_to=4)\n"
            "dense_fast_out = dense_fast(dense_batch, rhs_batch_fast)\n"
            "dense_vmap = jax.vmap(lambda aa, xx: api.eval_point('arb_mat_matvec', aa, xx, dtype='float64'))(dense_batch, rhs_batch_fast)\n"
            "display({'jit_shape': dense_fast_out.shape, 'jit_matches_vmap': bool(jnp.allclose(dense_fast_out, dense_vmap))})"
        ),
        *_ad_pattern_cells(
            "Dense AD should be demonstrated on the production-facing plan-based or operator-plan surface rather than only on a raw midpoint helper. "
            "This section differentiates a dense operator-plan objective and plots primal and gradient behavior over a scale sweep.",
            "import jax\n"
            "base = a_mid\n"
            "def dense_loss(scale):\n"
            "    scaled = di.interval(scale * base, scale * base)\n"
            "    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(scaled)\n"
            "    out = jrb_mat.jrb_mat_operator_plan_apply(plan, vec)\n"
            "    return jnp.sum(di.midpoint(out))\n"
            "scale_sweep = jnp.linspace(0.75, 1.25, 24, dtype=jnp.float64)\n"
            "primal_vals = jax.vmap(dense_loss)(scale_sweep)\n"
            "grad_vals = jax.vmap(jax.grad(dense_loss))(scale_sweep)\n"
            "ad_df = pd.DataFrame({'scale': np.asarray(scale_sweep), 'primal': np.asarray(primal_vals), 'grad': np.asarray(grad_vals)})\n"
            "display(ad_df.head())\n"
            "ax = ad_df.plot(x='scale', y=['primal', 'grad'], figsize=(8, 4), title='Dense Matrix AD Validation')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'ad_validation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Summary\n\n"
            "Run the dense matrix chassis and matrix-stack contract tests that own cached matvec/rmatvec and operator-plan adaptation."
        ),
        nbf.v4.new_code_cell(
            "tests = run([\n"
            "    PYTHON, '-m', 'pytest', '-q',\n"
            "    'tests/test_arb_mat_chassis.py',\n"
            "    'tests/test_acb_mat_chassis.py',\n"
            "    'tests/test_dense_broad_surface.py',\n"
            "    'tests/test_matrix_stack_contracts.py',\n"
            "], capture=True)\n"
            "print(tests.stdout)\n"
            "if tests.stderr:\n"
            "    print(tests.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'pytest_{JAX_MODE}.txt').write_text(tests.stdout + ('\\n' + tests.stderr if tests.stderr else ''), encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark Summary\n\n"
            "Run the dense matrix benchmark in schema-backed form and compare cached/direct/operator-friendly paths."
        ),
        nbf.v4.new_code_cell(
            "dense_report = EXAMPLE_OUTPUT_ROOT / f'dense_matrix_surface_{JAX_MODE}.json'\n"
            "run([PYTHON, 'benchmarks/benchmark_dense_matrix_surface.py', '--n', '8', '--warmup', '1', '--runs', '2', '--output', str(dense_report)])\n"
            "bench_payload = json.loads(dense_report.read_text())\n"
            "bench_df = pd.DataFrame(bench_payload['records']).sort_values('warm_time_s')\n"
            "bench_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'dense_benchmark_summary_{JAX_MODE}.csv', index=False)\n"
            "display(bench_df[['implementation', 'operation', 'dtype', 'warm_time_s']].head(20))"
        ),
        nbf.v4.new_markdown_cell(
            "## Comparison / Contrast\n\n"
            "Summarize direct solve, cached matvec/rmatvec, and operator-plan usage so dense callers can compare production calling styles."
        ),
        nbf.v4.new_code_cell(
            "compare_df = bench_df[bench_df['operation'].isin(['direct_solve', 'cached_matvec', 'cached_rmatvec', 'dense_plan_prepare'])].copy()\n"
            "display(compare_df[['implementation', 'operation', 'warm_time_s']])"
        ),
        nbf.v4.new_markdown_cell(
            "## Plots\n\n"
            "Plot dense matrix timing by operation to make direct, cached, and operator-plan-friendly paths easy to compare visually."
        ),
        nbf.v4.new_code_cell(
            "pivot = bench_df.pivot(index='operation', columns='implementation', values='warm_time_s')\n"
            "ax = pivot.plot(kind='bar', figsize=(11, 4), title='Dense Matrix Warm Time by Operation')\n"
            "ax.set_ylabel('warm_time_s')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'dense_benchmark_summary_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Optional Diagnostics\n\n"
            "Use the matrix stack diagnostics benchmark when compile, recompile, and operator-plan adaptation behavior needs deeper inspection."
        ),
        nbf.v4.new_code_cell(
            "summary_lines = [\n"
            "    f'# Example Dense Matrix Surface Summary ({JAX_MODE})',\n"
            "    '',\n"
            "    f'- backend: `{runtime_payload[\"platform\"]}`',\n"
            "    f'- benchmark_rows: `{len(bench_df)}`',\n"
            "    '',\n"
            "    '## Comparison Slice',\n"
            "    '',\n"
            "]\n"
            "for row in compare_df.to_dict(orient='records'):\n"
            "    summary_lines.append(f\"- `{row['implementation']}` / `{row['operation']}`: warm={row['warm_time_s']:.6g}s\")\n"
            "(EXAMPLE_OUTPUT_ROOT / f'summary_{JAX_MODE}.md').write_text('\\n'.join(summary_lines) + '\\n', encoding='utf-8')\n"
            "display('\\n'.join(summary_lines[:14]))"
        ),
    ]
    return cells


def _sparse_matrix_surface_notebook() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Example Sparse Matrix Surface\n\n"
            "Canonical sparse/block-sparse example notebook for the sparse matrix tranche."
        ),
        *_common_setup_cells("example_sparse_matrix_surface"),
        nbf.v4.new_markdown_cell(
            "## Direct Usage\n\n"
            "Construct sparse real/complex operators and exercise the public sparse API surface."
        ),
        nbf.v4.new_code_cell(
            "import jax.numpy as jnp\n"
            "from arbplusjax import api, scb_mat, srb_mat\n"
            "\n"
            "dense_real = jnp.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]], dtype=jnp.float64)\n"
            "dense_complex = dense_real + 1j * jnp.array([[0.0, 0.2, 0.0], [-0.2, 0.0, 0.1], [0.0, -0.1, 0.0]], dtype=jnp.float64)\n"
            "sparse_real = srb_mat.srb_mat_from_dense_bcoo(dense_real)\n"
            "sparse_complex = scb_mat.scb_mat_from_dense_bcoo(dense_complex)\n"
            "block_real = api.eval_point('srb_block_mat_from_dense_csr', dense_real, block_shape=(1, 1))\n"
            "vblock_real = api.eval_point('srb_vblock_mat_from_dense_csr', dense_real, row_block_sizes=jnp.asarray([1, 2], dtype=jnp.int32), col_block_sizes=jnp.asarray([1, 2], dtype=jnp.int32))\n"
            "vec_real = jnp.stack([jnp.array([1.0, 0.5, -0.25], dtype=jnp.float64), jnp.array([0.25, -0.5, 1.0], dtype=jnp.float64)], axis=0)\n"
            "vec_complex = jnp.stack([\n"
            "    jnp.array([1.0 + 0.2j, 0.5 - 0.1j, -0.25 + 0.3j], dtype=jnp.complex128),\n"
            "    jnp.array([0.25 - 0.1j, -0.5 + 0.2j, 1.0 + 0.0j], dtype=jnp.complex128),\n"
            "], axis=0)\n"
            "real_plan = api.eval_point('srb_mat_matvec_cached_prepare', sparse_real)\n"
            "complex_plan = api.eval_point('scb_mat_matvec_cached_prepare', sparse_complex)\n"
            "real_rplan = api.eval_point('srb_mat_rmatvec_cached_prepare', sparse_real)\n"
            "sparse_results = {\n"
            "    'srb_matvec': api.eval_point_batch('srb_mat_matvec', sparse_real, vec_real),\n"
            "    'srb_cached_matvec': api.eval_point_batch('srb_mat_matvec_cached_apply', real_plan, vec_real),\n"
            "    'srb_cached_rmatvec': api.eval_point_batch('srb_mat_rmatvec_cached_apply', real_rplan, vec_real),\n"
            "    'srb_block_matvec': api.eval_point_batch('srb_block_mat_matvec', block_real, vec_real),\n"
            "    'srb_vblock_matvec': api.eval_point_batch('srb_vblock_mat_matvec', vblock_real, vec_real),\n"
            "    'scb_matvec': api.eval_point_batch('scb_mat_matvec', sparse_complex, vec_complex),\n"
            "    'scb_cached_matvec': api.eval_point_batch('scb_mat_matvec_cached_apply', complex_plan, vec_complex),\n"
            "}\n"
            "display(sparse_results)"
        ),
        *_production_pattern_cells(
            "Production Pattern",
            "Sparse production calls should prepare cached plans once and reuse them for repeated apply or solve traffic. "
            "If an API-facing service loop feeds variable batch sizes, pad the RHS batch to a stable multiple before calling the padded batch kernels.",
            "rhs_batch = vec_real\n"
            "real_cached = api.eval_point('srb_mat_matvec_cached_prepare', sparse_real)\n"
            "real_rcached = api.eval_point('srb_mat_rmatvec_cached_prepare', sparse_real)\n"
            "sparse_service_results = {\n"
            "    'cached_apply': api.eval_point_batch('srb_mat_matvec_cached_apply', real_cached, rhs_batch),\n"
            "    'cached_rmatvec': api.eval_point_batch('srb_mat_rmatvec_cached_apply', real_rcached, rhs_batch),\n"
            "    'padded_batch_apply': api.eval_point_batch('srb_mat_matvec', sparse_real, rhs_batch, pad_to=8),\n"
            "    'block_sparse_apply': api.eval_point_batch('srb_block_mat_matvec', block_real, rhs_batch),\n"
            "    'vblock_sparse_apply': api.eval_point_batch('srb_vblock_mat_matvec', vblock_real, rhs_batch),\n"
            "}\n"
            "display(sparse_service_results)",
            "To extend sparse benchmarks, add the target sparse/storage/mode combination inside `benchmark_sparse_matrix_surface.py` and keep the printed metric keys stable so downstream notebook parsing still works."
        ),
        *_fast_jax_pattern_cells(
            "Sparse point-mode fast JAX should use the compiled batch binder on a prepared operator or cached plan, with shape-stable RHS batches.",
            "import jax\n"
            "sparse_fast = api.bind_point_batch_jit('srb_mat_matvec_cached_apply', dtype='float64', pad_to=4)\n"
            "sparse_fast_out = sparse_fast(real_cached, rhs_batch)\n"
            "sparse_vmap = jax.vmap(lambda row: api.eval_point('srb_mat_matvec_cached_apply', real_cached, row, dtype='float64'))(rhs_batch)\n"
            "display({'jit_shape': sparse_fast_out.shape, 'jit_matches_vmap': bool(jnp.allclose(sparse_fast_out, sparse_vmap))})"
        ),
        *_ad_pattern_cells(
            "Sparse AD should be demonstrated through stable public operations that are realistically differentiated in downstream models. "
            "This section differentiates a squared-output objective through sparse matvec and plots primal versus gradient norms.",
            "import jax\n"
            "dense_param = jnp.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]], dtype=jnp.float64)\n"
            "vec0 = jnp.array([1.0, 0.5, -0.25], dtype=jnp.float64)\n"
            "def sparse_loss(scale):\n"
            "    sparse_scaled = srb_mat.srb_mat_from_dense_bcoo(scale * dense_param)\n"
            "    out = api.eval_point('srb_mat_matvec', sparse_scaled, vec0)\n"
            "    return jnp.sum(out ** 2)\n"
            "scale_sweep = jnp.linspace(0.5, 1.5, 24, dtype=jnp.float64)\n"
            "primal_vals = jax.vmap(sparse_loss)(scale_sweep)\n"
            "grad_vals = jax.vmap(jax.grad(sparse_loss))(scale_sweep)\n"
            "ad_df = pd.DataFrame({'scale': np.asarray(scale_sweep), 'primal': np.asarray(primal_vals), 'grad': np.asarray(grad_vals)})\n"
            "display(ad_df.head())\n"
            "ax = ad_df.plot(x='scale', y=['primal', 'grad'], figsize=(8, 4), title='Sparse AD Validation')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'ad_validation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Summary\n\n"
            "Run the sparse API and chassis tests that own this surface."
        ),
        nbf.v4.new_code_cell(
            "tests = run([\n"
            "    PYTHON, '-m', 'pytest', '-q',\n"
            "    'tests/test_sparse_point_api.py',\n"
            "    'tests/test_sparse_basic_contracts.py',\n"
            "    'tests/test_srb_mat_chassis.py',\n"
            "    'tests/test_scb_mat_chassis.py',\n"
            "    'tests/test_srb_block_mat_chassis.py',\n"
            "    'tests/test_scb_block_mat_chassis.py',\n"
            "    'tests/test_srb_vblock_mat_chassis.py',\n"
            "    'tests/test_scb_vblock_mat_chassis.py',\n"
            "], capture=True)\n"
            "print(tests.stdout)\n"
            "if tests.stderr:\n"
            "    print(tests.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'pytest_{JAX_MODE}.txt').write_text(tests.stdout + ('\\n' + tests.stderr if tests.stderr else ''), encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark Summary\n\n"
            "Run the sparse surface benchmark and convert the printed metrics into a structured table."
        ),
        nbf.v4.new_code_cell(
            "completed = run([PYTHON, 'benchmarks/benchmark_sparse_matrix_surface.py', '--n', '16', '--warmup', '1', '--runs', '2'], capture=True)\n"
            "print(completed.stdout)\n"
            "rows = []\n"
            "for line in completed.stdout.splitlines():\n"
            "    if ': ' not in line:\n"
            "        continue\n"
            "    key, value = line.split(': ', 1)\n"
            "    if key in {'platform', 'jax'} or key.startswith('n'):\n"
            "        continue\n"
            "    try:\n"
            "        rows.append({'metric': key, 'seconds': float(value)})\n"
            "    except ValueError:\n"
            "        pass\n"
            "bench_df = pd.DataFrame(rows).sort_values('seconds')\n"
            "bench_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'sparse_benchmark_summary_{JAX_MODE}.csv', index=False)\n"
            "display(bench_df.head(20))"
        ),
        nbf.v4.new_markdown_cell(
            "## Comparison / Contrast\n\n"
            "Compare sparse, block-sparse, and variable-block matvec/cached-rmatvec surfaces in one place so users can see when storage structure changes the calling pattern."
        ),
        nbf.v4.new_code_cell(
            "compare_metrics = bench_df[bench_df['metric'].str.contains('matvec|rmatvec', regex=True, na=False)].copy()\n"
            "display(compare_metrics.head(20))"
        ),
        nbf.v4.new_markdown_cell(
            "## Plots\n\n"
            "Plot the fastest sparse benchmark metrics as a compact diagnostic summary."
        ),
        nbf.v4.new_code_cell(
            "top = bench_df.head(12).copy()\n"
            "ax = top.plot(x='metric', y='seconds', kind='barh', figsize=(10, 5), color='#5b7c8d', legend=False, title='Sparse Benchmark Summary')\n"
            "ax.set_xlabel('seconds')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'sparse_benchmark_summary_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Optional Diagnostics\n\n"
            "Detailed sparse solver/factor diagnostics remain owned by the sparse matrix modules and their dedicated benchmarks."
        ),
        nbf.v4.new_code_cell(
            "summary_lines = [\n"
            "    f'# Example Sparse Matrix Surface Summary ({JAX_MODE})',\n"
            "    '',\n"
            "    f'- backend: `{runtime_payload[\"platform\"]}`',\n"
            "    f'- benchmark_rows: `{len(bench_df)}`',\n"
            "    '',\n"
            "    '## Fastest Metrics',\n"
            "    '',\n"
            "]\n"
            "for row in top.to_dict(orient='records'):\n"
            "    summary_lines.append(f\"- `{row['metric']}`: {row['seconds']:.6g}s\")\n"
            "(EXAMPLE_OUTPUT_ROOT / f'summary_{JAX_MODE}.md').write_text('\\n'.join(summary_lines) + '\\n', encoding='utf-8')\n"
            "display('\\n'.join(summary_lines[:12]))"
        ),
    ]
    return cells


def _matrix_free_operator_surface_notebook() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Example Matrix-Free Operator Surface\n\n"
            "Canonical matrix-free/operator notebook for the Krylov and operator-plan tranche."
        ),
        *_common_setup_cells("example_matrix_free_operator_surface"),
        nbf.v4.new_markdown_cell(
            "## Direct Usage\n\n"
            "Build a small dense operator plan and exercise matrix-free apply, solve, and logdet-facing paths."
        ),
        nbf.v4.new_code_cell(
            "import jax.numpy as jnp\n"
            "from arbplusjax import double_interval as di, jrb_mat\n"
            "\n"
            "diag = jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)\n"
            "a_mid = jnp.diag(diag)\n"
            "a = di.interval(a_mid, a_mid)\n"
            "x = di.interval(jnp.array([1.0, 0.5, -0.25, 0.75], dtype=jnp.float64), jnp.array([1.0, 0.5, -0.25, 0.75], dtype=jnp.float64))\n"
            "plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)\n"
            "sparse_plan = jrb_mat.jrb_mat_sparse_operator_plan_prepare(jrb_mat.sparse_common.dense_to_sparse_bcoo(a_mid, algebra='jrb'))\n"
            "probes = jnp.stack([x, x], axis=0)\n"
            "operator_results = {\n"
            "    'apply': jrb_mat.jrb_mat_operator_plan_apply(plan, x),\n"
            "    'sparse_apply': jrb_mat.jrb_mat_operator_plan_apply(sparse_plan, x),\n"
            "    'solve': jrb_mat.jrb_mat_solve_action_point_jit(plan, x, symmetric=True),\n"
            "    'logdet': jrb_mat.jrb_mat_logdet_slq_point(plan, probes, steps=6),\n"
            "}\n"
            "display(operator_results)"
        ),
        *_production_pattern_cells(
            "Production Pattern",
            "Matrix-free production use should prepare operator plans once, reuse preconditioners, and keep problem size and Krylov steps stable across service requests where possible. "
            "This reduces recompiles and keeps diagnostics interpretable.",
            "precond = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(plan)\n"
            "solve_once = lambda rhs: jrb_mat.jrb_mat_solve_action_point_jit(plan, rhs, symmetric=True)\n"
            "multi_shift_once = lambda rhs: jrb_mat.jrb_mat_multi_shift_solve_point_jit(plan, rhs, jnp.asarray([0.0, 0.5], dtype=jnp.float64), symmetric=True, preconditioner=precond)\n"
            "matrix_free_service = {\n"
            "    'operator_plan': jrb_mat.jrb_mat_operator_plan_apply(plan, x),\n"
            "    'sparse_operator_plan': jrb_mat.jrb_mat_operator_plan_apply(sparse_plan, x),\n"
            "    'solve_once': solve_once(x),\n"
            "    'multi_shift_once': multi_shift_once(x),\n"
            "}\n"
            "display(matrix_free_service)",
            "To benchmark another operator path, add a new metric block in `benchmark_matrix_free_krylov.py` with a stable metric name and then include that section in the notebook parsing."
        ),
        *_fast_jax_pattern_cells(
            "Matrix-free fast JAX uses the family-owned compiled point kernels directly. "
            "The important contract is still the same: fixed problem shape, fixed Krylov steps, and no dynamic rescue logic in the hot path.",
            "logdet_jit = jrb_mat.jrb_mat_logdet_slq_point_jit(plan, probes, 6)\n"
            "logdet_ref = jrb_mat.jrb_mat_logdet_slq_point(plan, probes, 6)\n"
            "display({'jit_value': logdet_jit, 'jit_matches_point': bool(jnp.allclose(logdet_jit, logdet_ref, rtol=1e-6, atol=1e-6))})"
        ),
        *_ad_pattern_cells(
            "Matrix-free AD should be shown on operator-plan-first usage, since that is the production surface. "
            "This section differentiates a solve-based scalar objective and plots primal and gradient behavior over a scale sweep.",
            "import jax\n"
            "base_diag = jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)\n"
            "def mf_loss(scale):\n"
            "    a_mid = jnp.diag(scale * base_diag)\n"
            "    a = di.interval(a_mid, a_mid)\n"
            "    local_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)\n"
            "    solved = jrb_mat.jrb_mat_solve_action_point_jit(local_plan, x, symmetric=True)\n"
            "    return jnp.sum(di.midpoint(solved))\n"
            "scale_sweep = jnp.linspace(0.75, 1.25, 24, dtype=jnp.float64)\n"
            "primal_vals = jax.vmap(mf_loss)(scale_sweep)\n"
            "grad_vals = jax.vmap(jax.grad(mf_loss))(scale_sweep)\n"
            "ad_df = pd.DataFrame({'scale': np.asarray(scale_sweep), 'primal': np.asarray(primal_vals), 'grad': np.asarray(grad_vals)})\n"
            "display(ad_df.head())\n"
            "ax = ad_df.plot(x='scale', y=['primal', 'grad'], figsize=(8, 4), title='Matrix-Free AD Validation')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'ad_validation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Summary\n\n"
            "Run the matrix-free contract, chassis, and AD-facing tests."
        ),
        nbf.v4.new_code_cell(
            "tests = run([\n"
            "    PYTHON, '-m', 'pytest', '-q',\n"
            "    'tests/test_jrb_mat_chassis.py',\n"
            "    'tests/test_jcb_mat_chassis.py',\n"
            "    'tests/test_jrb_mat_logdet_contracts.py',\n"
            "    'tests/test_jrb_mat_selected_inverse.py',\n"
            "    'tests/test_matrix_free_basic.py',\n"
            "    'tests/test_matrix_stack_contracts.py',\n"
            "    'tests/test_matfree_adjoints.py',\n"
            "], capture=True)\n"
            "print(tests.stdout)\n"
            "if tests.stderr:\n"
            "    print(tests.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'pytest_{JAX_MODE}.txt').write_text(tests.stdout + ('\\n' + tests.stderr if tests.stderr else ''), encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark Summary\n\n"
            "Run the matrix-free Krylov benchmark on a reduced section set and structure the emitted metrics."
        ),
        nbf.v4.new_code_cell(
            "completed = run([\n"
            "    PYTHON, 'benchmarks/benchmark_matrix_free_krylov.py',\n"
            "    '--n-real', '12', '--n-complex', '8', '--steps-real', '6', '--steps-complex', '6', '--warmup', '1', '--runs', '2',\n"
            "    '--sections', 'real,complex',\n"
            "], capture=True)\n"
            "print(completed.stdout)\n"
            "rows = []\n"
            "for line in completed.stdout.splitlines():\n"
            "    if ': ' not in line:\n"
            "        continue\n"
            "    key, value = line.split(': ', 1)\n"
            "    if key in {'warmup', 'runs', 'plan_precompile', 'sections'} or key.startswith('[matrix_free_krylov]'):\n"
            "        continue\n"
            "    try:\n"
            "        rows.append({'metric': key, 'seconds': float(value)})\n"
            "    except ValueError:\n"
            "        pass\n"
            "bench_df = pd.DataFrame(rows).sort_values('seconds')\n"
            "bench_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'matrix_free_benchmark_summary_{JAX_MODE}.csv', index=False)\n"
            "display(bench_df.head(20))"
        ),
        nbf.v4.new_markdown_cell(
            "## Comparison / Contrast\n\n"
            "Compare dense-operator, sparse-operator, and solve/logdet-facing matrix-free usage so callers can decide when to stay dense, when to adapt sparse structure, and when to move fully into operator-free execution."
        ),
        nbf.v4.new_code_cell(
            "compare_df = bench_df[bench_df['metric'].str.contains('solve|matvec|logdet', regex=True, na=False)].copy()\n"
            "display(compare_df.head(20))"
        ),
        nbf.v4.new_markdown_cell(
            "## Diagnostics\n\n"
            "Matrix-free diagnostics are part of the production surface, so keep a compact summary of compile and execution metrics."
        ),
        nbf.v4.new_code_cell(
            "diag_rows = bench_df[bench_df['metric'].str.contains('compile|execute|cold|warm', regex=True, na=False)].head(20)\n"
            "diag_rows.to_csv(EXAMPLE_OUTPUT_ROOT / f'matrix_free_diagnostics_{JAX_MODE}.csv', index=False)\n"
            "display(diag_rows)"
        ),
        nbf.v4.new_code_cell(
            "top = bench_df.head(12).copy()\n"
            "ax = top.plot(x='metric', y='seconds', kind='barh', figsize=(10, 5), color='#8c5a3c', legend=False, title='Matrix-Free Benchmark Summary')\n"
            "ax.set_xlabel('seconds')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'matrix_free_benchmark_summary_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "summary_lines = [\n"
            "    f'# Example Matrix-Free Operator Surface Summary ({JAX_MODE})',\n"
            "    '',\n"
            "    f'- backend: `{runtime_payload[\"platform\"]}`',\n"
            "    f'- benchmark_rows: `{len(bench_df)}`',\n"
            "    f'- diagnostics_rows: `{len(diag_rows)}`',\n"
            "    '',\n"
            "    '## Fastest Metrics',\n"
            "    '',\n"
            "]\n"
            "for row in top.to_dict(orient='records'):\n"
            "    summary_lines.append(f\"- `{row['metric']}`: {row['seconds']:.6g}s\")\n"
            "(EXAMPLE_OUTPUT_ROOT / f'summary_{JAX_MODE}.md').write_text('\\n'.join(summary_lines) + '\\n', encoding='utf-8')\n"
            "display('\\n'.join(summary_lines[:12]))"
        ),
    ]
    return cells


def _fft_nufft_surface_notebook() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Example FFT NUFFT Surface\n\n"
            "Canonical transform notebook for DFT and NUFFT production surfaces."
        ),
        *_common_setup_cells("example_fft_nufft_surface"),
        nbf.v4.new_markdown_cell(
            "## Direct Usage\n\n"
            "Exercise direct DFT and cached NUFFT paths on representative complex inputs."
        ),
        nbf.v4.new_code_cell(
            "import jax.numpy as jnp\n"
            "from arbplusjax import dft, nufft\n"
            "\n"
            "x = jnp.asarray([1.0 + 0.0j, 0.5 + 0.25j, -0.75 + 0.1j, 0.25 - 0.5j], dtype=jnp.complex128)\n"
            "points = jnp.asarray([0.1, 0.25, 0.55, 0.8], dtype=jnp.float64)\n"
            "values = jnp.asarray([1.0 + 0.1j, 0.5 - 0.2j, -0.25 + 0.4j, 0.75 + 0.0j], dtype=jnp.complex128)\n"
            "plan = nufft.nufft_type1_cached_prepare(points, 8, method='lanczos')\n"
            "transform_results = {\n"
            "    'dft': dft.dft_jit(x),\n"
            "    'nufft_type1': nufft.nufft_type1(points, values, 8, method='lanczos'),\n"
            "    'nufft_type1_cached': nufft.nufft_type1_cached_apply_jit(plan, values),\n"
            "}\n"
            "display(transform_results)"
        ),
        *_production_pattern_cells(
            "Production Pattern",
            "Transform workloads should cache NUFFT plans and reuse them across repeated calls. "
            "Keep method choice and output grid size fixed inside a service path; avoid switching between direct and Lanczos or changing mode counts request-to-request unless you expect recompiles.",
            "cached_type1 = nufft.nufft_type1_cached_prepare(points, 8, method='lanczos')\n"
            "transform_service = {\n"
            "    'dft_repeat': dft.dft_jit(x),\n"
            "    'nufft_cached_repeat': nufft.nufft_type1_cached_apply_jit(cached_type1, values),\n"
            "}\n"
            "display(transform_service)",
            "To extend transform benchmarks, add the new DFT/NUFFT case in `benchmark_fft_nufft.py` with a stable `name` field so the CSV-style summary in this notebook remains compatible."
        ),
        *_fast_jax_pattern_cells(
            "Transforms already have explicit compiled kernels. "
            "For fast JAX usage, keep the plan cached and compare the compiled repeated path against the direct result.",
            "fast_transform = nufft.nufft_type1_cached_apply_jit(plan, values)\n"
            "direct_transform = nufft.nufft_type1(points, values, 8, method='lanczos')\n"
            "display({'jit_shape': fast_transform.shape, 'jit_matches_direct': bool(jnp.allclose(fast_transform, direct_transform, rtol=1e-6, atol=1e-6))})"
        ),
        *_ad_pattern_cells(
            "Transform AD should be shown through a realistic loss on transform outputs rather than isolated scalar identities. "
            "This section differentiates a NUFFT energy objective and plots primal versus gradient over a scale sweep.",
            "import jax\n"
            "base_values = values\n"
            "def transform_loss(scale):\n"
            "    scaled = scale * base_values\n"
            "    out = nufft.nufft_type1_cached_apply_jit(plan, scaled)\n"
            "    return jnp.real(jnp.vdot(out, out))\n"
            "scale_sweep = jnp.linspace(0.5, 1.5, 24, dtype=jnp.float64)\n"
            "primal_vals = jax.vmap(transform_loss)(scale_sweep)\n"
            "grad_vals = jax.vmap(jax.grad(transform_loss))(scale_sweep)\n"
            "ad_df = pd.DataFrame({'scale': np.asarray(scale_sweep), 'primal': np.asarray(primal_vals), 'grad': np.asarray(grad_vals)})\n"
            "display(ad_df.head())\n"
            "ax = ad_df.plot(x='scale', y=['primal', 'grad'], figsize=(8, 4), title='Transform AD Validation')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'ad_validation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Summary\n\n"
            "Run the DFT and NUFFT owner tests."
        ),
        nbf.v4.new_code_cell(
            "tests = run([\n"
            "    PYTHON, '-m', 'pytest', '-q',\n"
            "    'tests/test_dft_chassis.py',\n"
            "    'tests/test_nufft.py',\n"
            "    'tests/test_dft_parity.py',\n"
            "], capture=True)\n"
            "print(tests.stdout)\n"
            "if tests.stderr:\n"
            "    print(tests.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'pytest_{JAX_MODE}.txt').write_text(tests.stdout + ('\\n' + tests.stderr if tests.stderr else ''), encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark Summary\n\n"
            "Run the transform benchmark and parse the emitted CSV."
        ),
        nbf.v4.new_code_cell(
            "completed = run([PYTHON, 'benchmarks/benchmark_fft_nufft.py'], capture=True)\n"
            "print(completed.stdout)\n"
            "bench_df = pd.read_csv(io.StringIO(completed.stdout))\n"
            "bench_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'fft_nufft_benchmark_summary_{JAX_MODE}.csv', index=False)\n"
            "display(bench_df.head(20))"
        ),
        nbf.v4.new_markdown_cell(
            "## Plots\n\n"
            "Plot the fastest transform cases."
        ),
        nbf.v4.new_code_cell(
            "top = bench_df.sort_values('time_s').head(12)\n"
            "ax = top.plot(x='name', y='time_s', kind='barh', figsize=(10, 5), color='#3f6b5b', legend=False, title='FFT/NUFFT Benchmark Summary')\n"
            "ax.set_xlabel('time_s')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'fft_nufft_benchmark_summary_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Optional Diagnostics\n\n"
            "GPU-focused runs remain optional and depend on the installed JAX runtime."
        ),
        nbf.v4.new_code_cell(
            "summary_lines = [\n"
            "    f'# Example FFT NUFFT Surface Summary ({JAX_MODE})',\n"
            "    '',\n"
            "    f'- backend: `{runtime_payload[\"platform\"]}`',\n"
            "    f'- benchmark_rows: `{len(bench_df)}`',\n"
            "    '',\n"
            "    '## Fastest Metrics',\n"
            "    '',\n"
            "]\n"
            "for row in top.to_dict(orient='records'):\n"
            "    summary_lines.append(f\"- `{row['name']}`: {row['time_s']:.6g}s\")\n"
            "(EXAMPLE_OUTPUT_ROOT / f'summary_{JAX_MODE}.md').write_text('\\n'.join(summary_lines) + '\\n', encoding='utf-8')\n"
            "display('\\n'.join(summary_lines[:12]))"
        ),
    ]
    return cells


def _dirichlet_surface_notebook() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Example Dirichlet Surface\n\n"
            "Canonical Dirichlet family notebook for real interval zeta/eta and complex-box `acb_dirichlet` point/basic surfaces."
        ),
        *_common_setup_cells("example_dirichlet_surface"),
        nbf.v4.new_markdown_cell(
            "## Direct Usage\n\n"
            "Construct representative real intervals and complex boxes, then evaluate Dirichlet zeta and eta on both the real and complex helper surfaces."
        ),
        nbf.v4.new_code_cell(
            "import jax.numpy as jnp\n"
            "from arbplusjax import acb_core, acb_dirichlet, api, dirichlet, double_interval as di\n"
            "\n"
            "s_real = di.interval(jnp.array([1.5, 2.0, 2.5], dtype=jnp.float64), jnp.array([1.55, 2.05, 2.55], dtype=jnp.float64))\n"
            "s_complex = acb_core.acb_box(\n"
            "    di.interval(jnp.array([1.5, 2.0, 2.5], dtype=jnp.float64), jnp.array([1.55, 2.05, 2.55], dtype=jnp.float64)),\n"
            "    di.interval(jnp.array([-0.2, -0.05, 0.1], dtype=jnp.float64), jnp.array([-0.15, 0.0, 0.15], dtype=jnp.float64)),\n"
            ")\n"
            "direct_results = {\n"
            "    'dirichlet_zeta': dirichlet.dirichlet_zeta_batch(s_real, n_terms=32),\n"
            "    'dirichlet_eta': dirichlet.dirichlet_eta_batch(s_real, n_terms=32),\n"
            "    'acb_dirichlet_zeta': acb_dirichlet.acb_dirichlet_zeta_batch(s_complex, n_terms=48),\n"
            "    'acb_dirichlet_eta': acb_dirichlet.acb_dirichlet_eta_batch(s_complex, n_terms=48),\n"
            "}\n"
            "display(direct_results)"
        ),
        *_production_pattern_cells(
            "Production Pattern",
            "For repeated Dirichlet usage, keep `n_terms`, `dtype`, and `pad_to` stable. "
            "Use the real batch JIT path directly for interval zeta/eta, and use the API-bound compiled point batch for the complex `acb_dirichlet` surfaces.",
            "real_zeta = dirichlet.dirichlet_zeta_batch_jit(s_real, n_terms=32)\n"
            "real_eta = dirichlet.dirichlet_eta_batch_jit(s_real, n_terms=32)\n"
            "real_zeta_basic = dirichlet.dirichlet_zeta_batch_prec_jit(s_real, n_terms=32, prec_bits=53)\n"
            "complex_zeta_bound = api.bind_point_batch_jit('acb_dirichlet_zeta', dtype='float64', pad_to=8, n_terms=48)\n"
            "complex_eta_bound = api.bind_point_batch_jit('acb_dirichlet_eta', dtype='float64', pad_to=8, n_terms=48)\n"
            "service_results = {\n"
            "    'real_zeta_jit': real_zeta,\n"
            "    'real_eta_jit': real_eta,\n"
            "    'real_zeta_basic': real_zeta_basic,\n"
            "    'complex_zeta_bound': complex_zeta_bound(s_complex),\n"
            "    'complex_eta_bound': complex_eta_bound(s_complex),\n"
            "}\n"
            "display(service_results)",
            "To extend Dirichlet benchmarks, add stable metric rows to `benchmark_dirichlet.py` and `benchmark_acb_dirichlet.py`, and keep the stdout summary format stable for notebook parsing."
        ),
        *_fast_jax_pattern_cells(
            "The complex Dirichlet helper family already has a compiled point-batch API path. "
            "Keep `n_terms` static and `pad_to` fixed so repeated compiled calls stay on the same kernel.",
            "import jax\n"
            "jit_bound = api.bind_point_batch_jit('acb_dirichlet_zeta', dtype='float64', pad_to=8, n_terms=48)\n"
            "jit_vals = jit_bound(s_complex)\n"
            "vmap_vals = jax.vmap(lambda s_i: api.eval_point('acb_dirichlet_zeta', s_i, n_terms=48))(s_complex)\n"
            "display({'jit_shape': jit_vals.shape, 'jit_matches_vmap': bool(jnp.allclose(jit_vals, vmap_vals, rtol=1e-10, atol=1e-10))})"
        ),
        *_ad_pattern_cells(
            "AD should be shown on the actual Dirichlet public surfaces. "
            "This section differentiates the real interval midpoint path and the real part of the complex-box zeta path, then plots primal and gradient values.",
            "import jax\n"
            "sweep = jnp.linspace(1.3, 2.7, 24, dtype=jnp.float64)\n"
            "real_primal = jax.vmap(lambda x: jnp.squeeze(di.midpoint(dirichlet.dirichlet_zeta(di.interval(x, x), n_terms=32))))(sweep)\n"
            "real_grad = jax.vmap(jax.grad(lambda x: jnp.squeeze(di.midpoint(dirichlet.dirichlet_zeta(di.interval(x, x), n_terms=32)))))(sweep)\n"
            "complex_primal = jax.vmap(lambda x: jnp.real(acb_core.acb_midpoint(acb_dirichlet.acb_dirichlet_zeta(acb_core.acb_box(di.interval(x, x), di.interval(0.2, 0.2)), n_terms=48))))(sweep)\n"
            "complex_grad = jax.vmap(jax.grad(lambda x: jnp.real(acb_core.acb_midpoint(acb_dirichlet.acb_dirichlet_zeta(acb_core.acb_box(di.interval(x, x), di.interval(0.2, 0.2)), n_terms=48))))))(sweep)\n"
            "ad_df = pd.DataFrame({'s': [float(v) for v in sweep], 'real_primal': [float(v) for v in real_primal], 'real_grad': [float(v) for v in real_grad], 'complex_primal': [float(v) for v in complex_primal], 'complex_grad': [float(v) for v in complex_grad]})\n"
            "display(ad_df.head())\n"
            "ax = ad_df.plot(x='s', y=['real_primal', 'real_grad', 'complex_primal', 'complex_grad'], figsize=(10, 4), title='Dirichlet AD Validation')\n"
            "ax.set_ylabel('value')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'ad_validation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Summary\n\n"
            "Run the Dirichlet chassis and engineering tests that back the real and complex helper surfaces."
        ),
        nbf.v4.new_code_cell(
            "tests = run([\n"
            "    PYTHON, '-m', 'pytest', '-q',\n"
            "    'tests/test_dirichlet_chassis.py',\n"
            "    'tests/test_acb_dirichlet_chassis.py',\n"
            "    'tests/test_dirichlet_engineering.py',\n"
            "], capture=True)\n"
            "print(tests.stdout)\n"
            "if tests.stderr:\n"
            "    print(tests.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'pytest_{JAX_MODE}.txt').write_text(tests.stdout + ('\\n' + tests.stderr if tests.stderr else ''), encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark Summary\n\n"
            "Run the real and complex Dirichlet benchmark entrypoints and summarize their reported warm timings."
        ),
        nbf.v4.new_code_cell(
            "real_report = EXAMPLE_OUTPUT_ROOT / f'dirichlet_real_{JAX_MODE}.json'\n"
            "complex_report = EXAMPLE_OUTPUT_ROOT / f'dirichlet_complex_{JAX_MODE}.json'\n"
            "run([PYTHON, 'benchmarks/benchmark_dirichlet.py', '--which', 'zeta', '--samples', '512', '--terms', '32', '--dtype', JAX_DTYPE, '--jax-mode', JAX_MODE, '--output', str(real_report)])\n"
            "run([PYTHON, 'benchmarks/benchmark_acb_dirichlet.py', '--which', 'zeta', '--samples', '512', '--terms', '48', '--dtype', JAX_DTYPE, '--jax-mode', JAX_MODE, '--output', str(complex_report)])\n"
            "bench_rows = []\n"
            "for path in (real_report, complex_report):\n"
            "    payload = json.loads(path.read_text())\n"
            "    bench_rows.extend(payload['records'])\n"
            "bench_df = pd.DataFrame(bench_rows)\n"
            "bench_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'dirichlet_benchmark_summary_{JAX_MODE}.csv', index=False)\n"
            "display(bench_df[['implementation', 'operation', 'dtype', 'warm_time_s']])"
        ),
        nbf.v4.new_markdown_cell(
            "## Comparison Summary\n\n"
            "Run the existing reference compare scripts when the local C reference libraries are available; otherwise record the missing-reference state explicitly."
        ),
        nbf.v4.new_code_cell(
            "compare_cmds = [\n"
            "    [PYTHON, 'benchmarks/compare_dirichlet.py', '--samples', '256', '--terms', '24', '--which', 'zeta'],\n"
            "    [PYTHON, 'benchmarks/compare_acb_dirichlet.py', '--samples', '256', '--terms', '24'],\n"
            "]\n"
            "comparison_rows = []\n"
            "for cmd in compare_cmds:\n"
            "    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=RUN_ENV, text=True, capture_output=True)\n"
            "    comparison_rows.append({'script': cmd[1], 'returncode': completed.returncode, 'stdout': completed.stdout[-2000:], 'stderr': completed.stderr[-2000:]})\n"
            "    print(completed.stdout)\n"
            "    if completed.stderr:\n"
            "        print(completed.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'comparison_status_{JAX_MODE}.json').write_text(json.dumps(comparison_rows, indent=2) + '\\n', encoding='utf-8')\n"
            "display(pd.DataFrame(comparison_rows)[['script', 'returncode']])"
        ),
        nbf.v4.new_markdown_cell(
            "## Plots\n\n"
            "Plot the benchmark summary and the real midpoint relation `eta = (1 - 2^(1-s)) zeta` over the representative sweep."
        ),
        nbf.v4.new_code_cell(
            "eta_mid = jax.vmap(lambda x: jnp.squeeze(di.midpoint(dirichlet.dirichlet_eta(di.interval(x, x), n_terms=32))))(sweep)\n"
            "relation_df = pd.DataFrame({'s': [float(v) for v in sweep], 'zeta_mid': [float(v) for v in real_primal], 'eta_mid': [float(v) for v in eta_mid]})\n"
            "relation_df['eta_from_zeta'] = (1.0 - 2.0 ** (1.0 - relation_df['s'])) * relation_df['zeta_mid']\n"
            "ax = bench_df.plot(x='implementation', y='warm_time_s', kind='bar', figsize=(8, 4), title='Dirichlet Warm Time Summary')\n"
            "ax.set_ylabel('warm_time_s')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'dirichlet_benchmark_summary_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()\n"
            "ax = relation_df.plot(x='s', y=['eta_mid', 'eta_from_zeta'], figsize=(8, 4), title='Dirichlet Eta Relation Check')\n"
            "ax.set_ylabel('value')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'dirichlet_eta_relation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
    ]
    return cells


def _gamma_family_surface_notebook() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Example Gamma Family Surface\n\n"
            "Canonical gamma-family notebook covering incomplete gamma and related routed API metadata."
        ),
        *_common_setup_cells("example_gamma_family_surface"),
        nbf.v4.new_markdown_cell(
            "## Direct Usage\n\n"
            "Exercise lower/upper incomplete gamma surfaces, complement identities, and diagnostics."
        ),
        nbf.v4.new_code_cell(
            "import jax.numpy as jnp\n"
            "from arbplusjax import api\n"
            "\n"
            "s = jnp.asarray([1.5, 2.5, 3.5], dtype=jnp.float64)\n"
            "z = jnp.asarray([0.75, 1.25, 1.75], dtype=jnp.float64)\n"
            "gamma_results = {\n"
            "    'upper_point': api.eval_point_batch('incomplete_gamma_upper', s, z, method='quadrature', regularized=True),\n"
            "    'lower_point': api.eval_point_batch('incomplete_gamma_lower', s, z, method='quadrature', regularized=True),\n"
            "    'upper_basic': api.eval_interval_batch('incomplete_gamma_upper', s, z, mode='basic', method='quadrature', regularized=True, prec_bits=53),\n"
            "    'metadata': api.get_public_function_metadata('incomplete_gamma_upper'),\n"
            "}\n"
            "value, diagnostics = api.incomplete_gamma_upper(jnp.float64(0.75), jnp.float64(0.05), mode='point', method='auto', return_diagnostics=True)\n"
            "display(gamma_results)\n"
            "display({'diagnostic_method': diagnostics.method, 'fallback_used': diagnostics.fallback_used, 'value': value})"
        ),
        *_production_pattern_cells(
            "Production Pattern",
            "Gamma-family service calls should bind the method and precision policy up front and then reuse the bound callable over stable batch shapes. "
            "Diagnostics are optional but should be sampled in staging so fallback behavior is visible before deployment.",
            "gamma_bound = api.bind_point_batch('incomplete_gamma_upper', dtype='float64', pad_to=8, method='quadrature', regularized=True)\n"
            "gamma_interval_bound = api.bind_interval_batch('incomplete_gamma_upper', mode='basic', dtype='float64', pad_to=8, prec_bits=53, method='quadrature', regularized=True)\n"
            "gamma_service = {\n"
            "    'point_bound': gamma_bound(s, z),\n"
            "    'interval_bound': gamma_interval_bound(s, z),\n"
            "}\n"
            "display(gamma_service)",
            "To benchmark more gamma-family functions, extend the representative operation list in `benchmark_special_function_service_api.py` or add a dedicated comparison/benchmark entrypoint if the function has special reference needs."
        ),
        *_fast_jax_pattern_cells(
            "Special-function fast JAX at the API level should use the compiled point-batch binder on a safe box with fixed method policy.",
            "import jax\n"
            "gamma_fast = api.bind_point_batch_jit('incomplete_gamma_upper', dtype='float64', pad_to=8, method='quadrature', regularized=True)\n"
            "gamma_fast_out = gamma_fast(s, z)\n"
            "gamma_vmap = jax.vmap(lambda s_i, z_i: api.eval_point('incomplete_gamma_upper', s_i, z_i, dtype='float64', method='quadrature', regularized=True))(s, z)\n"
            "display({'jit_shape': gamma_fast_out.shape, 'jit_matches_vmap': bool(jnp.allclose(gamma_fast_out, gamma_vmap, rtol=1e-6, atol=1e-6))})"
        ),
        *_ad_pattern_cells(
            "Special-function AD should be shown on the production-facing differentiated surface with explicit method policy. "
            "This section validates incomplete-gamma derivatives over a sweep and plots primal versus gradient.",
            "import jax\n"
            "s_fixed = jnp.asarray(2.5, dtype=jnp.float64)\n"
            "def gamma_loss(zv):\n"
            "    return api.incomplete_gamma_upper(s_fixed, zv, mode='point', method='quadrature')\n"
            "z_sweep = jnp.linspace(0.25, 2.0, 32, dtype=jnp.float64)\n"
            "primal_vals = jax.vmap(gamma_loss)(z_sweep)\n"
            "grad_vals = jax.vmap(jax.grad(gamma_loss))(z_sweep)\n"
            "ad_df = pd.DataFrame({'z': np.asarray(z_sweep), 'primal': np.asarray(primal_vals), 'grad': np.asarray(grad_vals)})\n"
            "display(ad_df.head())\n"
            "ax = ad_df.plot(x='z', y=['primal', 'grad'], figsize=(8, 4), title='Gamma AD Validation')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'ad_validation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Summary\n\n"
            "Run the incomplete-gamma and hardening tests."
        ),
        nbf.v4.new_code_cell(
            "tests = run([\n"
            "    PYTHON, '-m', 'pytest', '-q',\n"
            "    'tests/test_incomplete_gamma.py',\n"
            "    'tests/test_gamma_hardening.py',\n"
            "    'tests/test_api_metadata.py',\n"
            "], capture=True)\n"
            "print(tests.stdout)\n"
            "if tests.stderr:\n"
            "    print(tests.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'pytest_{JAX_MODE}.txt').write_text(tests.stdout + ('\\n' + tests.stderr if tests.stderr else ''), encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark / Comparison Summary\n\n"
            "Use the existing gamma comparison benchmarks when local reference libraries are available."
        ),
        nbf.v4.new_code_cell(
            "compare_cmds = [\n"
            "    [PYTHON, 'benchmarks/benchmark_gamma_compare.py', '--arb-repo', str(REPO_ROOT), '--iters', '128'],\n"
            "    [PYTHON, 'benchmarks/benchmark_loggamma_compare.py', '--arb-repo', str(REPO_ROOT), '--iters', '128'],\n"
            "]\n"
            "rows = []\n"
            "for cmd in compare_cmds:\n"
            "    try:\n"
            "        completed = run(cmd, capture=True)\n"
            "        print(completed.stdout)\n"
            "        rows.append({'script': Path(cmd[1]).name, 'status': 'ok', 'tail': completed.stdout[-2000:]})\n"
            "    except subprocess.CalledProcessError as exc:\n"
            "        print(exc.stdout)\n"
            "        print(exc.stderr)\n"
            "        rows.append({'script': Path(cmd[1]).name, 'status': 'failed_or_unavailable', 'tail': ((exc.stdout or '') + '\\n' + (exc.stderr or ''))[-2000:]})\n"
            "bench_df = pd.DataFrame(rows)\n"
            "bench_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'gamma_compare_status_{JAX_MODE}.csv', index=False)\n"
            "display(bench_df)"
        ),
        nbf.v4.new_markdown_cell(
            "## Optional Diagnostics\n\n"
            "This tranche relies on the function-returned diagnostics objects rather than a separate compile-diagnostics benchmark."
        ),
        nbf.v4.new_code_cell(
            "summary_lines = [\n"
            "    f'# Example Gamma Family Surface Summary ({JAX_MODE})',\n"
            "    '',\n"
            "    f'- backend: `{runtime_payload[\"platform\"]}`',\n"
            "    f'- comparison_rows: `{len(bench_df)}`',\n"
            "    f'- auto_method: `{diagnostics.method}`',\n"
            "    '',\n"
            "    '## Comparison Scripts',\n"
            "    '',\n"
            "]\n"
            "for row in bench_df.to_dict(orient='records'):\n"
            "    summary_lines.append(f\"- `{row['script']}`: `{row['status']}`\")\n"
            "(EXAMPLE_OUTPUT_ROOT / f'summary_{JAX_MODE}.md').write_text('\\n'.join(summary_lines) + '\\n', encoding='utf-8')\n"
            "display('\\n'.join(summary_lines[:12]))"
        ),
    ]
    return cells


def _barnes_double_gamma_surface_notebook() -> list:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Example Barnes Double Gamma Surface\n\n"
            "Canonical Barnes and double-gamma notebook for the production special-function tranche."
        ),
        *_common_setup_cells("example_barnes_double_gamma_surface"),
        nbf.v4.new_markdown_cell(
            "## Direct Usage\n\n"
            "Exercise Barnes G and double-gamma public surfaces, including diagnostics."
        ),
        nbf.v4.new_code_cell(
            "import jax.numpy as jnp\n"
            "from arbplusjax import api, double_gamma\n"
            "\n"
            "z = jnp.asarray(1.7 + 0.1j, dtype=jnp.complex128)\n"
            "tau = jnp.float64(0.5)\n"
            "barnes_results = {\n"
            "    'barnesg': api.eval_point('acb_barnes_g', z),\n"
            "    'double_gamma_legacy': double_gamma.bdg_barnesdoublegamma(z, tau, prec_bits=80),\n"
            "    'double_gamma_ifj': double_gamma.ifj_barnesdoublegamma(z, tau, dps=60),\n"
            "}\n"
            "diagnostics = double_gamma.ifj_barnesdoublegamma_diagnostics(0.2 + 0.05j, 1.0, dps=60, max_m_cap=256)\n"
            "display(barnes_results)\n"
            "display({'m_base': diagnostics.m_base, 'm_used': diagnostics.m_used, 'n_shift': diagnostics.n_shift, 'm_capped': diagnostics.m_capped})"
        ),
        *_production_pattern_cells(
            "Production Pattern",
            "Barnes and double-gamma usage should keep the chosen implementation, precision, and diagnostics policy explicit. "
            "These are not hot scalar helpers; production code should avoid switching precision knobs per call unless that is a deliberate fallback path.",
            "barnes_service = {\n"
            "    'legacy_fixed_prec': double_gamma.bdg_barnesdoublegamma(z, tau, prec_bits=80),\n"
            "    'ifj_fixed_dps': double_gamma.ifj_barnesdoublegamma(z, tau, dps=60),\n"
            "}\n"
            "display(barnes_service)",
            "To extend Barnes/double-gamma benchmarks, add new printed metrics in `benchmark_barnes_double_gamma.py` or split out a schema-backed service benchmark if repeated-call API usage becomes a primary concern."
        ),
        *_fast_jax_pattern_cells(
            "Barnes-family point evaluation is still more specialized than the lightweight scalar helpers. "
            "The fast-JAX proof here is a family-owned compiled derivative path rather than a generic repeated batch service.",
            "import jax\n"
            "barnes_fast = jax.jit(lambda xs: jax.vmap(lambda t: jnp.real(double_gamma.ifj_barnesdoublegamma(jnp.asarray(t + 0.05j, dtype=jnp.complex128), 1.0, dps=60)))(xs))\n"
            "barnes_x = jnp.linspace(0.8, 2.0, 8, dtype=jnp.float64)\n"
            "barnes_fast_out = barnes_fast(barnes_x)\n"
            "display({'jit_shape': barnes_fast_out.shape, 'finite': bool(jnp.all(jnp.isfinite(barnes_fast_out)))})"
        ),
        *_ad_pattern_cells(
            "Barnes-family AD should be shown explicitly because these surfaces are more specialized and their derivative expectations are less obvious to users. "
            "This section differentiates the IFJ log-form path over a real sweep and plots primal versus gradient.",
            "import jax\n"
            "def barnes_loss(xv):\n"
            "    val = double_gamma.ifj_barnesdoublegamma(jnp.asarray(xv + 0.05j, dtype=jnp.complex128), 1.0, dps=60)\n"
            "    return jnp.real(val)\n"
            "x_sweep = jnp.linspace(0.8, 2.0, 24, dtype=jnp.float64)\n"
            "primal_vals = jax.vmap(barnes_loss)(x_sweep)\n"
            "grad_vals = jax.vmap(jax.grad(barnes_loss))(x_sweep)\n"
            "ad_df = pd.DataFrame({'x': np.asarray(x_sweep), 'primal': np.asarray(primal_vals), 'grad': np.asarray(grad_vals)})\n"
            "display(ad_df.head())\n"
            "ax = ad_df.plot(x='x', y=['primal', 'grad'], figsize=(8, 4), title='Barnes AD Validation')\n"
            "plt.tight_layout()\n"
            "plt.savefig(EXAMPLE_OUTPUT_ROOT / f'ad_validation_{JAX_MODE}.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Summary\n\n"
            "Run the Barnes/double-gamma contract tests."
        ),
        nbf.v4.new_code_cell(
            "tests = run([\n"
            "    PYTHON, '-m', 'pytest', '-q',\n"
            "    'tests/test_barnes_tier1.py',\n"
            "    'tests/test_double_gamma_contracts.py',\n"
            "    'tests/test_shahen_double_gamma.py',\n"
            "], capture=True)\n"
            "print(tests.stdout)\n"
            "if tests.stderr:\n"
            "    print(tests.stderr)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'pytest_{JAX_MODE}.txt').write_text(tests.stdout + ('\\n' + tests.stderr if tests.stderr else ''), encoding='utf-8')"
        ),
        nbf.v4.new_markdown_cell(
            "## Benchmark Summary\n\n"
            "Run the Barnes/double-gamma benchmark and parse the key-value output."
        ),
        nbf.v4.new_code_cell(
            "completed = run([PYTHON, 'benchmarks/benchmark_barnes_double_gamma.py'], capture=True)\n"
            "print(completed.stdout)\n"
            "rows = []\n"
            "for line in completed.stdout.splitlines():\n"
            "    if ': ' not in line:\n"
            "        continue\n"
            "    key, value = line.split(': ', 1)\n"
            "    try:\n"
            "        rows.append({'metric': key, 'value': float(value)})\n"
            "    except ValueError:\n"
            "        rows.append({'metric': key, 'value': value})\n"
            "bench_df = pd.DataFrame(rows)\n"
            "bench_df.to_csv(EXAMPLE_OUTPUT_ROOT / f'barnes_double_gamma_summary_{JAX_MODE}.csv', index=False)\n"
            "display(bench_df)"
        ),
        nbf.v4.new_markdown_cell(
            "## Optional Diagnostics\n\n"
            "The IFJ diagnostics object is the primary hardening signal for this notebook."
        ),
        nbf.v4.new_code_cell(
            "summary_lines = [\n"
            "    f'# Example Barnes Double Gamma Surface Summary ({JAX_MODE})',\n"
            "    '',\n"
            "    f'- backend: `{runtime_payload[\"platform\"]}`',\n"
            "    f'- benchmark_rows: `{len(bench_df)}`',\n"
            "    f'- diagnostics_m_used: `{diagnostics.m_used}`',\n"
            "    '',\n"
            "    '## Key Metrics',\n"
            "    '',\n"
            "]\n"
            "for row in bench_df.to_dict(orient='records')[:12]:\n"
            "    summary_lines.append(f\"- `{row['metric']}`: `{row['value']}`\")\n"
            "(EXAMPLE_OUTPUT_ROOT / f'summary_{JAX_MODE}.md').write_text('\\n'.join(summary_lines) + '\\n', encoding='utf-8')\n"
            "display('\\n'.join(summary_lines[:14]))"
        ),
    ]
    return cells


def main() -> None:
    notebooks = {
        "example_core_scalar_surface.ipynb": _core_scalar_notebook(),
        "example_api_surface.ipynb": _api_surface_notebook(),
        "example_dense_matrix_surface.ipynb": _dense_matrix_surface_notebook(),
        "example_sparse_matrix_surface.ipynb": _sparse_matrix_surface_notebook(),
        "example_matrix_free_operator_surface.ipynb": _matrix_free_operator_surface_notebook(),
        "example_fft_nufft_surface.ipynb": _fft_nufft_surface_notebook(),
        "example_dirichlet_surface.ipynb": _dirichlet_surface_notebook(),
        "example_gamma_family_surface.ipynb": _gamma_family_surface_notebook(),
        "example_barnes_double_gamma_surface.ipynb": _barnes_double_gamma_surface_notebook(),
    }
    for name, cells in notebooks.items():
        _write_notebook(EXAMPLES_DIR / name, cells)


if __name__ == "__main__":
    main()
