from __future__ import annotations

from pathlib import Path

import nbformat as nbf


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
            "import json\n"
            "import os\n"
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
            "The notebook reports interpreter, selected JAX mode, and the active backend/device view."
        ),
        nbf.v4.new_code_cell(
            "print('python:', PYTHON)\n"
            "print('jax_mode:', JAX_MODE)\n"
            "print('jax_dtype:', JAX_DTYPE)\n"
            "runtime = run([PYTHON, 'tools/check_jax_runtime.py'], capture=True)\n"
            "print(runtime.stdout)\n"
            "runtime_payload = json.loads(runtime.stdout)\n"
            "(EXAMPLE_OUTPUT_ROOT / f'runtime_{JAX_MODE}.json').write_text(json.dumps(runtime_payload, indent=2) + '\\n', encoding='utf-8')"
        ),
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


def main() -> None:
    notebooks = {
        "example_core_scalar_surface.ipynb": _core_scalar_notebook(),
        "example_api_surface.ipynb": _api_surface_notebook(),
    }
    for name, cells in notebooks.items():
        _write_notebook(EXAMPLES_DIR / name, cells)


if __name__ == "__main__":
    main()
