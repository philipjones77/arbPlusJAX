from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def _base_cells(title: str, description: str, functions: str, samples: str, seeds: str, profile_name: str):
    return [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n{description}\n\n"
            "This notebook runs arbPlusJAX point/basic/adaptive/rigorous modes and compares to `c_chassis` where available."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import platform\n"
            "import subprocess\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "import pandas as pd\n"
            "\n"
            "def find_repo_root(start: Path) -> Path:\n"
            "    cur = start.resolve()\n"
            "    for p in [cur, *cur.parents]:\n"
            "        if (p / 'pyproject.toml').exists() and (p / 'src' / 'arbplusjax').exists() and (p / 'tools').exists():\n"
            "            return p\n"
            "    raise RuntimeError(f'Could not locate repo root from: {start}')\n"
            "\n"
            "REPO_ROOT = find_repo_root(Path.cwd())\n"
            "os.chdir(REPO_ROOT)\n"
            "if str(REPO_ROOT / 'src') not in sys.path:\n"
            "    sys.path.insert(0, str(REPO_ROOT / 'src'))\n"
            "\n"
            "PYTHON = os.getenv('ARBPLUSJAX_PYTHON', sys.executable)\n"
            "JAX_MODE = os.getenv('JAX_MODE', 'cpu').strip().lower()  # cpu or gpu\n"
            "if JAX_MODE not in {'cpu', 'gpu'}:\n"
            "    raise ValueError(f'JAX_MODE must be cpu or gpu, got: {JAX_MODE}')\n"
            "JAX_DTYPE = os.getenv('JAX_DTYPE', 'float64').strip().lower()\n"
            "if JAX_DTYPE not in {'float64', 'float32'}:\n"
            "    raise ValueError(f'JAX_DTYPE must be float64 or float32, got: {JAX_DTYPE}')\n"
            "SAMPLES = os.getenv('EXAMPLE_SAMPLES', '" + samples + "')\n"
            "SEEDS = os.getenv('EXAMPLE_SEEDS', '" + seeds + "')\n"
            "C_REF_DIR = os.getenv('ARB_C_REF_DIR', str(REPO_ROOT / 'stuff' / 'migration' / 'c_chassis' / 'build_linux_wsl'))\n"
            "\n"
            "def run(cmd: list[str], env_override: dict[str, str] | None = None):\n"
            "    print('\\n[cmd]', ' '.join(cmd))\n"
            "    env = os.environ.copy()\n"
            "    if env_override:\n"
            "        env.update(env_override)\n"
            "    cp = subprocess.run(cmd, cwd=REPO_ROOT, text=True, env=env)\n"
            "    if cp.returncode != 0:\n"
            "        raise RuntimeError(f'Command failed: {cmd}')\n"
            "\n"
            "print('platform:', platform.system(), platform.release())\n"
            "print('python:', PYTHON)\n"
            "print('jax_mode:', JAX_MODE)\n"
            "print('jax_dtype:', JAX_DTYPE)\n"
            "print('samples:', SAMPLES)\n"
            "print('seeds:', SEEDS)\n"
            "print('c_ref_dir:', C_REF_DIR)"
        ),
        nbf.v4.new_code_cell(
            "env = os.environ.copy()\n"
            "if JAX_MODE == 'gpu':\n"
            "    run_env = {'JAX_PLATFORMS': 'cuda'}\n"
            "    expected = 'gpu'\n"
            "else:\n"
            "    run_env = {'JAX_PLATFORMS': 'cpu'}\n"
            "    expected = 'cpu'\n"
            "\n"
            "run([PYTHON, 'tools/check_jax_runtime.py', '--expect-backend', expected], env_override=run_env)"
        ),
        nbf.v4.new_code_cell(
            "name = '" + profile_name + "_' + JAX_MODE\n"
            "cmd = [\n"
            "    PYTHON, 'tools/run_harness_profile.py',\n"
            "    '--name', name,\n"
            "    '--functions', '" + functions + "',\n"
            "    '--samples', SAMPLES,\n"
            "    '--seeds', SEEDS,\n"
            "    '--jax-mode', JAX_MODE,\n"
            "    '--jax-dtype', JAX_DTYPE,\n"
            "    '--c-ref-dir', C_REF_DIR,\n"
            "]\n"
            "run(cmd)\n"
            "OUT_DIR = REPO_ROOT / 'results' / 'benchmarks' / name\n"
            "CSV_PATH = OUT_DIR / 'profile_summary.csv'\n"
            "MD_PATH = OUT_DIR / 'profile_summary.md'\n"
            "print('summary csv:', CSV_PATH)\n"
            "print('summary md :', MD_PATH)"
        ),
        nbf.v4.new_code_cell(
            "df = pd.read_csv(CSV_PATH)\n"
            "display(df.head(20))\n"
            "\n"
            "num_cols = ['time_ms', 'mean_abs_err', 'containment_rate', 'peak_rss_mb']\n"
            "for c in num_cols:\n"
            "    if c in df.columns:\n"
            "        df[c] = pd.to_numeric(df[c], errors='coerce')\n"
            "\n"
            "agg = (\n"
            "    df.groupby('backend', dropna=False)[['time_ms', 'mean_abs_err', 'containment_rate', 'peak_rss_mb']]\n"
            "      .mean(numeric_only=True)\n"
            "      .sort_values('time_ms', na_position='last')\n"
            ")\n"
            "display(agg)"
        ),
    ]


def _write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nbf.write(nb, path)


def main() -> None:
    specs = [
        {
            "path": EXAMPLES_DIR / "example_core_modes_sweep.ipynb",
            "title": "Example Core Modes Sweep",
            "description": "Core transcendental functions (`exp..tanh`) across four JAX modes.",
            "functions": "exp,log,sqrt,sin,cos,tan,sinh,cosh,tanh",
            "samples": "300,600",
            "seeds": "0,1",
            "name": "example_core_modes",
        },
        {
            "path": EXAMPLES_DIR / "example_special_modes_sweep.ipynb",
            "title": "Example Special Modes Sweep",
            "description": "Special functions (`gamma`, `erf`, `erfc`, `barnesg`) in point/basic/adaptive/rigorous.",
            "functions": "gamma,erf,erfc,barnesg",
            "samples": "300,600",
            "seeds": "0,1",
            "name": "example_special_modes",
        },
        {
            "path": EXAMPLES_DIR / "example_bessel_modes_sweep.ipynb",
            "title": "Example Bessel Modes Sweep",
            "description": "Bessel-family functions (`bessel*`, `CubesselK`) with mode comparison and C containment.",
            "functions": "besselj,bessely,besseli,besselk,CubesselK",
            "samples": "240,480",
            "seeds": "0,1",
            "name": "example_bessel_modes",
        },
        {
            "path": EXAMPLES_DIR / "example_all_modes_sweep.ipynb",
            "title": "Example All-Libraries Modes Sweep",
            "description": "Broad moderate sweep over all current benchmark families.",
            "functions": "exp,log,sqrt,sin,cos,tan,sinh,cosh,tanh,gamma,erf,erfc,barnesg,besselj,bessely,besseli,besselk,CubesselK",
            "samples": "200,400",
            "seeds": "0,1",
            "name": "example_all_modes",
        },
    ]

    for s in specs:
        cells = _base_cells(
            title=s["title"],
            description=s["description"],
            functions=s["functions"],
            samples=s["samples"],
            seeds=s["seeds"],
            profile_name=s["name"],
        )
        _write_notebook(s["path"], cells)

    # Robust hypgeom notebook with an extra stress pass.
    hyp_cells = _base_cells(
        title="Example Hypgeom Robust Modes Sweep",
        description=(
            "Robust hypgeom-focused run with a standard pass and an additional stress pass "
            "at higher precision to stabilize branch-sensitive regions."
        ),
        functions="gamma,erf,erfc,besselj,bessely,besseli,besselk,CubesselK",
        samples="280,560",
        seeds="0,1",
        profile_name="example_hypgeom_robust",
    )
    hyp_cells.append(
        nbf.v4.new_code_cell(
            "# Extra robustness pass (smaller samples, higher precision bits)\n"
            "name2 = 'example_hypgeom_robust_stress_' + JAX_MODE\n"
            "cmd2 = [\n"
            "    PYTHON, 'tools/run_harness_profile.py',\n"
            "    '--name', name2,\n"
            "    '--functions', 'gamma,erf,erfc,besselj,bessely,besseli,besselk,CubesselK',\n"
            "    '--samples', '180,360',\n"
            "    '--seeds', '0,1,2',\n"
            "    '--jax-mode', JAX_MODE,\n"
            "    '--jax-dtype', JAX_DTYPE,\n"
            "    '--c-ref-dir', C_REF_DIR,\n"
            "    '--prec-bits', '256',\n"
            "]\n"
            "run(cmd2)\n"
            "CSV_PATH_2 = REPO_ROOT / 'results' / 'benchmarks' / name2 / 'profile_summary.csv'\n"
            "df2 = pd.read_csv(CSV_PATH_2)\n"
            "for c in ['time_ms', 'mean_abs_err', 'containment_rate', 'peak_rss_mb']:\n"
            "    if c in df2.columns:\n"
            "        df2[c] = pd.to_numeric(df2[c], errors='coerce')\n"
            "display(df2.groupby('backend')[['time_ms', 'mean_abs_err', 'containment_rate', 'peak_rss_mb']].mean(numeric_only=True))"
        )
    )
    _write_notebook(EXAMPLES_DIR / "example_hypgeom_robust_modes_sweep.ipynb", hyp_cells)


if __name__ == "__main__":
    main()
