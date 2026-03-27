from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> dict[str, bool]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("JAX_PLATFORMS", "cpu")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_importing_api_tracks_double_gamma_and_cubesselk_cold_path() -> None:
    payload = _run_python(
        """
import json
import sys
from arbplusjax import api
print(json.dumps({
    "double_gamma": "arbplusjax.double_gamma" in sys.modules,
    "cubesselk": "arbplusjax.cubesselk" in sys.modules,
    "dirichlet_wrappers": "arbplusjax.dirichlet_wrappers" in sys.modules,
    "modular_elliptic_wrappers": "arbplusjax.modular_elliptic_wrappers" in sys.modules,
}))
"""
    )
    assert payload == {
        "double_gamma": False,
        "cubesselk": False,
        "dirichlet_wrappers": False,
        "modular_elliptic_wrappers": False,
    }


def test_double_gamma_point_call_stays_off_other_family_modules() -> None:
    payload = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api

_ = api.eval_point(
    "bdg_barnesgamma2",
    jnp.asarray(1.2 + 0.1j, dtype=jnp.complex64),
    jnp.asarray(0.9 + 0.0j, dtype=jnp.complex64),
    dtype="float32",
)
print(json.dumps({
    "double_gamma": "arbplusjax.double_gamma" in sys.modules,
    "cubesselk": "arbplusjax.cubesselk" in sys.modules,
    "dirichlet_wrappers": "arbplusjax.dirichlet_wrappers" in sys.modules,
    "modular_elliptic_wrappers": "arbplusjax.modular_elliptic_wrappers" in sys.modules,
}))
"""
    )
    assert payload == {
        "double_gamma": True,
        "cubesselk": False,
        "dirichlet_wrappers": False,
        "modular_elliptic_wrappers": False,
    }


def test_importing_api_keeps_cubesselk_off_the_cold_path() -> None:
    payload = _run_python(
        """
import json
import sys
from arbplusjax import api
print(json.dumps({
    "cubesselk": "arbplusjax.cubesselk" in sys.modules,
    "hypgeom_wrappers": "arbplusjax.hypgeom_wrappers" in sys.modules,
}))
"""
    )
    assert payload == {
        "cubesselk": False,
        "hypgeom_wrappers": False,
    }


def test_cuda_besselk_point_call_stays_off_interval_hypgeom_wrappers() -> None:
    payload = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api

_ = api.bind_point_batch_jit("cuda_besselk", dtype="float64", pad_to=4)(
    jnp.array([0.5, 1.0], dtype=jnp.float64),
    jnp.array([1.5, 2.0], dtype=jnp.float64),
)
print(json.dumps({
    "cubesselk": "arbplusjax.cubesselk" in sys.modules,
    "hypgeom_wrappers": "arbplusjax.hypgeom_wrappers" in sys.modules,
}))
"""
    )
    assert payload == {
        "cubesselk": True,
        "hypgeom_wrappers": False,
    }


def test_importing_api_keeps_sparse_matrix_families_off_the_cold_path() -> None:
    payload = _run_python(
        """
import json
import sys
from arbplusjax import api
print(json.dumps({
    "srb_mat": "arbplusjax.srb_mat" in sys.modules,
    "scb_mat": "arbplusjax.scb_mat" in sys.modules,
    "arb_mat": "arbplusjax.arb_mat" in sys.modules,
    "acb_mat": "arbplusjax.acb_mat" in sys.modules,
    "mat_wrappers": "arbplusjax.mat_wrappers" in sys.modules,
}))
"""
    )
    assert payload == {
        "srb_mat": False,
        "scb_mat": False,
        "arb_mat": False,
        "acb_mat": False,
        "mat_wrappers": False,
    }


def test_sparse_cached_apply_point_calls_only_load_the_requested_family() -> None:
    payload = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api
from arbplusjax import sparse_common as sc

sparse_r = sc.SparseCSR(
    data=jnp.array([2.0, -1.0, 3.0], dtype=jnp.float64),
    indices=jnp.array([0, 1, 1], dtype=jnp.int32),
    indptr=jnp.array([0, 2, 3], dtype=jnp.int32),
    rows=2,
    cols=2,
    algebra="srb",
)
plan_r = api.eval_point("srb_mat_matvec_cached_prepare", sparse_r)
_ = api.bind_point_batch_jit("srb_mat_matvec_cached_apply", dtype="float64", pad_to=4)(
    plan_r,
    jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64),
)
after_srb = {
    "srb_mat": "arbplusjax.srb_mat" in sys.modules,
    "arb_mat": "arbplusjax.arb_mat" in sys.modules,
    "mat_wrappers": "arbplusjax.mat_wrappers" in sys.modules,
    "scb_mat": "arbplusjax.scb_mat" in sys.modules,
    "acb_mat": "arbplusjax.acb_mat" in sys.modules,
}

for name in ["arbplusjax.srb_mat", "arbplusjax.arb_mat"]:
    sys.modules.pop(name, None)

sparse_c = sc.SparseCSR(
    data=jnp.array([2.0 + 0.5j, -1.0 + 0.25j, 3.0 - 0.75j], dtype=jnp.complex64),
    indices=jnp.array([0, 1, 1], dtype=jnp.int32),
    indptr=jnp.array([0, 2, 3], dtype=jnp.int32),
    rows=2,
    cols=2,
    algebra="scb",
)
plan_c = api.eval_point("scb_mat_matvec_cached_prepare", sparse_c)
_ = api.bind_point_batch_jit("scb_mat_matvec_cached_apply", pad_to=4)(
    plan_c,
    jnp.array(
        [[1.0 + 0.5j, 2.0 - 0.25j], [3.0 + 0.75j, 4.0 - 0.5j]],
        dtype=jnp.complex64,
    ),
)
after_scb = {
    "scb_mat": "arbplusjax.scb_mat" in sys.modules,
    "acb_mat": "arbplusjax.acb_mat" in sys.modules,
    "mat_wrappers": "arbplusjax.mat_wrappers" in sys.modules,
    "srb_mat": "arbplusjax.srb_mat" in sys.modules,
    "arb_mat": "arbplusjax.arb_mat" in sys.modules,
}

print(json.dumps({"after_srb": after_srb, "after_scb": after_scb}))
"""
    )
    assert payload == {
        "after_srb": {
            "srb_mat": True,
            "arb_mat": True,
            "mat_wrappers": False,
            "scb_mat": False,
            "acb_mat": False,
        },
        "after_scb": {
            "scb_mat": True,
            "acb_mat": True,
            "mat_wrappers": False,
            "srb_mat": False,
            "arb_mat": False,
        },
    }


def test_importing_api_tracks_dirichlet_modular_elliptic_point_families() -> None:
    payload = _run_python(
        """
import json
import sys
from arbplusjax import api
print(json.dumps({
    "acb_dirichlet": "arbplusjax.acb_dirichlet" in sys.modules,
    "acb_modular": "arbplusjax.acb_modular" in sys.modules,
    "acb_elliptic": "arbplusjax.acb_elliptic" in sys.modules,
    "point_wrappers_dirichlet_modular": "arbplusjax.point_wrappers_dirichlet_modular" in sys.modules,
    "point_wrappers_elliptic": "arbplusjax.point_wrappers_elliptic" in sys.modules,
    "dirichlet_wrappers": "arbplusjax.dirichlet_wrappers" in sys.modules,
    "modular_elliptic_wrappers": "arbplusjax.modular_elliptic_wrappers" in sys.modules,
}))
"""
    )
    assert payload == {
        "acb_dirichlet": False,
        "acb_modular": False,
        "acb_elliptic": False,
        "point_wrappers_dirichlet_modular": False,
        "point_wrappers_elliptic": False,
        "dirichlet_wrappers": False,
        "modular_elliptic_wrappers": False,
    }


def test_dirichlet_modular_elliptic_point_calls_stay_off_wrapper_modules() -> None:
    payload = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api

_ = api.bind_point_batch_jit("acb_dirichlet_zeta", pad_to=4, n_terms=32)(
    jnp.array([0.75 + 2.0j, 0.8 + 2.5j], dtype=jnp.complex64),
)
after_dirichlet = {
    "acb_dirichlet": "arbplusjax.acb_dirichlet" in sys.modules,
    "acb_modular": "arbplusjax.acb_modular" in sys.modules,
    "acb_elliptic": "arbplusjax.acb_elliptic" in sys.modules,
    "point_wrappers_dirichlet_modular": "arbplusjax.point_wrappers_dirichlet_modular" in sys.modules,
    "point_wrappers_elliptic": "arbplusjax.point_wrappers_elliptic" in sys.modules,
    "dirichlet_wrappers": "arbplusjax.dirichlet_wrappers" in sys.modules,
    "modular_elliptic_wrappers": "arbplusjax.modular_elliptic_wrappers" in sys.modules,
}

_ = api.bind_point_batch_jit("acb_modular_j", pad_to=4)(
    jnp.array([0.2 + 0.9j, 0.25 + 1.0j], dtype=jnp.complex64),
)
after_modular = {
    "acb_dirichlet": "arbplusjax.acb_dirichlet" in sys.modules,
    "acb_modular": "arbplusjax.acb_modular" in sys.modules,
    "acb_elliptic": "arbplusjax.acb_elliptic" in sys.modules,
    "point_wrappers_dirichlet_modular": "arbplusjax.point_wrappers_dirichlet_modular" in sys.modules,
    "point_wrappers_elliptic": "arbplusjax.point_wrappers_elliptic" in sys.modules,
    "dirichlet_wrappers": "arbplusjax.dirichlet_wrappers" in sys.modules,
    "modular_elliptic_wrappers": "arbplusjax.modular_elliptic_wrappers" in sys.modules,
}

_ = api.bind_point_batch_jit("acb_elliptic_k", pad_to=4)(
    jnp.array([0.1 + 0.2j, 0.2 + 0.1j], dtype=jnp.complex64),
)
after_elliptic = {
    "acb_dirichlet": "arbplusjax.acb_dirichlet" in sys.modules,
    "acb_modular": "arbplusjax.acb_modular" in sys.modules,
    "acb_elliptic": "arbplusjax.acb_elliptic" in sys.modules,
    "point_wrappers_dirichlet_modular": "arbplusjax.point_wrappers_dirichlet_modular" in sys.modules,
    "point_wrappers_elliptic": "arbplusjax.point_wrappers_elliptic" in sys.modules,
    "dirichlet_wrappers": "arbplusjax.dirichlet_wrappers" in sys.modules,
    "modular_elliptic_wrappers": "arbplusjax.modular_elliptic_wrappers" in sys.modules,
}

print(json.dumps({
    "after_dirichlet": after_dirichlet,
    "after_modular": after_modular,
    "after_elliptic": after_elliptic,
}))
"""
    )
    assert payload == {
        "after_dirichlet": {
            "acb_dirichlet": False,
            "acb_modular": False,
            "acb_elliptic": False,
            "point_wrappers_dirichlet_modular": True,
            "point_wrappers_elliptic": False,
            "dirichlet_wrappers": False,
            "modular_elliptic_wrappers": False,
        },
        "after_modular": {
            "acb_dirichlet": False,
            "acb_modular": False,
            "acb_elliptic": False,
            "point_wrappers_dirichlet_modular": True,
            "point_wrappers_elliptic": False,
            "dirichlet_wrappers": False,
            "modular_elliptic_wrappers": False,
        },
        "after_elliptic": {
            "acb_dirichlet": False,
            "acb_modular": False,
            "acb_elliptic": True,
            "point_wrappers_dirichlet_modular": True,
            "point_wrappers_elliptic": True,
            "dirichlet_wrappers": False,
            "modular_elliptic_wrappers": False,
        },
    }
