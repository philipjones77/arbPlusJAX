import ast
from pathlib import Path

import jax.numpy as jnp

from arbplusjax import point_wrappers

from tests._test_checks import _check


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "arbplusjax"


def _public_defs(path: Path, prefix: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix) and not node.name.startswith("_"):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith(prefix) and not target.id.startswith("_"):
                    names.add(target.id)
    return names


def _core_surface(path: Path, prefix: str) -> list[str]:
    defs = _public_defs(path, prefix)
    return sorted(name for name in defs if f"{name}_prec" in defs and not name.endswith("_batch"))


def test_every_core_surface_function_has_a_point_wrapper():
    point_defs: set[str] = set()
    for path in (
        SRC / "point_wrappers.py",
        SRC / "point_wrappers_core.py",
        SRC / "point_wrappers_barnes.py",
        SRC / "point_wrappers_dirichlet_modular.py",
        SRC / "point_wrappers_elliptic.py",
    ):
        point_defs |= _public_defs(path, "")
    missing: list[str] = []
    for module_name, prefix in (("arb_core", "arb_"), ("acb_core", "acb_")):
        for name in _core_surface(SRC / f"{module_name}.py", prefix):
            wrapper = f"{name}_point"
            if wrapper not in point_defs:
                missing.append(wrapper)
    _check(not missing, f"missing point wrappers: {missing}")


def test_new_complex_point_wrappers_smoke():
    z = jnp.asarray(0.4 + 0.2j, dtype=jnp.complex128)
    w = jnp.asarray(1.3 + 0.1j, dtype=jnp.complex128)
    u = jnp.asarray(0.2 + 0.05j, dtype=jnp.complex128)

    unary = [
        point_wrappers.acb_exp_point,
        point_wrappers.acb_log_point,
        point_wrappers.acb_sqrt_point,
        point_wrappers.acb_rsqrt_point,
        point_wrappers.acb_sin_point,
        point_wrappers.acb_cos_point,
        point_wrappers.acb_tan_point,
        point_wrappers.acb_cot_point,
        point_wrappers.acb_sinh_point,
        point_wrappers.acb_cosh_point,
        point_wrappers.acb_tanh_point,
        point_wrappers.acb_sech_point,
        point_wrappers.acb_csch_point,
        point_wrappers.acb_sin_pi_point,
        point_wrappers.acb_cos_pi_point,
        point_wrappers.acb_tan_pi_point,
        point_wrappers.acb_cot_pi_point,
        point_wrappers.acb_csc_pi_point,
        point_wrappers.acb_sinc_point,
        point_wrappers.acb_sinc_pi_point,
        point_wrappers.acb_exp_pi_i_point,
        point_wrappers.acb_gamma_point,
        point_wrappers.acb_rgamma_point,
        point_wrappers.acb_lgamma_point,
        point_wrappers.acb_log_sin_pi_point,
        point_wrappers.acb_digamma_point,
        point_wrappers.acb_zeta_point,
        point_wrappers.acb_agm1_point,
        point_wrappers.acb_agm1_cpx_point,
    ]
    for fn in unary:
        out = fn(z)
        _check(bool(jnp.all(jnp.isfinite(jnp.real(out)))))

    binary = [
        (point_wrappers.acb_pow_point, (w, z)),
        (point_wrappers.acb_pow_arb_point, (w, z)),
        (point_wrappers.acb_hurwitz_zeta_point, (w, z)),
        (point_wrappers.acb_polylog_point, (w, z)),
        (point_wrappers.acb_agm_point, (w, z)),
    ]
    for fn, args in binary:
        out = fn(*args)
        _check(bool(jnp.all(jnp.isfinite(jnp.real(out)))))

    _check(bool(jnp.all(jnp.isfinite(jnp.real(point_wrappers.acb_pow_ui_point(w, 3))))))
    _check(bool(jnp.all(jnp.isfinite(jnp.real(point_wrappers.acb_pow_si_point(w, 3))))))
    _check(bool(jnp.all(jnp.isfinite(jnp.real(point_wrappers.acb_pow_fmpz_point(w, jnp.asarray(3.0)))))))
    _check(bool(jnp.all(jnp.isfinite(jnp.real(point_wrappers.acb_sqr_point(w))))))
    _check(bool(jnp.all(jnp.isfinite(jnp.real(point_wrappers.acb_root_ui_point(w, 3))))))
    _check(bool(jnp.all(jnp.isfinite(jnp.real(point_wrappers.acb_polygamma_point(w, 1))))))
    _check(bool(jnp.all(jnp.isfinite(jnp.real(point_wrappers.acb_bernoulli_poly_ui_point(w, 3))))))
    _check(bool(jnp.all(jnp.isfinite(jnp.real(point_wrappers.acb_polylog_si_point(u, 2))))))

    ex, invex = point_wrappers.acb_exp_invexp_point(z)
    _check(bool(jnp.isfinite(jnp.real(ex))))
    _check(bool(jnp.isfinite(jnp.real(invex))))
    s, c = point_wrappers.acb_sin_cos_pi_point(z)
    _check(bool(jnp.isfinite(jnp.real(s))))
    _check(bool(jnp.isfinite(jnp.real(c))))
