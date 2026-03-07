from pathlib import Path

from arbplusjax import (
    acb_calc,
    acb_dirichlet,
    acb_elliptic,
    acb_modular,
    arb_calc,
    barnesg,
    boost_hypgeom,
    cubesselk,
    cusf_compat,
    double_gamma,
    function_provenance as fpr,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_function_provenance_reports_are_current():
    expected = {
        REPO_ROOT / "docs" / "function_naming.md": fpr.render_policy(),
        REPO_ROOT / "docs" / "reports" / "function_provenance_registry.md": fpr.render_registry_summary(),
        REPO_ROOT / "docs" / "reports" / "function_implementation_index.md": fpr.render_implementation_index(),
        REPO_ROOT / "docs" / "reports" / "arb_like_functions.md": fpr.render_report("arb_like", "Arb-like Functions"),
        REPO_ROOT / "docs" / "reports" / "alternative_functions.md": fpr.render_report("alternative", "Alternative Functions"),
        REPO_ROOT / "docs" / "reports" / "new_functions.md": fpr.render_report("new", "New Functions"),
    }
    for path, content in expected.items():
        assert path.read_text(encoding="utf-8") == content, f"Generated report out of date: {path}"


def test_besselk_lookup_lists_all_registered_implementations():
    text = fpr.render_lookup("besselk")
    assert "besselk" in text
    assert "cusf_besselk" in text
    assert "point|basic|adaptive|rigorous" in text
    assert "canonical interval/precision path" in text or "implementation-specific mode-aware tightening" in text


def test_barnesgamma2_lookup_uses_bdg_prefixed_name():
    text = fpr.render_lookup("barnesgamma2")
    assert "bdg_barnesgamma2" in text
    assert "BarnesDoubleGamma.jl" in text or "Julia/" in text


def test_every_public_inventory_symbol_is_classified():
    inventory = set(fpr.public_functions())
    entries = {row.public_name for row in fpr.build_entries()}
    assert inventory == entries


def test_implementation_index_has_broad_coverage():
    rows = fpr.build_implementation_entries()
    assert len(rows) > 500


def test_known_alternative_overlays_are_present():
    rows = {(row.base_name, row.preferred_public_name) for row in fpr.build_implementation_entries()}
    assert ("besselk", "cusf_besselk") in rows
    assert ("besselk", "cuda_besselk") in rows
    assert ("gamma", "cusf_gamma") in rows
    assert ("hypergeometric_1f1", "boost_hypergeometric_1f1") in rows
    assert ("barnesgamma2", "bdg_barnesgamma2") in rows


def test_noncanonical_modules_expose_provenance_metadata():
    for module in (
        cubesselk,
        cusf_compat,
        boost_hypgeom,
        double_gamma,
        acb_modular,
        acb_dirichlet,
        acb_elliptic,
        arb_calc,
        acb_calc,
        barnesg,
    ):
        assert hasattr(module, "PROVENANCE")
        data = module.PROVENANCE
        assert "classification" in data
        assert "naming_policy" in data
        assert "registry_report" in data
