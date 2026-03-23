import json

from arbplusjax import capability_registry as cr


def test_capability_registry_has_stable_top_level_contract():
    registry = cr.build_capability_registry()

    assert registry["generated_at"] == "2026-03-17T00:00:00Z"
    assert registry["source"] == "arbplusjax.capability_registry"
    assert registry["policy_refs"] == [
        "docs/standards/jax_surface_policy_standard.md",
        "docs/standards/engineering_standard.md",
    ]
    assert "functions" in registry
    assert "downstream_kernels" in registry
    assert registry["functions"]
    assert registry["downstream_kernels"]


def test_capability_registry_downstream_aliases_resolve_to_public_capabilities():
    registry = cr.build_capability_registry()
    downstream = registry["downstream_kernels"]
    functions = registry["functions"]

    for alias, row in downstream.items():
        public_name = row["public_name"]
        capability = row["capability"]
        assert alias == row["alias"]
        assert public_name in functions
        assert capability["name"] == public_name
        assert capability["downstream_supported"] is True
        assert alias in capability["downstream_aliases"]
        if "complex_public_name" in row:
            complex_name = row["complex_public_name"]
            assert complex_name in functions
            assert row["complex_capability"]["name"] == complex_name


def test_capability_lookup_supports_function_and_alias_queries():
    function_row = cr.lookup_capability("arb_mat_matvec_cached_apply")
    alias_row = cr.lookup_capability("loggamma")
    barnes_row = cr.lookup_capability("barnesdoublegamma")

    assert function_row["name"] == "arb_mat_matvec_cached_apply"
    assert function_row["family"] == "matrix"
    assert "cached" in function_row["execution_strategies"]

    assert alias_row["alias"] == "loggamma"
    assert alias_row["public_name"] == "arb_lgamma"
    assert alias_row["complex_public_name"] == "acb_lgamma"
    assert alias_row["capability"]["family"] == "gamma"
    assert alias_row["complex_capability"]["family"] == "gamma"

    assert barnes_row["alias"] == "barnesdoublegamma"
    assert barnes_row["public_name"] == "ifj_barnesdoublegamma"
    assert barnes_row["capability"]["family"] == "barnes"
    assert barnes_row["capability"]["stability"] == "stable"


def test_capability_lookup_prefers_downstream_alias_rows_when_names_overlap():
    row = cr.lookup_capability("incomplete_bessel_k")

    assert row["alias"] == "incomplete_bessel_k"
    assert row["public_name"] == "incomplete_bessel_k"
    assert row["capability"]["name"] == "incomplete_bessel_k"


def test_capability_registry_includes_barnes_provider_grade_aliases():
    registry = cr.build_capability_registry()
    downstream = registry["downstream_kernels"]

    assert downstream["barnesdoublegamma"]["public_name"] == "ifj_barnesdoublegamma"
    assert downstream["log_barnesdoublegamma"]["public_name"] == "ifj_log_barnesdoublegamma"
    assert downstream["barnesdoublegamma"]["capability"]["derivative_status"]
    assert "barnesdoublegamma" in downstream["barnesdoublegamma"]["capability"]["downstream_aliases"]


def test_capability_registry_includes_fragile_regime_promotion_aliases():
    registry = cr.build_capability_registry()
    downstream = registry["downstream_kernels"]

    assert downstream["fragile_regime_promotion_gamma_upper"]["public_name"] == "incomplete_gamma_upper"
    assert downstream["fragile_regime_promotion_bessel_k"]["public_name"] == "incomplete_bessel_k"
    assert downstream["fragile_regime_promotion_bessel_i"]["public_name"] == "incomplete_bessel_i"
    assert "high_precision_refine" in downstream["fragile_regime_promotion_gamma_upper"]["capability"]["method_tags"]
    assert "high_precision_refine" in downstream["fragile_regime_promotion_bessel_k"]["capability"]["method_tags"]
    assert "high_precision_refine" in downstream["fragile_regime_promotion_bessel_i"]["capability"]["method_tags"]


def test_capability_registry_json_render_round_trips_to_current_schema():
    payload = json.loads(cr.render_capability_registry_json())

    assert payload["generated_at"] == "2026-03-17T00:00:00Z"
    assert sorted(payload["downstream_kernels"]) == sorted(cr.DOWNSTREAM_KERNELS)
    assert payload["functions"]["incomplete_gamma_upper"]["default_method"] == "quadrature"
    assert payload["functions"]["arb_mat_matvec_cached_apply"]["family"] == "matrix"
