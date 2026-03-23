import json
from dataclasses import asdict, fields

from arbplusjax import api
from arbplusjax import public_metadata as pm


def test_public_metadata_registry_matches_public_api_surface():
    entries = api.list_public_function_metadata()
    registry = api._PUBLIC_METADATA_REGISTRY()

    assert entries
    assert entries == sorted(entries, key=lambda entry: entry.name)
    assert {entry.name for entry in entries} == set(registry)
    assert api.get_public_function_metadata("incomplete_bessel_k") == registry["incomplete_bessel_k"]
    assert api.get_public_function_metadata("arb_mat_matvec_cached_apply") == registry["arb_mat_matvec_cached_apply"]


def test_public_metadata_entries_expose_complete_contract_shape():
    required_fields = {field.name for field in fields(pm.PublicFunctionMetadata)}

    for entry in api.list_public_function_metadata():
        payload = asdict(entry)
        assert set(payload) == required_fields
        assert entry.name
        assert entry.qualified_name.startswith(f"{entry.module}.")
        assert entry.value_kinds or (entry.point_support is False and entry.interval_support is False)
        assert entry.implementation_options
        assert entry.implementation_versions == ("current",)
        assert entry.default_implementation in entry.implementation_options
        assert entry.derivative_status
        if entry.interval_support:
            assert entry.interval_modes == ("point", "basic", "adaptive", "rigorous")
        else:
            assert entry.interval_modes in {(), ("point",)}


def test_public_metadata_family_specific_contracts_cover_matrix_and_special_surfaces():
    matrix_meta = api.get_public_function_metadata("arb_mat_matvec_cached_apply")
    gamma_meta = api.get_public_function_metadata("incomplete_gamma_upper")
    bessel_meta = api.get_public_function_metadata("incomplete_bessel_k")

    assert matrix_meta.family == "matrix"
    assert "cached" in matrix_meta.execution_strategies
    assert "matvec" in matrix_meta.execution_strategies
    assert "real_matrix" in matrix_meta.value_kinds

    assert gamma_meta.family == "gamma"
    assert gamma_meta.default_method == "quadrature"
    assert "samples_per_panel" in gamma_meta.method_parameter_names
    assert "transition_zone" in gamma_meta.regime_tags

    assert bessel_meta.family == "bessel"
    assert "high_precision_refine" in bessel_meta.method_tags
    assert "quadrature" in bessel_meta.method_tags
    assert bessel_meta.derivative_status == "explicit_custom_jvp_and_explicit_argument_lower"


def test_public_metadata_low_level_builder_handles_minimal_custom_registry():
    def direct_point(x, *, method: str = "direct"):
        return x

    registry = pm.build_public_metadata_registry(
        {"custom_surface": direct_point},
        point_names={"custom_surface"},
        interval_names=set(),
    )
    meta = registry["custom_surface"]

    assert meta.name == "custom_surface"
    assert meta.family == "core"
    assert meta.stability == "stable"
    assert meta.point_support is True
    assert meta.interval_support is False
    assert meta.interval_modes == ("point",)
    assert meta.default_method == "direct"
    assert meta.method_parameter_names == ()
    assert meta.execution_strategies == ("direct",)


def test_public_metadata_helper_only_rows_are_explicitly_non_callable_surfaces():
    meta = api.get_public_function_metadata("IFJBarnesDoubleGammaDiagnostics")

    assert meta.point_support is False
    assert meta.interval_support is False
    assert meta.value_kinds == ()
    assert meta.notes


def test_public_metadata_filtering_supports_report_facing_queries():
    matrix_rows = api.list_public_function_metadata(family="matrix", module="arb_mat")
    stable_gamma_rows = api.list_public_function_metadata(family="gamma", stability="stable")
    derivative_rows = api.list_public_function_metadata(derivative_status="explicit_custom_jvp_and_explicit_argument_lower")
    prefixed_rows = api.list_public_function_metadata(name_prefix="incomplete_bessel_")

    assert matrix_rows
    assert all(entry.family == "matrix" and entry.module == "arb_mat" for entry in matrix_rows)
    assert stable_gamma_rows
    assert all(entry.family == "gamma" and entry.stability == "stable" for entry in stable_gamma_rows)
    assert derivative_rows
    assert "incomplete_bessel_k" in {entry.name for entry in derivative_rows}
    assert all(entry.derivative_status == "explicit_custom_jvp_and_explicit_argument_lower" for entry in derivative_rows)
    assert {entry.name for entry in prefixed_rows} >= {"incomplete_bessel_i", "incomplete_bessel_k"}


def test_public_metadata_json_render_has_stable_top_level_shape():
    payload = json.loads(
        api.render_public_function_metadata_json(
            family="matrix",
            module="arb_mat",
            stability="stable",
        )
    )

    assert payload["generated_at"] == "2026-03-23T00:00:00Z"
    assert payload["source"] == "arbplusjax.public_metadata"
    assert payload["filters"] == {
        "family": "matrix",
        "stability": "stable",
        "module": "arb_mat",
        "name_prefix": None,
        "derivative_status": None,
    }
    assert payload["functions"]
    assert payload["functions"] == sorted(payload["functions"], key=lambda row: row["name"])
    assert all(row["family"] == "matrix" for row in payload["functions"])
    assert all(row["module"] == "arb_mat" for row in payload["functions"])
