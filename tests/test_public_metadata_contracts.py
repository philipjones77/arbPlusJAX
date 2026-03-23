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
