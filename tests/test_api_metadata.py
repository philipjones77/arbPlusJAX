import pytest

from arbplusjax import api


def test_besselk_metadata_marks_stable_bessel_surface():
    metadata = api.get_public_function_metadata("besselk")

    assert metadata.family == "bessel"
    assert metadata.stability == "stable"
    assert metadata.point_support is True
    assert metadata.interval_support is True
    assert metadata.interval_modes == ("point", "basic", "adaptive", "rigorous")
    assert metadata.value_kinds == ("real", "complex", "real_interval", "complex_interval")
    assert "besselk" in metadata.implementation_options
    assert "cuda_besselk" in metadata.implementation_options
    assert metadata.default_implementation == "besselk"
    assert "asymptotic" in metadata.method_tags


def test_cuda_besselk_metadata_marks_experimental_surface():
    metadata = api.get_public_function_metadata("cuda_besselk")

    assert metadata.family == "bessel"
    assert metadata.stability == "experimental"
    assert metadata.point_support is True
    assert metadata.interval_support is True


def test_list_public_function_metadata_filters_family_and_stability():
    entries = api.list_public_function_metadata(family="bessel", stability="stable")
    names = {entry.name for entry in entries}

    assert "besselk" in names
    assert "cuda_besselk" not in names


def test_unknown_public_function_metadata_raises_key_error():
    with pytest.raises(KeyError):
        api.get_public_function_metadata("not_a_real_function")


def test_incomplete_bessel_k_metadata_marks_experimental_surface():
    metadata = api.get_public_function_metadata("incomplete_bessel_k")

    assert metadata.family == "bessel"
    assert metadata.stability == "experimental"
    assert metadata.point_support is True
    assert metadata.interval_support is True


def test_unknown_planned_incomplete_bessel_i_still_raises_key_error():
    metadata = api.get_public_function_metadata("incomplete_bessel_i")

    assert metadata.family == "bessel"
    assert metadata.stability == "experimental"
    assert metadata.point_support is True
    assert metadata.interval_support is True


def test_incomplete_bessel_metadata_exposes_methods_and_derivative_contract():
    k_meta = api.get_public_function_metadata("incomplete_bessel_k")
    i_meta = api.get_public_function_metadata("incomplete_bessel_i")

    assert "quadrature" in k_meta.method_tags
    assert "recurrence" in k_meta.method_tags
    assert "high_precision_refine" in k_meta.method_tags
    assert k_meta.derivative_status == "explicit_custom_jvp_and_explicit_argument_lower"
    assert "quadrature" in i_meta.method_tags
    assert "high_precision_refine" in i_meta.method_tags
    assert "near_full_interval" in i_meta.regime_tags
    assert i_meta.derivative_status == "explicit_custom_jvp_and_explicit_argument_upper"


def test_tail_integral_accelerated_metadata_exposes_method_selection_surface():
    metadata = api.get_public_function_metadata("tail_integral_accelerated")

    assert metadata.family == "integration"
    assert "aitken" in metadata.method_tags
    assert "wynn" in metadata.method_tags
    assert "high_precision_refine" in metadata.method_tags


def test_incomplete_gamma_metadata_marks_experimental_tail_specialization():
    upper = api.get_public_function_metadata("incomplete_gamma_upper")
    lower = api.get_public_function_metadata("incomplete_gamma_lower")

    assert upper.family == "gamma"
    assert upper.stability == "experimental"
    assert "tail_specialization" in upper.method_tags
    assert "custom_jvp" in upper.method_tags
    assert "transition_zone" in upper.regime_tags
    assert upper.derivative_status == "explicit_custom_jvp_and_explicit_argument_parameter"
    assert lower.family == "gamma"
    assert lower.stability == "experimental"
    assert "complement" in lower.method_tags


def test_laplace_bessel_tail_metadata_marks_experimental_tail_specialization():
    metadata = api.get_public_function_metadata("laplace_bessel_k_tail")

    assert metadata.family == "bessel"
    assert metadata.stability == "experimental"
    assert "tail_specialization" in metadata.method_tags
    assert "slow_combined_decay" in metadata.regime_tags
    assert metadata.derivative_status == "explicit_custom_jvp_and_explicit_lambda_lower"


def test_incomplete_gamma_metadata_exposes_method_parameters_and_execution_strategy():
    metadata = api.get_public_function_metadata("incomplete_gamma_upper")

    assert metadata.default_method == "quadrature"
    assert "regularized" in metadata.method_parameter_names
    assert "panel_width" in metadata.method_parameter_names
    assert metadata.execution_strategies == ("direct",)


def test_matrix_and_sparse_metadata_expose_execution_strategies_for_production_routing():
    dense = api.get_public_function_metadata("arb_mat_matvec_cached_apply")
    sparse = api.get_public_function_metadata("srb_mat_matvec")
    operator = api.get_public_function_metadata("jrb_mat_operator_plan_apply")

    assert "dense" in dense.execution_strategies
    assert "cached" in dense.execution_strategies
    assert "matvec" in dense.execution_strategies
    assert "sparse" in sparse.execution_strategies
    assert "matvec" in sparse.execution_strategies
    assert "operator_plan" in operator.execution_strategies


def test_special_metadata_exposes_parameterized_production_surface():
    bessel = api.get_public_function_metadata("incomplete_bessel_k")
    gamma = api.get_public_function_metadata("incomplete_gamma_upper")

    assert bessel.default_method == "quadrature"
    assert "auto" in bessel.method_tags
    assert "samples_per_panel" in gamma.method_parameter_names
    assert "max_panels" in gamma.method_parameter_names
    assert "regularized" in gamma.method_parameter_names
    assert "high_precision_refine" in bessel.method_tags


def test_evaluate_routes_to_alternative_implementation_by_name():
    x = 0.5
    y = 2.0

    routed = api.evaluate("besselk", x, y, implementation="cuda_besselk", value_kind="real")
    direct = api.eval_point("cuda_besselk", x, y)

    assert routed == direct


def test_evaluate_routes_method_and_method_params_to_direct_signature():
    s = 2.5
    z = 1.0

    routed = api.evaluate(
        "incomplete_gamma_upper",
        s,
        z,
        method="quadrature",
        method_params={"samples_per_panel": 8, "max_panels": 16},
        return_diagnostics=False,
    )
    direct = api.incomplete_gamma_upper(
        s,
        z,
        method="quadrature",
        samples_per_panel=8,
        max_panels=16,
        return_diagnostics=False,
    )

    assert routed == direct


def test_evaluate_rejects_interval_value_kind_with_point_mode():
    with pytest.raises(ValueError):
        api.evaluate("besselk", 0.5, 2.0, value_kind="real_interval")
