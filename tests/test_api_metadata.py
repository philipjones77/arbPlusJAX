import pytest

from arbplusjax import api


def test_besselk_metadata_marks_stable_bessel_surface():
    metadata = api.get_public_function_metadata("besselk")

    assert metadata.family == "bessel"
    assert metadata.stability == "stable"
    assert metadata.point_support is True
    assert metadata.interval_support is True
    assert metadata.interval_modes == ("point", "basic", "adaptive", "rigorous")
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
