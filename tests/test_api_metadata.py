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
    with pytest.raises(KeyError):
        api.get_public_function_metadata("incomplete_bessel_i")
