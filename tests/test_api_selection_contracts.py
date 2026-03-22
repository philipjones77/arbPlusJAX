from arbplusjax import api
from arbplusjax import capability_registry as cr


def test_matrix_metadata_exposes_execution_strategy_surface():
    direct = api.get_public_function_metadata("arb_mat_solve")
    cached = api.get_public_function_metadata("arb_mat_matvec_cached_apply")
    operator = api.get_public_function_metadata("jrb_mat_operator_plan_apply")

    assert direct.family == "matrix"
    assert "dense" in direct.execution_strategies
    assert "factorized" in direct.execution_strategies
    assert cached.family == "matrix"
    assert "dense" in cached.execution_strategies
    assert "cached" in cached.execution_strategies
    assert "matvec" in cached.execution_strategies
    assert operator.family == "matrix"
    assert "operator_plan" in operator.execution_strategies


def test_evaluate_rejects_unknown_matrix_strategy():
    try:
        api.evaluate("arb_mat_solve", 1.0, 1.0, strategy="not_a_strategy")
    except ValueError as exc:
        assert "unsupported strategy" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported strategy")


def test_capability_registry_exposes_selection_axes():
    registry = cr.build_capability_registry()
    row = registry["functions"]["besselk"]

    assert row["value_kinds"] == ["real", "complex", "real_interval", "complex_interval"]
    assert row["default_implementation"] == "besselk"
    assert "cuda_besselk" in row["implementation_options"]
    assert row["default_method"] is None or isinstance(row["default_method"], str)
    assert "execution_strategies" in row


def test_capability_registry_exposes_matrix_strategy_metadata():
    registry = cr.build_capability_registry()
    row = registry["functions"]["arb_mat_matvec_cached_apply"]

    assert "dense" in row["execution_strategies"]
    assert "cached" in row["execution_strategies"]
    assert "matvec" in row["execution_strategies"]
