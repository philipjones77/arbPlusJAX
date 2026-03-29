import jax.numpy as jnp

from arbplusjax import api


def test_matrix_free_policy_helper_prefers_gpu_only_for_large_or_sparse_complex(monkeypatch) -> None:
    monkeypatch.setattr(api, "_available_backends", lambda: ("cpu", "gpu"))

    dense_small = api.choose_matrix_free_plan_policy(
        algebra="jrb",
        plan_kind="dense",
        problem_size=64,
        steps=8,
        probe_count=2,
        backend="auto",
        min_gpu_size_dense=256,
    )
    dense_large = api.choose_matrix_free_plan_policy(
        algebra="jrb",
        plan_kind="dense",
        problem_size=512,
        steps=8,
        probe_count=2,
        backend="auto",
        min_gpu_size_dense=256,
    )
    sparse_complex = api.choose_matrix_free_plan_policy(
        algebra="jcb",
        plan_kind="sparse_bcoo",
        problem_size=64,
        steps=8,
        probe_count=2,
        backend="auto",
        min_gpu_size_sparse_complex=32,
    )

    assert dense_small.chosen_backend == "cpu"
    assert dense_large.chosen_backend == "gpu"
    assert sparse_complex.chosen_backend == "gpu"


def test_matrix_free_prewarm_entrypoint_returns_backend_diagnostics() -> None:
    diagnostics = api.prewarm_matrix_free_kernels(
        cases=(
            ("jrb", "dense", "apply"),
            ("jrb", "dense", "multi_shift"),
            ("jcb", "dense", "adjoint_apply"),
            ("jcb", "sparse_bcoo", "apply"),
        ),
        backend="cpu",
        dense_problem_size=4,
        sparse_problem_size=4,
        steps=2,
        probe_count=2,
    )

    assert set(diagnostics) == {
        "jrb:dense:apply",
        "jrb:dense:multi_shift",
        "jcb:dense:adjoint_apply",
        "jcb:sparse_bcoo:apply",
    }
    assert diagnostics["jrb:dense:apply"].chosen_backend == "cpu"
    assert diagnostics["jrb:dense:apply"].compiled_this_call is True
    assert diagnostics["jrb:dense:multi_shift"].operation == "multi_shift"
    assert diagnostics["jcb:dense:adjoint_apply"].plan_kind == "dense"
    assert diagnostics["jcb:sparse_bcoo:apply"].problem_size == 4


def test_matrix_free_policy_supports_explicit_backend_override() -> None:
    policy = api.choose_matrix_free_plan_policy(
        algebra="jcb",
        plan_kind="dense",
        problem_size=8,
        steps=4,
        probe_count=2,
        backend="cpu",
    )

    assert policy.requested_backend == "cpu"
    assert policy.chosen_backend == "cpu"
    assert policy.steps == 4
    assert policy.probe_count == 2
