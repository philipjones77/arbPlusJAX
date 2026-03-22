from arbplusjax import mat_wrappers
from arbplusjax import scb_mat
from arbplusjax import srb_mat


SPARSE_BASIC_GOVERNED = (
    "to_dense",
    "transpose",
    "symmetric_part",
    "is_symmetric",
    "is_spd",
    "trace",
    "norm_fro",
    "norm_1",
    "norm_inf",
    "matvec",
    "rmatvec",
    "matvec_cached_prepare",
    "matvec_cached_apply",
    "rmatvec_cached_prepare",
    "rmatvec_cached_apply",
    "matmul_dense_rhs",
    "triangular_solve",
    "cho",
    "ldl",
    "charpoly",
    "pow_ui",
    "exp",
    "eigvalsh",
    "eigh",
    "lu_solve_plan_prepare",
    "lu_solve_plan_apply",
    "spd_solve_plan_prepare",
    "spd_solve_plan_apply",
    "spd_solve",
    "spd_inv",
    "solve_lu",
    "solve_lu_precomp",
    "solve_transpose",
    "solve_add",
    "solve_transpose_add",
    "mat_solve",
    "mat_solve_transpose",
    "solve",
    "det",
    "inv",
    "sqr",
)


SCB_BASIC_GOVERNED = (
    "to_dense",
    "transpose",
    "conjugate_transpose",
    "hermitian_part",
    "is_hermitian",
    "is_hpd",
    "trace",
    "norm_fro",
    "norm_1",
    "norm_inf",
    "matvec",
    "rmatvec",
    "matvec_cached_prepare",
    "matvec_cached_apply",
    "rmatvec_cached_prepare",
    "rmatvec_cached_apply",
    "matmul_dense_rhs",
    "triangular_solve",
    "cho",
    "ldl",
    "charpoly",
    "pow_ui",
    "exp",
    "eigvalsh",
    "eigh",
    "lu_solve_plan_prepare",
    "lu_solve_plan_apply",
    "hpd_solve_plan_prepare",
    "hpd_solve_plan_apply",
    "hpd_solve",
    "hpd_inv",
    "solve_lu",
    "solve_lu_precomp",
    "solve_transpose",
    "solve_add",
    "solve_transpose_add",
    "mat_solve",
    "mat_solve_transpose",
    "solve",
    "det",
    "inv",
    "sqr",
)


SPARSE_POINT_ONLY = (
    "coo",
    "csr",
    "bcoo",
    "from_dense_coo",
    "from_dense_csr",
    "from_dense_bcoo",
    "coo_to_csr",
    "csr_to_coo",
    "coo_to_bcoo",
    "csr_to_bcoo",
    "bcoo_to_coo",
    "coo_to_dense",
    "csr_to_dense",
    "bcoo_to_dense",
)


def _has_mode_wrapper(name: str) -> bool:
    return hasattr(mat_wrappers, f"{name}_mode")


def test_srb_governed_sparse_surface_has_basic_coverage():
    for suffix in SPARSE_BASIC_GOVERNED:
        point_name = f"srb_mat_{suffix}"
        assert hasattr(srb_mat, point_name)
        assert hasattr(srb_mat, f"{point_name}_basic")
        assert _has_mode_wrapper(point_name)


def test_scb_governed_sparse_surface_has_basic_coverage():
    for suffix in SCB_BASIC_GOVERNED:
        point_name = f"scb_mat_{suffix}"
        assert hasattr(scb_mat, point_name)
        assert hasattr(scb_mat, f"{point_name}_basic")
        assert _has_mode_wrapper(point_name)


def test_sparse_point_only_constructors_and_converters_do_not_get_mode_wrappers():
    for prefix in ("srb_mat", "scb_mat"):
        for suffix in SPARSE_POINT_ONLY:
            point_name = f"{prefix}_{suffix}"
            assert hasattr(srb_mat if prefix == "srb_mat" else scb_mat, point_name)
            assert not _has_mode_wrapper(point_name)
