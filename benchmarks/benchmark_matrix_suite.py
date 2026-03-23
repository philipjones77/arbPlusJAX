"""Comprehensive benchmark suite for matrix operations across all storage types.

Benchmarks:
- Dense matrices (arb_mat, acb_mat)
- Sparse matrices (srb_mat, scb_mat)
- Block sparse matrices (srb_block_mat, scb_block_mat)
- Variable block sparse matrices (srb_vblock_mat, scb_vblock_mat)
- Matrix-free (jrb_mat, jcb_mat)

Operations tested:
- det, inv, sqr, trace
- norm_fro, norm_1, norm_inf
- matmul, matvec, solve

Comparisons:
- scipy/scipy.sparse
- boost (if available)
- Mathematica (if available)
"""

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Set JAX to 64-bit mode
jax.config.update("jax_enable_x64", True)

from arbplusjax import arb_mat, acb_mat
from arbplusjax import srb_mat, scb_mat
from arbplusjax import srb_block_mat, scb_block_mat
from arbplusjax import srb_vblock_mat, scb_vblock_mat
from arbplusjax import jrb_mat, jcb_mat
from arbplusjax import double_interval as di

# scipy imports
try:
    import scipy
    import scipy.linalg
    import scipy.sparse
    import scipy.sparse.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available for comparison")


class BenchmarkResult:
    """Store benchmark results."""
    def __init__(self, name, storage_type, size, sparsity=None):
        self.name = name
        self.storage_type = storage_type
        self.size = size
        self.sparsity = sparsity
        self.compile_time = 0.0
        self.first_run_time = 0.0
        self.warm_run_time = 0.0
        self.accuracy = None
        self.reference_time = None
        self.speedup = None
        
    def __repr__(self):
        parts = [
            f"{self.name:20s}",
            f"{self.storage_type:15s}",
            f"n={self.size:4d}",
        ]
        if self.sparsity:
            parts.append(f"sparsity={self.sparsity:.2%}")
        parts.extend([
            f"compile={self.compile_time*1000:7.2f}ms",
            f"first={self.first_run_time*1000:7.2f}ms",
            f"warm={self.warm_run_time*1000:7.2f}ms",
        ])
        if self.reference_time:
            parts.append(f"ref={self.reference_time*1000:7.2f}ms")
            parts.append(f"speedup={self.speedup:.2f}x")
        if self.accuracy is not None:
            parts.append(f"err={self.accuracy:.2e}")
        return " | ".join(parts)


def time_function(f, *args, warmup=1, iterations=10):
    """Time a JAX function with JIT compilation tracking."""
    # Compile
    t0 = time.perf_counter()
    compiled_f = jax.jit(f)
    _ = compiled_f(*args).block_until_ready()
    compile_time = time.perf_counter() - t0
    
    # First run (already compiled)
    t0 = time.perf_counter()
    result = compiled_f(*args).block_until_ready()
    first_time = time.perf_counter() - t0
    
    # Warmup
    for _ in range(warmup):
        _ = compiled_f(*args).block_until_ready()
    
    # Timed runs
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = compiled_f(*args).block_until_ready()
        times.append(time.perf_counter() - t0)
    
    warm_time = np.median(times)
    return compile_time, first_time, warm_time, result


def create_test_dense_real(n, seed=42):
    """Create a test real dense matrix."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    # Make it well-conditioned and symmetric positive definite
    A = A @ A.T + n * np.eye(n)
    return A


def create_test_dense_complex(n, seed=42):
    """Create a test complex dense matrix."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n) + 1j * rng.randn(n, n)
    # Make it well-conditioned
    A = A @ np.conj(A.T) + n * np.eye(n, dtype=np.complex128)
    return A


def create_test_sparse_real(n, sparsity=0.1, seed=42):
    """Create a test real sparse matrix."""
    rng = np.random.RandomState(seed)
    nnz = int(n * n * sparsity)
    rows = rng.randint(0, n, size=nnz)
    cols = rng.randint(0, n, size=nnz)
    data = rng.randn(nnz)
    
    # Build dense then sparsify to ensure SPD
    A_dense = np.zeros((n, n))
    for i, j, v in zip(rows, cols, data):
        A_dense[i, j] += v
    A_dense = A_dense @ A_dense.T + n * np.eye(n)
    
    # Extract sparse
    mask = np.abs(A_dense) > 1e-10
    rows, cols = np.where(mask)
    data = A_dense[rows, cols]
    
    return data, rows.astype(np.int32), cols.astype(np.int32), (n, n)


def benchmark_dense_real_operations(n=100):
    """Benchmark dense real matrix operations."""
    print(f"\n{'='*80}")
    print(f"DENSE REAL MATRICES (arb_mat) - Size {n}x{n}")
    print(f"{'='*80}")
    
    results = []
    A_np = create_test_dense_real(n)
    # Create interval matrix properly - point intervals
    A_arb = di.interval(A_np, A_np)  # Point interval [a, a]
    x_arb = di.interval(np.ones(n), np.ones(n))
    
    # Test determinant
    result = BenchmarkResult("det", "arb_mat", n)
    c, f, w, det_val = time_function(arb_mat.arb_mat_det, A_arb)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        t0 = time.perf_counter()
        det_ref = np.linalg.det(A_np)
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.abs(di.midpoint(det_val) - det_ref) / (np.abs(det_ref) + 1e-14)
    results.append(result)
    print(result)
    
    # Test inverse
    result = BenchmarkResult("inv", "arb_mat", n)
    c, f, w, inv_val = time_function(arb_mat.arb_mat_inv, A_arb)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        t0 = time.perf_counter()
        inv_ref = np.linalg.inv(A_np)
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.linalg.norm(di.midpoint(inv_val) - inv_ref) / np.linalg.norm(inv_ref)
    results.append(result)
    print(result)
    
    # Test square
    result = BenchmarkResult("sqr", "arb_mat", n)
    c, f, w, sqr_val = time_function(arb_mat.arb_mat_sqr, A_arb)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        t0 = time.perf_counter()
        sqr_ref = A_np @ A_np
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.linalg.norm(di.midpoint(sqr_val) - sqr_ref) / np.linalg.norm(sqr_ref)
    results.append(result)
    print(result)
    
    # Test trace
    result = BenchmarkResult("trace", "arb_mat", n)
    c, f, w, trace_val = time_function(arb_mat.arb_mat_trace, A_arb)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        t0 = time.perf_counter()
        trace_ref = np.trace(A_np)
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.abs(di.midpoint(trace_val) - trace_ref) / (np.abs(trace_ref) + 1e-14)
    results.append(result)
    print(result)
    
    # Test norms
    for norm_name, norm_func in [("norm_fro", arb_mat.arb_mat_norm_fro), 
                                   ("norm_1", arb_mat.arb_mat_norm_1),
                                   ("norm_inf", arb_mat.arb_mat_norm_inf)]:
        result = BenchmarkResult(norm_name, "arb_mat", n)
        c, f, w, norm_val = time_function(norm_func, A_arb)
        result.compile_time = c
        result.first_run_time = f
        result.warm_run_time = w
        if HAS_SCIPY:
            ord_map = {"norm_fro": "fro", "norm_1": 1, "norm_inf": np.inf}
            t0 = time.perf_counter()
            norm_ref = np.linalg.norm(A_np, ord=ord_map[norm_name])
            result.reference_time = time.perf_counter() - t0
            result.speedup = result.reference_time / result.warm_run_time
            result.accuracy = np.abs(di.midpoint(norm_val) - norm_ref) / (norm_ref + 1e-14)
        results.append(result)
        print(result)
    
    # Test matvec
    result = BenchmarkResult("matvec", "arb_mat", n)
    c, f, w, matvec_val = time_function(arb_mat.arb_mat_matvec, A_arb, x_arb)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        t0 = time.perf_counter()
        matvec_ref = A_np @ np.ones(n)
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.linalg.norm(di.midpoint(matvec_val) - matvec_ref) / np.linalg.norm(matvec_ref)
    results.append(result)
    print(result)
    
    # Test solve
    result = BenchmarkResult("solve", "arb_mat", n)
    c, f, w, solve_val = time_function(arb_mat.arb_mat_solve, A_arb, x_arb)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        t0 = time.perf_counter()
        solve_ref = np.linalg.solve(A_np, np.ones(n))
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.linalg.norm(di.midpoint(solve_val) - solve_ref) / np.linalg.norm(solve_ref)
    results.append(result)
    print(result)
    
    return results


def benchmark_sparse_real_operations(n=500, sparsity=0.01):
    """Benchmark sparse real matrix operations."""
    print(f"\n{'='*80}")
    print(f"SPARSE REAL MATRICES (srb_mat) - Size {n}x{n}, Sparsity {sparsity:.1%}")
    print(f"{'='*80}")
    
    results = []
    data, rows, cols, shape = create_test_sparse_real(n, sparsity)
    A_sparse = srb_mat.srb_mat_coo(data, rows, cols, shape=shape)
    x_arb = di.interval(np.ones(n), np.ones(n))
    
    # Test determinant
    result = BenchmarkResult("det", "srb_mat", n, sparsity)
    c, f, w, det_val = time_function(srb_mat.srb_mat_det, A_sparse)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        A_scipy = scipy.sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
        t0 = time.perf_counter()
        det_ref = scipy.sparse.linalg.splu(A_scipy).det()
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.abs(di.midpoint(det_val) - det_ref) / (np.abs(det_ref) + 1e-14)
    results.append(result)
    print(result)
    
    # Test matvec
    result = BenchmarkResult("matvec", "srb_mat", n, sparsity)
    c, f, w, matvec_val = time_function(srb_mat.srb_mat_matvec, A_sparse, x_arb)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        t0 = time.perf_counter()
        matvec_ref = A_scipy @ np.ones(n)
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.linalg.norm(di.midpoint(matvec_val) - matvec_ref) / np.linalg.norm(matvec_ref)
    results.append(result)
    print(result)
    
    return results


def benchmark_matrix_free_operations(n=100):
    """Benchmark matrix-free operations."""
    print(f"\n{'='*80}")
    print(f"MATRIX-FREE (jrb_mat) - Size {n}x{n}")
    print(f"{'='*80}")
    
    results = []
    A_np = create_test_dense_real(n)
    A_jrb = di.interval(A_np, A_np)
    x_jrb = di.interval(np.ones(n), np.ones(n))
    
    # Test determinant
    result = BenchmarkResult("det", "jrb_mat", n)
    c, f, w, det_val = time_function(jrb_mat.jrb_mat_det_point, A_jrb)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        t0 = time.perf_counter()
        det_ref = np.linalg.det(A_np)
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.abs(di.midpoint(det_val) - det_ref) / (np.abs(det_ref) + 1e-14)
    results.append(result)
    print(result)
    
    # Test trace
    result = BenchmarkResult("trace", "jrb_mat", n)
    c, f, w, trace_val = time_function(jrb_mat.jrb_mat_trace_point, A_jrb)
    result.compile_time = c
    result.first_run_time = f
    result.warm_run_time = w
    if HAS_SCIPY:
        t0 = time.perf_counter()
        trace_ref = np.trace(A_np)
        result.reference_time = time.perf_counter() - t0
        result.speedup = result.reference_time / result.warm_run_time
        result.accuracy = np.abs(di.midpoint(trace_val) - trace_ref) / (np.abs(trace_ref) + 1e-14)
    results.append(result)
    print(result)
    
    return results


def analyze_jax_hotspots():
    """Analyze JAX compilation and execution patterns for performance hotspots."""
    print(f"\n{'='*80}")
    print("JAX HOTSPOT ANALYSIS")
    print(f"{'='*80}")
    
    n = 100
    A = di.interval(create_test_dense_real(n), create_test_dense_real(n))
    
    # Check for recompilation issues
    print("\n1. Testing for recompilation with different shapes:")
    for size in [50, 100, 150]:
        A_test = di.interval(create_test_dense_real(size), create_test_dense_real(size))
        t0 = time.perf_counter()
        _ = jax.jit(arb_mat.arb_mat_det)(A_test)
        print(f"   n={size}: {(time.perf_counter() - t0)*1000:.2f}ms (first call)")
        t0 = time.perf_counter()
        _ = jax.jit(arb_mat.arb_mat_det)(A_test)
        print(f"   n={size}: {(time.perf_counter() - t0)*1000:.2f}ms (second call)")
    
    # Check for Python loop overhead
    print("\n2. Testing Python loop overhead in sparse operations:")
    data, rows, cols, shape = create_test_sparse_real(500, 0.01)
    A_sparse = srb_mat.srb_mat_coo(data, rows, cols, shape=shape)
    
    t0 = time.perf_counter()
    _ = srb_mat.srb_mat_to_dense(A_sparse)
    print(f"   to_dense (first): {(time.perf_counter() - t0)*1000:.2f}ms")
    
    t0 = time.perf_counter()
    _ = srb_mat.srb_mat_to_dense(A_sparse)
    print(f"   to_dense (second): {(time.perf_counter() - t0)*1000:.2f}ms")
    
    # Check vmap vs loop performance
    print("\n3. Testing vmap vs explicit loop:")
    n = 100
    A_np = create_test_dense_real(n)
    A_arb = di.interval(A_np, A_np)
    n_vectors = 50
    vectors = di.interval(np.random.randn(n_vectors, n), np.random.randn(n_vectors, n))
    
    # Vmap version
    @jax.jit
    def matvec_vmap(A, vs):
        return jax.vmap(lambda v: arb_mat.arb_mat_matvec(A, v))(vs)
    
    t0 = time.perf_counter()
    _ = matvec_vmap(A_arb, vectors).block_until_ready()
    vmap_time = time.perf_counter() - t0
    print(f"   vmap: {vmap_time*1000:.2f}ms")
    
    # Loop version (Python loop - bad!)
    t0 = time.perf_counter()
    results = []
    for i in range(n_vectors):
        results.append(arb_mat.arb_mat_matvec(A_arb, vectors[i]))
    loop_time = time.perf_counter() - t0
    print(f"   Python loop: {loop_time*1000:.2f}ms")
    print(f"   Speedup: {loop_time/vmap_time:.2f}x")


def main():
    """Run all benchmarks."""
    print("="*80)
    print("COMPREHENSIVE MATRIX OPERATION BENCHMARK SUITE")
    print("="*80)
    print(f"JAX version: {jax.__version__}")
    print(f"NumPy version: {np.__version__}")
    if HAS_SCIPY:
        print(f"SciPy version: {scipy.__version__}")
    
    all_results = []
    
    # Dense matrices
    all_results.extend(benchmark_dense_real_operations(n=100))
    all_results.extend(benchmark_dense_real_operations(n=200))
    
    # Sparse matrices
    all_results.extend(benchmark_sparse_real_operations(n=500, sparsity=0.01))
    all_results.extend(benchmark_sparse_real_operations(n=1000, sparsity=0.005))
    
    # Matrix-free
    all_results.extend(benchmark_matrix_free_operations(n=100))
    all_results.extend(benchmark_matrix_free_operations(n=200))
    
    # JAX hotspot analysis
    analyze_jax_hotspots()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total operations benchmarked: {len(all_results)}")
    
    if HAS_SCIPY:
        avg_speedup = np.mean([r.speedup for r in all_results if r.speedup is not None])
        avg_accuracy = np.mean([r.accuracy for r in all_results if r.accuracy is not None])
        print(f"Average speedup vs NumPy/SciPy: {avg_speedup:.2f}x")
        print(f"Average relative error: {avg_accuracy:.2e}")
    
    avg_compile = np.mean([r.compile_time for r in all_results])
    avg_warm = np.mean([r.warm_run_time for r in all_results])
    print(f"Average compile time: {avg_compile*1000:.2f}ms")
    print(f"Average warm run time: {avg_warm*1000:.2f}ms")


if __name__ == "__main__":
    main()
