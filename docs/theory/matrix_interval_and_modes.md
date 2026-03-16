Last updated: 2026-03-16T18:24:00Z

# Matrix Interval Arithmetic and Matrix Modes

## Purpose

This note extends the scalar interval discussion to matrices. The goal is to make the project’s matrix semantics explicit:

- what an interval matrix or complex box matrix means,
- how basic matrix operations are enclosed,
- where arbPlusJAX uses true entrywise interval/box arithmetic,
- where it still falls back to midpoint linear algebra with outward boxing,
- how this interacts with the four-mode model.

## 1. Matrix enclosure model

### 1.1 Real interval matrices

A real interval matrix is a matrix

\[
A = [A^-, A^+],
\]

meaning that each entry satisfies

\[
a_{ij} \in [a^-_{ij}, a^+_{ij}].
\]

Equivalently, one may write

\[
A = M \pm R,
\]

where

\[
M = \frac{A^- + A^+}{2}, \qquad R = \frac{A^+ - A^-}{2},
\]

with midpoint matrix \(M\) and nonnegative radius matrix \(R\).

In arbPlusJAX, a real interval matrix is stored entrywise with shape `(..., n, n, 2)`.

### 1.2 Complex box matrices

A complex box matrix is represented entrywise by rectangular enclosures in the complex plane:

\[
a_{ij} \in [\Re^-_{ij}, \Re^+_{ij}] + i[\Im^-_{ij}, \Im^+_{ij}].
\]

This is not a circular complex ball per entry; it is a Cartesian product box for the real and imaginary parts.

In arbPlusJAX, a complex box matrix is stored with shape `(..., n, n, 4)`.

## 2. Entrywise interval matrix arithmetic

For matrices \(A\) and \(B\), entrywise interval addition is direct:

\[
[A^- , A^+] + [B^- , B^+] = [A^- + B^- , A^+ + B^+].
\]

Matrix multiplication is defined through enclosure of each entry:

\[
(AB)_{ij} = \sum_k a_{ik} b_{kj}.
\]

In interval arithmetic, this becomes

\[
(AB)_{ij} \subseteq \sum_k [a_{ik}] [b_{kj}],
\]

where interval products and interval sums are carried out with outward rounding.

That is the mathematically correct enclosure pattern for direct interval matrix multiplication, and it is also the pattern used in the project’s `*_matmul_basic` and `*_matvec_basic` kernels.

## 3. What arbPlusJAX does today

The matrix implementation is intentionally mixed between:

- direct interval/box algebra where it is cheap and structurally clear,
- midpoint linear algebra where full enclosure algorithms are not yet implemented.

### 3.1 Direct interval / box paths

For real interval matrices:

- `arb_mat_matmul_basic`
- `arb_mat_matvec_basic`
- `arb_mat_banded_matvec_basic`
- `arb_mat_trace_basic`
- `arb_mat_det_basic` for `1x1` to `3x3`
- `arb_mat_norm_fro_basic`

For complex box matrices:

- `acb_mat_matmul_basic`
- `acb_mat_matvec_basic`
- `acb_mat_banded_matvec_basic`
- `acb_mat_trace_basic`
- `acb_mat_det_basic` for `1x1` to `3x3`
- `acb_mat_norm_fro_basic`

These paths compute with interval or box primitives directly instead of collapsing everything to midpoint linear algebra first.

### 3.2 Midpoint-first paths

The following families are still midpoint-first in the current codebase:

- `solve`
- `inv`
- `triangular_solve`
- `lu`
- `qr`

The pattern is:

1. form the midpoint matrix \(M\),
2. solve or factorize using ordinary floating-point linear algebra,
3. wrap the point result back into an interval or box with outward rounding.

Schematically,

\[
X \approx f(M),
\]

then

\[
\operatorname{out} = \operatorname{round\_outward}(X).
\]

This gives a box-shaped result, but it is not yet a full verified matrix enclosure algorithm in the sense of interval linear algebra.

## 4. Determinant and trace formulas

The determinant and trace make the distinction between direct enclosure and midpoint fallback especially clear.

### 4.1 Trace

For any square matrix,

\[
\operatorname{tr}(A) = \sum_{i=1}^n a_{ii}.
\]

So for an interval matrix, a direct enclosure is obtained by summing the diagonal intervals entrywise. This is why `trace_basic` is naturally rigorous at the entrywise level.

### 4.2 `2x2` determinant

For

\[
A = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix},
\]

the determinant is

\[
\det(A) = ad - bc.
\]

If \(a,b,c,d\) are intervals or boxes, this formula can be evaluated directly with interval or box multiplication and subtraction, giving an enclosure for the determinant.

### 4.3 `3x3` determinant

For

\[
A =
\begin{pmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{pmatrix},
\]

the determinant is

\[
\det(A) = a(ei - fh) - b(di - fg) + c(dh - eg).
\]

Again, this can be evaluated in interval or box arithmetic directly, which is why the code keeps exact enclosure formulas through `3x3`.

### 4.4 Larger determinants

For larger \(n\), arbPlusJAX currently falls back to midpoint determinant computation followed by outward boxing:

\[
\det(A) \approx \det(M), \qquad \operatorname{out} = \operatorname{round\_outward}(\det(M)).
\]

This is computationally simple but is weaker than a full verified determinant algorithm.

## 5. Matrix norms

The project currently supports Frobenius, `1`, and `inf` norms in interval/box form.

### 5.1 Frobenius norm

For a real matrix,

\[
\|A\|_F = \left(\sum_{i,j} a_{ij}^2\right)^{1/2}.
\]

For complex matrices,

\[
\|A\|_F = \left(\sum_{i,j} |a_{ij}|^2\right)^{1/2}.
\]

This structure is compatible with interval accumulation, so the basic Frobenius norm path is relatively natural.

### 5.2 One norm and infinity norm

\[
\|A\|_1 = \max_j \sum_i |a_{ij}|, \qquad
\|A\|_\infty = \max_i \sum_j |a_{ij}|.
\]

In the current implementation, the absolute values and row/column sums are interval-aware, but the final maximum is currently taken from midpoint summaries rather than a fully propagated interval maximum. So these are best viewed as tightened practical bounds, not a complete verified norm calculus.

## 6. Matrix interpretation of the four modes

The same four-mode vocabulary applies to matrices, but its meaning is slightly different than for scalar functions.

### 6.1 `point`

`point` mode treats the matrix as a point matrix and runs standard linear algebra:

\[
Y = f(M).
\]

This is the fastest path and the cleanest for JAX compilation. It carries no enclosure guarantee.

### 6.2 `basic`

`basic` mode means:

- use direct interval/box matrix arithmetic where implemented,
- otherwise use midpoint linear algebra plus outward rounding.

So `basic` is not one single algorithm. It is a mode contract that selects the default interval-aware matrix path currently available for each operation.

For `matmul` and `matvec`, this is a genuine entrywise interval/box computation. For `solve` and `inv`, it is still midpoint-first.

### 6.3 `adaptive`

`adaptive` mode for matrices means midpoint-centered tightening with additional inflation based on fixed-shape sampling or wrapper-level bounds where such paths exist.

Conceptually, if

\[
A = M \pm R,
\]

then adaptive mode tries to estimate how much the output changes in a neighborhood of \(M\), using a static stencil compatible with JAX compilation.

This is mainly an engineering compromise: better practical enclosures than naive midpoint boxing, but without dynamic verified linear-algebra algorithms.

### 6.4 `rigorous`

`rigorous` mode is the strongest containment-oriented matrix mode available in the repository, but it must be interpreted operation by operation.

Today:

- `trace_rigorous` is effectively the exact interval/box diagonal sum.
- `det_rigorous` is exact through `3x3`, then midpoint-based beyond that.
- `norm_*_rigorous` currently alias the tightened `basic` paths.
- solve and factorization families do not yet have distinct verified rigorous matrix algorithms.

So for matrices, `rigorous` currently means “use the most containment-oriented implementation available for this matrix kernel”, not “all matrix operations are fully verified interval linear algebra”.

## 7. Relation to Arb / FLINT matrix arithmetic

Arb’s scalar model is ball arithmetic:

\[
x = m \pm r.
\]

For matrices, Arb-style verified linear algebra aims to propagate enclosure information through matrix algorithms themselves, not just around the final result. In a fully verified setting, one wants guarantees of the form:

\[
A \in \mathbf{A}, \quad b \in \mathbf{b}
\quad \Longrightarrow \quad
x^\ast \in \mathbf{x},
\]

where \(\mathbf{x}\) encloses the exact solution of

\[
Ax = b.
\]

arbPlusJAX does not yet provide this level of verified matrix solving for the general dense solve/factorization stack. Its current matrix implementation is therefore best read as:

- direct interval/box algebra for algebraically simple kernels,
- midpoint linear algebra plus enclosure wrappers for the harder dense kernels,
- a four-mode interface that leaves room for future verified tightening.

## 8. Practical guidance

For current users of the matrix API, the safest interpretation is:

- trust `matmul`, `matvec`, `banded_matvec`, `trace`, and small determinant formulas as the strongest native interval/box matrix paths,
- treat `solve`, `inv`, `LU`, and `QR` as midpoint computations with interval-shaped outputs,
- treat matrix `rigorous` mode as a best-available policy, not a blanket theorem about all matrix kernels.

That is the mathematically honest comparison between the current implementation and classical verified interval matrix methods.

## 9. References

References available in the local bibliography library:

- `[@Johansson2017]` for the underlying midpoint-radius interval arithmetic model.
- `[@flint2026]` for the FLINT / Arb software lineage that motivates the comparison point for rigorous arithmetic.

Bibliography source used for these keys:

- `/mnt/c/dev/references/bibliography/library/software.bib`

The local bibliography snapshot does not currently appear to contain the standard interval-linear-algebra monographs I would normally cite for the verified matrix background, so the matrix note is currently documented against the Arb/FLINT lineage plus the repository’s own implementation notes.

## 10. Local implementation pointers

For the concrete matrix semantics described here, see:

- [arb_mat.md](/home/phili/projects/arbplusJAX/docs/implementation/modules/arb_mat.md)
- [acb_mat.md](/home/phili/projects/arbplusJAX/docs/implementation/modules/acb_mat.md)
- [mat_wrappers.md](/home/phili/projects/arbplusJAX/docs/implementation/wrappers/mat_wrappers.md)
- [ball_arithmetic_and_modes.md](/home/phili/projects/arbplusJAX/docs/theory/ball_arithmetic_and_modes.md)
