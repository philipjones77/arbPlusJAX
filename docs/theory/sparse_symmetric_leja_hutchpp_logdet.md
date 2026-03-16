Last updated: 2026-03-16T21:00:00Z

# Sparse Symmetric Leja Plus Hutch++ Log-Det

## Scope

This note describes the sparse symmetric positive-definite log-determinant path currently implemented in [jrb_mat.py](/home/phili/projects/arbplusJAX/src/arbplusjax/jrb_mat.py):

- `jrb_mat_log_action_leja_point`
- `jrb_mat_log_action_leja_with_diagnostics_point`
- `jrb_mat_hutchpp_trace_point`
- `jrb_mat_logdet_leja_hutchpp_point`
- `jrb_mat_logdet_leja_hutchpp_with_diagnostics_point`
- `jrb_mat_bcoo_spectral_bounds_adaptive`
- `jrb_mat_bcoo_logdet_leja_hutchpp_point`
- `jrb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point`

The target problem is

$$
\log \det(A), \qquad A \in \mathbb{R}^{n \times n}, \quad A = A^\top, \quad A \succ 0,
$$

with `A` accessed through a sparse `BCOO` operator or an operator callback. The implementation is point-mode and matrix-free: it does not claim interval enclosure or validated spectral certification. Its design follows the Newton-Leja plus Hutch++ direction in [@Mbingui2026], with the trace-estimation split following Hutch++ [@Meyer2021].

## 1. Log-Det As A Trace

For SPD `A`,

$$
\log \det(A) = \operatorname{tr}(\log A).
$$

So the task is reduced to approximating the trace of the matrix function `log(A)`.

Instead of forming `\log(A)` explicitly, the implementation approximates the action

$$
v \mapsto \log(A) v
$$

and then estimates the trace from repeated action evaluations.

## 2. Newton-Leja Action Approximation

Let the spectral interval be bounded by

$$
\sigma(A) \subseteq [a,b], \qquad 0 < a < b.
$$

Choose real Leja nodes

$$
\xi_0, \xi_1, \ldots, \xi_{m-1} \in [a,b].
$$

The scalar logarithm is interpolated in Newton form:

$$
p_{m-1}(x)
= c_0 + c_1(x-\xi_0) + c_2(x-\xi_0)(x-\xi_1) + \cdots
+ c_{m-1}\prod_{j=0}^{m-2}(x-\xi_j),
$$

where the coefficients `c_k` are the divided differences of `\log(x)` at the nodes.

The matrix action is then evaluated by the same Newton basis recurrence:

$$
w_0 = v,
$$

$$
w_k = (A - \xi_{k-1} I) w_{k-1}, \qquad k \ge 1,
$$

$$
p_{m-1}(A) v = \sum_{k=0}^{m-1} c_k w_k.
$$

In the current code:

- `_jrb_leja_points_interval_point` builds a discrete real Leja sequence on `[a,b]`
- `_jrb_newton_divided_differences_point` computes the divided differences of `\log`
- `_jrb_funm_action_newton_point` evaluates the fixed-degree Newton action

## 3. Adaptive Degree Stop Rule

The first sparse integration used a fixed polynomial degree. The current path adds an adaptive termination rule through `_jrb_funm_action_newton_adaptive_point`.

At step `k`, define the Newton increment

$$
t_k = c_k w_k
$$

and the running approximation

$$
y_k = \sum_{j=0}^k c_j w_j.
$$

The adaptive rule stops once

$$
\lVert t_k \rVert_2
\le
\text{atol} + \text{rtol} \max(\lVert y_k \rVert_2, 1),
$$

subject to a lower truncation floor `k + 1 \ge m_{\min}`. In code, this is exposed through:

- `max_degree`
- `min_degree`
- `rtol`
- `atol`

This is an a posteriori action-convergence rule. It is practical and cheap because it reuses the Newton basis vectors already being computed. It is not a proof of uniform interpolation error over the entire spectrum.

## 4. Hutch++ Trace Estimation

Let `F = \log(A)`. Hutch++ estimates `\operatorname{tr}(F)` using a low-rank sketch plus a residual Hutchinson estimator [@Meyer2021].

With sketch probes collected in the columns of `S` and residual probes in the columns of `G`,

$$
Y = F S,
$$

then compute an orthonormal basis `Q` for the range of `Y`. The trace is decomposed as

$$
\operatorname{tr}(F)
= \operatorname{tr}(Q^\top F Q)
  + \operatorname{tr}((I - QQ^\top) F (I - QQ^\top)).
$$

The implementation estimates this as

$$
\widehat{\tau}
= \operatorname{tr}(Q^\top F Q)
  + \frac{1}{s}
    \sum_{i=1}^s
    g_i^\top (I - QQ^\top) F (I - QQ^\top) g_i,
$$

where the `g_i` are the residual probes. This is exactly the role of `jrb_mat_hutchpp_trace_point`.

For the log-determinant path,

$$
\widehat{\log \det(A)} = \widehat{\operatorname{tr}(\log A)}.
$$

## 5. Sparse Spectral Bounds

The interpolation interval `[a,b]` matters because the Newton-Leja action approximates `\log(x)` on that interval.

### 5.1 Conservative default: Gershgorin

For each row `i`,

$$
R_i = \sum_{j \ne i} |a_{ij}|.
$$

Every eigenvalue lies in the union of Gershgorin discs, so for symmetric real matrices the implementation uses the scalar interval

$$
a_{\mathrm{G}} = \min_i (a_{ii} - R_i),
\qquad
b_{\mathrm{G}} = \max_i (a_{ii} + R_i).
$$

This is available in `jrb_mat_bcoo_gershgorin_bounds`.

### 5.2 Heuristic refinement: multi-start short Lanczos

The current sparse `BCOO` convenience path adds `jrb_mat_bcoo_spectral_bounds_adaptive`. It runs a short symmetric Lanczos process from several deterministic start vectors, computes the Ritz interval

$$
[\theta_{\min}, \theta_{\max}],
$$

and expands it by the final recurrence tail size `\beta_k`:

$$
a_{\mathrm{L}} = \max(\varepsilon, \theta_{\min} - c \beta_k),
\qquad
b_{\mathrm{L}} = \max(a_{\mathrm{L}} + \varepsilon, \theta_{\max} + c \beta_k),
$$

with a safety factor `c > 1`.

The returned interval is then clipped against the Gershgorin interval:

$$
a = \max(\varepsilon, \min(a_{\mathrm{G}}, a_{\mathrm{L}})),
\qquad
b = \max(a + \varepsilon, \min(b_{\mathrm{G}}, b_{\mathrm{L}})).
$$

This is a heuristic narrowing rule for point-mode performance. It is useful for the current implementation, but it is not a rigorous spectral certificate.

## 6. Full Sparse BCOO Path

For a sparse `BCOO` matrix `A`, the implemented convenience wrapper does:

1. Build a sparse operator `v \mapsto Av`.
2. Estimate `[a,b]` automatically with `jrb_mat_bcoo_spectral_bounds_adaptive` if the user does not supply spectral bounds.
3. Approximate `\log(A) v` by adaptive or fixed-degree Newton-Leja interpolation.
4. Estimate `\operatorname{tr}(\log A)` with Hutch++.

Symbolically, the current estimator is

$$
\widehat{\log \det(A)}
=
\operatorname{Hutch++}
\bigl(v \mapsto p_{m-1}(A)v \bigr),
$$

where `p_{m-1}` is either the fixed-degree or adaptively truncated Newton-Leja polynomial for `\log`.

## 7. Diagnostics

The current diagnostics reuse `JrbMatKrylovDiagnostics`. For the Leja plus Hutch++ path:

- `algorithm_code = 3`
- `steps` records the used polynomial degree
- `tail_norm` records the final Newton increment norm on the representative action probe
- `probe_count` records sketch plus residual probes

This is enough to expose the adaptive stop depth and a lightweight tail-size indicator without adding a second diagnostics type.

## 8. What Is Complete And What Is Still Approximate

The current implementation now includes:

- sparse `BCOO` convenience wrappers instead of operator-only entry
- automatic sparse spectral interval selection
- adaptive Newton-Leja stopping
- test coverage for exact diagonal sparse cases and small SPD sparse cases

It still does not provide:

- a rigorous proof-producing spectral bound refinement
- a validated interpolation remainder bound
- a mathematically certified adaptive Hutch++ stopping rule
- a full arbitrary-precision ball-arithmetic version of the sparse log-determinant estimator

So this path should be understood as a well-structured, JAX-native point-mode sparse estimator, not as a verified interval algorithm.

## References

- `[@Mbingui2026]`
- `[@Meyer2021]`
