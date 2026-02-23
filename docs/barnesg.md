# Barnes G notes

Barnes G is defined by the functional equation:

- `G(z+1) = Γ(z) G(z)` with `G(1)=1`.

We implement `log G(z)` using a log-asymptotic expansion for `log G(w+1)` with `w=z-1`:

```
log G(w+1) ≈ (w^2/2 - 1/12) log w - 3 w^2 / 4 + (w/2) log(2π) + log A
           + Σ_{k≥1} B_{2k+2} / (4 k (k+1) w^{2k})
```

where `A` is the Glaisher–Kinkelin constant and `B_n` are Bernoulli numbers.

For general `z`, we shift to `Re(z) >= 5` using the recurrence and subtract `log Γ(z+k)` terms.

Modes:
- Baseline: midpoint evaluation with endpoint sampling.
- Adaptive: midpoint + sampling-based Lipschitz bounds in ball wrappers.
- Rigorous: gradient/Lipschitz bounds with interval inputs in ball wrappers.

Poles: for nonpositive integers on the real axis, Barnes G has poles; our interval path returns full intervals when the input box crosses a pole.
