from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import make_curvature_operator


def fisher_matvec(logprob, params, data, v):
    def scalar_logprob(p):
        return logprob(p, data)

    grad_fun = jax.grad(scalar_logprob)
    return jax.jvp(grad_fun, (params,), (v,))[1]


def make_fisher_operator(logprob, params, data):
    params_arr = jnp.asarray(params)
    dim = int(params_arr.size)
    return make_curvature_operator(
        shape=(dim, dim),
        dtype=params_arr.dtype,
        matvec=lambda v: fisher_matvec(logprob, params_arr, data, v),
        rmatvec=lambda v: fisher_matvec(logprob, params_arr, data, v),
        metadata={"kind": "fisher", "symmetric": True, "psd": True},
    )
