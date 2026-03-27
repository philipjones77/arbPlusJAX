from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import make_curvature_operator


def ggn_matvec(model, loss, params, data, v):
    def scalar_loss(p):
        return loss(model(p, data), data)

    grad_fun = jax.grad(scalar_loss)
    return jax.jvp(grad_fun, (params,), (v,))[1]


def make_ggn_operator(model, loss, params, data):
    params_arr = jnp.asarray(params)
    dim = int(params_arr.size)
    return make_curvature_operator(
        shape=(dim, dim),
        dtype=params_arr.dtype,
        matvec=lambda v: ggn_matvec(model, loss, params_arr, data, v),
        rmatvec=lambda v: ggn_matvec(model, loss, params_arr, data, v),
        metadata={"kind": "ggn", "symmetric": True, "psd": True},
    )
