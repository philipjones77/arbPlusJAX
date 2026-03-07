from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy import special


def _eval_unary(name: str, xs: np.ndarray) -> np.ndarray:
    table = {
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "gamma": special.gamma,
        "erf": special.erf,
        "erfc": special.erfc,
    }
    fn = table.get(name)
    if fn is None:
        raise ValueError(f"unsupported unary function: {name}")
    return np.asarray(fn(xs), dtype=np.float64)


def _eval_bivariate(name: str, nu: np.ndarray, z: np.ndarray) -> np.ndarray:
    table = {
        "besselj": special.jv,
        "bessely": special.yv,
        "besseli": special.iv,
        "besselk": special.kv,
    }
    fn = table.get(name)
    if fn is None:
        raise ValueError(f"unsupported bivariate function: {name}")
    return np.asarray(fn(nu, z), dtype=np.float64)


def main() -> int:
    payload = json.loads(sys.stdin.read())
    fn_name = str(payload["function"])
    xs = np.asarray(payload.get("x", []), dtype=np.float64)

    if "nu" in payload and "z" in payload:
        nu = np.asarray(payload["nu"], dtype=np.float64)
        z = np.asarray(payload["z"], dtype=np.float64)
        ys = _eval_bivariate(fn_name, nu, z)
    else:
        ys = _eval_unary(fn_name, xs)

    sys.stdout.write(json.dumps([float(v) for v in ys]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
