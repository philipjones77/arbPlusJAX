from __future__ import annotations

import json
import math
import sys


def _eval_unary(name: str, x: float) -> float:
    table = {
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "gamma": math.gamma,
        "erf": math.erf,
        "erfc": math.erfc,
    }
    fn = table.get(name)
    if fn is None:
        raise ValueError(f"unsupported function for template adapter: {name}")
    return float(fn(x))


def main() -> int:
    payload = json.loads(sys.stdin.read())
    fn_name = payload["function"]
    xs = payload["x"]
    # Contract:
    # - unary: {"function": str, "x": [..]}
    # - bivariate: {"function": str, "x": [..], "nu": [..], "z": [..]}
    # For a production Boost adapter, replace _eval_unary with a C++/Boost-backed binary.
    ys = [_eval_unary(fn_name, float(x)) for x in xs]
    sys.stdout.write(json.dumps(ys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
