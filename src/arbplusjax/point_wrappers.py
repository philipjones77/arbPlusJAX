from __future__ import annotations

from . import point_wrappers_barnes as _barnes
from . import point_wrappers_core as _core
from . import point_wrappers_matrix as _matrix
from .point_wrappers_barnes import *  # noqa: F401,F403
from .point_wrappers_core import *  # noqa: F401,F403
from .point_wrappers_matrix import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_core, name)


__all__ = list(dict.fromkeys([*_core.__all__, *_matrix.__all__, *_barnes.__all__]))
