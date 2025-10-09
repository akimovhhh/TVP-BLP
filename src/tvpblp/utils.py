# src/tvpblp/utils.py

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Iterable, Sequence, Any

import numpy as np


# ----------------------------
# Public: packaged sample data
# ----------------------------
def sample_data_path() -> Path:
    """
    Return a filesystem path to the packaged sample CSV.
    Works in editable installs and wheels.
    """
    return files("tvpblp.data") / "yoghurt_sample.csv"


# ----------------------------
# Internal utilities
# ----------------------------


def ensure_list(x: Any, *, fallback: Sequence[Any] | None = None) -> list:
    """Normalize None/str/sequence into a list. If None -> fallback or []."""
    if x is None:
        return [] if fallback is None else list(fallback)
    if isinstance(x, str):
        return [x]
    return list(x)


def unique_preserve_order(seq: Iterable[Any]) -> list:
    """Return unique items of `seq` preserving first-seen order."""
    return list(dict.fromkeys(seq))


def as_float64(df, cols) -> np.ndarray:
    """Extract columns from pandas.DataFrame as a contiguous float64 ndarray."""
    # pandas import kept local to avoid a hard dependency at import time
    import pandas as pd  # noqa: F401

    return np.ascontiguousarray(df[list(cols)].to_numpy(dtype=np.float64))


def is_zero_L(L) -> bool:
    """True if L should trigger simple-logit: None/empty/scalar-0/all-zeros."""
    if L is None:
        return True
    arr = np.asarray(L, dtype=np.float64)
    return (arr.size == 0) or np.allclose(arr, 0.0)


def build_base_ppf_grid(
    q_dim: int, points_per_dim: int, qmin: float, qmax: float
) -> np.ndarray | None:
    """
    Build a tensor grid of standard-normal nodes via inverse-CDF on a
    uniform grid of quantiles in (qmin, qmax). Returns (..., q_dim) ndarray.
    """
    if q_dim <= 0:
        return None
    if not (0.0 < qmin < qmax < 1.0):
        raise ValueError("integration_int must satisfy 0 < qmin < qmax < 1")

    from scipy.stats import norm  # local import keeps top-level imports lean

    q = np.linspace(qmin, qmax, num=int(points_per_dim), dtype=np.float64)
    grids = np.meshgrid(*([q] * q_dim), indexing="ij")
    U = np.stack(grids, axis=-1)  # (..., q_dim)
    return norm.ppf(U).astype(np.float64)  # base z ~ N(0,1)


def inv_or_reciprocal(x):
    """
    Return 1/x if x is a scalar, otherwise return inv(x) if x is a square matrix.
    """
    x = np.asarray(x)
    if x.ndim == 0:  # scalar
        return 1.0 / x
    elif x.ndim == 1:
        raise ValueError("Input is a 1D array — cannot invert.")
    elif x.shape[0] != x.shape[1]:
        raise ValueError("Matrix must be square for inversion.")
    else:
        return np.linalg.inv(x)


def ensure_colvec(w):
    w = np.atleast_1d(w).astype(np.float64)
    w = np.ascontiguousarray(w.reshape(-1, 1))
    return w
