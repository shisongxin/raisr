"""Regularized normal-equation solver (historical function name retained)."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def cgls(A, b, regularization=1e-4):
    """Solve Q h = V with a ridge prior toward the center/identity filter.

    lambda = regularization * mean(diag(Q)); the prior preserves the bilinear
    input in empty or poorly constrained directions instead of producing black.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or b.shape != (A.shape[0],):
        raise ValueError("Expected square Q and matching one-dimensional V")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Q and V must be finite")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    size = b.size
    if size == 0:
        raise ValueError("Empty filter vector")
    prior = np.zeros(size)
    prior[size // 2] = 1.0
    if not np.any(A):
        if np.any(b):
            raise ValueError("Nonzero V with zero Q")
        return prior
    A = (A + A.T) * 0.5
    scale = np.trace(A) / size
    if scale <= 0:
        raise ValueError("Q must have positive trace")
    ridge = max(regularization * scale, np.finfo(np.float64).eps * scale)
    system = A + np.eye(size) * ridge
    rhs = b + ridge * prior
    try:
        result = cho_solve(cho_factor(system, lower=True), rhs)
    except np.linalg.LinAlgError:
        result = np.linalg.lstsq(system, rhs, rcond=None)[0]
    if not np.isfinite(result).all():
        raise ValueError("Filter solution is not finite")
    return result
