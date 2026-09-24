import numpy as np
from math import atan2, floor, pi


def hashkey(block, Qangle, W,
            mode='sr',
            strength_thresholds=(0.0001, 0.001),
            coherence_thresholds=(0.25, 0.5)):
    """Compute structural hash bucket indices for a gradient block.

    Parameters
    ----------
    block : ndarray
        2-D gradient block (e.g. 9x9).
    Qangle : int
        Number of angle buckets (typically 24).
    W : ndarray
        Square diagonal weighting matrix for the structure tensor.
    mode : str
        'sr'      – super-resolution mode (original behaviour; angle clamped
                    to [0, Qangle-1] by an explicit guard after quantisation).
        'denoise' – denoising mode; direction is unified to [0, pi) so that
                    floor(theta/pi*Qangle) always lands in [0, Qangle-1]
                    without relying on the post-hoc clamp.
    strength_thresholds : tuple of two floats
        (low, high) boundaries for strength quantisation.  Values < low map
        to bucket 0; values > high map to bucket 2.  Default (0.0001, 0.001)
        matches the original SR implementation.
    coherence_thresholds : tuple of two floats
        (low, high) boundaries for coherence quantisation.  Default (0.25, 0.5)
        matches the original SR implementation.

    Returns
    -------
    angle, strength, coherence : int
        Each in [0, Qangle-1], [0, 2], [0, 2] respectively.
    """
    # Calculate gradient
    gy, gx = np.gradient(block)

    # Transform 2D matrix into 1D array
    gx = gx.ravel()
    gy = gy.ravel()

    # Structure tensor G^T W G
    G = np.vstack((gx, gy)).T
    GTWG = G.T.dot(W).dot(G)
    w, v = np.linalg.eig(GTWG)

    # Discard negligible imaginary parts caused by floating-point noise
    w = np.real(w)
    v = np.real(v)

    # Sort by descending eigenvalue
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]

    # Dominant direction (principal eigenvector angle)
    theta = atan2(v[1, 0], v[0, 0])
    if theta < 0:
        theta = theta + pi

    # In denoise mode, map theta strictly into [0, pi) so that
    # floor(theta/pi*Qangle) is always in [0, Qangle-1].
    if mode == 'denoise' and theta >= pi:
        theta = pi * (1.0 - 1e-10)

    # Eigenvalue-derived features (non-negative guard for numerical errors)
    lamda = w[0]
    sqrtl1 = np.sqrt(max(float(w[0]), 0.0))
    sqrtl2 = np.sqrt(max(float(w[1]), 0.0))
    if sqrtl1 + sqrtl2 == 0:
        u = 0.0
    else:
        u = (sqrtl1 - sqrtl2) / (sqrtl1 + sqrtl2)

    # Quantise angle
    angle = floor(theta / pi * Qangle)

    # Quantise strength
    s_lo, s_hi = strength_thresholds
    if lamda < s_lo:
        strength = 0
    elif lamda > s_hi:
        strength = 2
    else:
        strength = 1

    # Quantise coherence
    c_lo, c_hi = coherence_thresholds
    if u < c_lo:
        coherence = 0
    elif u > c_hi:
        coherence = 2
    else:
        coherence = 1

    # Hard clamp for SR mode (preserves original behaviour); denoise mode
    # should never need this thanks to the theta adjustment above.
    if angle >= Qangle:
        angle = Qangle - 1
    elif angle < 0:
        angle = 0

    return angle, strength, coherence
