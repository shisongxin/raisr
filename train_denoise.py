"""RAISR denoising trainer.

Trains per-bucket Wiener filters for additive white Gaussian noise removal.
Usage:
    python train_denoise.py --help
"""
import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from cgls import cgls
from filterplot import filterplot
from gaussian2d import gaussian2d
from hashkey import hashkey

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATCHSIZE    = 7          # input patch side length
GRADSIZE     = 5           # gradient block side length
QANGLE       = 24
QSTRENGTH    = 3
QCOHERENCE   = 3
PATCHMARGIN  = PATCHSIZE  // 2   # 5
GRADMARGIN   = GRADSIZE   // 2   # 4
N_FILTERS    = QANGLE * QSTRENGTH * QCOHERENCE   # 216
PATCH_AREA   = PATCHSIZE * PATCHSIZE             # 121

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Train RAISR denoising filters")
    p.add_argument("--sigma",             type=float, default=25.0,
                   help="Noise standard deviation in [0,255] scale (default 25)")
    p.add_argument("--max-images",        type=int,   default=0,
                   help="Maximum training images to use (0 = all)")
    p.add_argument("--samples-per-image", type=int,   default=20000,
                   help="Max center positions per image (0 = all pixels)")
    p.add_argument("--hash-mode",         default="guided", choices=["noisy","guided"],
                   help="Compute hash on noisy image or lightly smoothed guide")
    p.add_argument("--guide-sigma",       type=float, default=0.8,
                   help="Gaussian sigma for guided hash pre-smoothing (pixels)")
    p.add_argument("--regularization",    type=float, default=1e-4,
                   help="Ridge regularization coefficient for cgls solver")
    p.add_argument("--no-calibrate",      action="store_true",
                   help="Skip strength-threshold calibration and use the SR "
                        "defaults (0.0001, 0.001). Useful for baseline diagnosis.")
    p.add_argument("--calib-images",      type=int,   default=30,
                   help="Number of training images sampled to calibrate "
                        "strength thresholds (default 30)")
    p.add_argument("--calib-samples",     type=int,   default=10000,
                   help="Hash samples per image during calibration (default 10000)")
    p.add_argument("--calib-method",      default="noise-floor",
                   choices=["noise-floor", "quantile"],
                   help="Strength-threshold calibration method. 'noise-floor' "
                        "(default) anchors LOW to the AWGN noise floor so smooth "
                        "regions cluster in bucket 0; 'quantile' splits samples "
                        "into equal thirds (no physical meaning).")
    p.add_argument("--calib-low-pct",     type=float, default=95.0,
                   help="noise-floor method: percentile of the pure-noise lambda "
                        "distribution used as the LOW threshold (default 95). "
                        "Below it, lambda is indistinguishable from noise -> smooth.")
    p.add_argument("--calib-high-pct",    type=float, default=50.0,
                   help="noise-floor method: percentile of the signal-bearing "
                        "lambda (lambda>LOW) used as the HIGH threshold "
                        "(default 50 = median of structured samples).")
    p.add_argument("--output-dir",        default="runs/denoise_sigma25",
                   help="Directory to save model and statistics")
    p.add_argument("-q", "--qmatrix",     default=None,
                   help="Resume: path to existing Q statistics file")
    p.add_argument("-v", "--vmatrix",     default=None,
                   help="Resume: path to existing V statistics file")
    p.add_argument("-p", "--plot",        action="store_true",
                   help="Plot the 216 learned filters after solving")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def reflect_pad(img, margin):
    """Reflect-pad a 2-D float array by `margin` pixels on all sides."""
    return np.pad(img, margin, mode='reflect')


def load_gray(path):
    """Read an image and return float64 Y channel in [0, 1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Cannot read image: {path}")
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0].astype(np.float64) / 255.0


def make_guide(noisy, guide_sigma):
    """Return a lightly Gaussian-smoothed version of noisy for hash computation."""
    return gaussian_filter(noisy, sigma=guide_sigma, mode='reflect')


def _lamda_of(block, W):
    """Dominant structure-tensor eigenvalue for a gradient block (matches hashkey)."""
    gy, gx = np.gradient(block)
    G = np.vstack((gx.ravel(), gy.ravel())).T
    GTWG = G.T.dot(W).dot(G)
    w = np.real(np.linalg.eigvals(GTWG))
    return float(w.max())


def _sample_real_lamda_and_u(imagelist, weighting, sigma_f, args):
    """Sample lambda and u from real noisy training images."""
    calib_paths = imagelist[:min(args.calib_images, len(imagelist))]
    lamdas, us = [], []
    for idx, path in enumerate(calib_paths):
        ss = np.random.SeedSequence(1234, spawn_key=(0xCA11B, idx))
        rng = np.random.default_rng(ss)
        try:
            clean = load_gray(path)
        except IOError:
            continue
        noisy = clean + rng.normal(0.0, sigma_f, size=clean.shape)
        hash_src = make_guide(noisy, args.guide_sigma) if args.hash_mode == "guided" else noisy
        hash_pad = reflect_pad(hash_src, GRADMARGIN)
        H, W = clean.shape
        n = min(args.calib_samples, H * W)
        rows = rng.integers(0, H, n)
        cols = rng.integers(0, W, n)
        for r, c in zip(rows, cols):
            rg, cg = r + GRADMARGIN, c + GRADMARGIN
            block = hash_pad[rg - GRADMARGIN: rg + GRADMARGIN + 1,
                             cg - GRADMARGIN: cg + GRADMARGIN + 1]
            gy, gx = np.gradient(block)
            G = np.vstack((gx.ravel(), gy.ravel())).T
            GTWG = G.T.dot(weighting).dot(G)
            w = np.sort(np.real(np.linalg.eigvals(GTWG)))[::-1]
            lamdas.append(float(w[0]))
            l1 = np.sqrt(max(w[0], 0.0)); l2 = np.sqrt(max(w[1], 0.0))
            us.append((l1 - l2) / (l1 + l2) if (l1 + l2) > 0 else 0.0)
        print_progress(idx, len(calib_paths))
    print()
    return np.array(lamdas), np.array(us)


def _estimate_noise_floor(weighting, sigma_f, guide_sigma, hash_mode,
                          n_mc=100000):
    """Monte-Carlo lambda distribution of pure AWGN on a flat patch.

    In a flat region the clean gradient is zero, so lambda is produced entirely
    by the noise. This distribution is the physical anchor for the LOW strength
    threshold: below its upper tail, an observed lambda is indistinguishable
    from pure noise, i.e. the region is smooth. If guided hashing is used, the
    same smoothing is applied to each synthetic noise patch so the floor matches
    the hash source seen at training time.
    """
    rng = np.random.default_rng(np.random.SeedSequence(1234, spawn_key=(0xF100,)))
    lamdas = np.empty(n_mc)
    # Generate on a slightly larger tile when smoothing so edge effects of the
    # Gaussian match reflect-padding behaviour, then crop to GRADSIZE.
    pad = GRADMARGIN if hash_mode == "guided" else 0
    size = GRADSIZE + 2 * pad
    for i in range(n_mc):
        patch = rng.normal(0.0, sigma_f, size=(size, size))
        if hash_mode == "guided":
            patch = gaussian_filter(patch, sigma=guide_sigma, mode='reflect')
            patch = patch[pad:pad + GRADSIZE, pad:pad + GRADSIZE]
        lamdas[i] = _lamda_of(patch, weighting)
    return lamdas


def calibrate_strength_thresholds(imagelist, weighting, sigma_f, args):
    """Return (low, high) strength thresholds and (low, high) coherence thresholds.

    Strength: noise-floor anchored (default) or equal-quantile.
    Coherence: signal-restricted equal-quantile (1/3, 2/3 of samples where
               lambda > strength_low). This gives physically meaningful groups:
               bucket 0 = low direction reliability, 1 = moderate, 2 = strong.

    Returns (strength_thresholds, coherence_thresholds) as two (low, high) tuples.
    """
    print(f"Calibrating thresholds on "
          f"{min(args.calib_images, len(imagelist))} images "
          f"(method={args.calib_method}) ...")
    real_l, real_u = _sample_real_lamda_and_u(imagelist, weighting, sigma_f, args)

    # --- strength ---
    if args.calib_method == "quantile":
        s_low  = float(np.percentile(real_l, 100.0 / 3.0))
        s_high = float(np.percentile(real_l, 200.0 / 3.0))
    else:  # noise-floor
        floor = _estimate_noise_floor(weighting, sigma_f,
                                      args.guide_sigma, args.hash_mode)
        s_low = float(np.percentile(floor, args.calib_low_pct))
        signal_l = real_l[real_l > s_low]
        s_high = float(np.percentile(signal_l, args.calib_high_pct)) \
                 if signal_l.size else s_low * 2.0
        print(f"  noise floor: median={np.median(floor):.6f}, "
              f"P{args.calib_low_pct:g}={s_low:.6f}")
    if not (s_high > s_low):
        s_high = s_low + max(abs(s_low) * 1e-6, 1e-9)

    # --- coherence: 1/3, 2/3 quantiles of signal-bearing samples only ---
    signal_u = real_u[real_l > s_low]
    if signal_u.size >= 6:
        c_low  = float(np.percentile(signal_u, 100.0 / 3.0))
        c_high = float(np.percentile(signal_u, 200.0 / 3.0))
    else:
        c_low, c_high = 0.25, 0.5   # fallback to SR defaults if too few signal samples
    if not (c_high > c_low):
        c_high = c_low + max(abs(c_low) * 1e-6, 1e-9)

    # --- report ---
    b0 = float(np.mean(real_l < s_low))
    b2 = float(np.mean(real_l > s_high))
    print(f"  real lambda range: [{real_l.min():.6f}, {real_l.max():.6f}]  n={len(real_l)}")
    print(f"  strength thresholds: low={s_low:.6f}, high={s_high:.6f}")
    print(f"  strength split: smooth={b0*100:.1f}%  "
          f"moderate={(1-b0-b2)*100:.1f}%  strong={b2*100:.1f}%")
    print(f"  coherence thresholds (signal-restricted): "
          f"low={c_low:.4f}, high={c_high:.4f}")
    if signal_u.size:
        cu0 = float(np.mean(signal_u < c_low))
        cu2 = float(np.mean(signal_u > c_high))
        print(f"  coherence split (signal only): "
              f"low={cu0*100:.1f}%  mid={(1-cu0-cu2)*100:.1f}%  high={cu2*100:.1f}%")

    return (s_low, s_high), (c_low, c_high)


def print_progress(done, total):
    print('\r|', end='')
    print('#' * round((done+1)*100/total/2), end='')
    print(' ' * (50 - round((done+1)*100/total/2)), end='')
    print('|  ' + str(round((done+1)*100/total)) + '%', end='')
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Q/V accumulation for one image
# ---------------------------------------------------------------------------
def accumulate(noisy_pad, hash_pad, clean_orig,
               center_rows, center_cols,
               Q, V, counts,
               weighting, args, strength_thresholds):
    """Accumulate Q, V, counts for the selected center positions."""
    n_pos = len(center_rows)
    for i, (row, col) in enumerate(zip(center_rows, center_cols)):
        if i%1000==0:
            print_progress(i, n_pos)

        # patch from noisy (padded)
        r_p = row + PATCHMARGIN   # index into padded array
        c_p = col + PATCHMARGIN
        patch = noisy_pad[r_p - PATCHMARGIN: r_p + PATCHMARGIN + 1,
                          c_p - PATCHMARGIN: c_p + PATCHMARGIN + 1].ravel()

        # gradient block from hash source (padded)
        r_g = row + GRADMARGIN
        c_g = col + GRADMARGIN
        gblock = hash_pad[r_g - GRADMARGIN: r_g + GRADMARGIN + 1,
                          c_g - GRADMARGIN: c_g + GRADMARGIN + 1]

        angle, strength, coherence = hashkey(gblock, QANGLE, weighting,
                                             mode='denoise',
                                             strength_thresholds=strength_thresholds)

        label = clean_orig[row, col]

        x = patch          # shape (121,)
        Q[angle, strength, coherence] += np.outer(x, x)
        V[angle, strength, coherence] += x * label
        counts[angle, strength, coherence] += 1


# ---------------------------------------------------------------------------
# Eight-fold statistical augmentation
# ---------------------------------------------------------------------------
# Patch index layout: row-major, shape (PATCHSIZE, PATCHSIZE).
# For a 2-D index (r, c) the flat index is r*PATCHSIZE + c.

def _make_patch_permutation(transform_fn):
    """Return a flat permutation array for the given 2-D transform."""
    idx = np.arange(PATCH_AREA).reshape(PATCHSIZE, PATCHSIZE)
    transformed = transform_fn(idx)
    return transformed.ravel()


def _rot90_indices(m):  return np.rot90(m, k=1)
def _rot180_indices(m): return np.rot90(m, k=2)
def _rot270_indices(m): return np.rot90(m, k=3)
def _flip_indices(m):   return np.fliplr(m)
def _fliprot90_indices(m):  return np.rot90(np.fliplr(m), k=1)
def _fliprot180_indices(m): return np.rot90(np.fliplr(m), k=2)
def _fliprot270_indices(m): return np.rot90(np.fliplr(m), k=3)

_TRANSFORMS = [
    _rot90_indices, _rot180_indices, _rot270_indices,
    _flip_indices, _fliprot90_indices, _fliprot180_indices, _fliprot270_indices,
]

# Angle bucket remapping per transform (24 angle buckets, direction in [0,pi)).
# A rotation by k*90° shifts the dominant gradient angle by k*90°.
# A flip (horizontal) negates the x-component of the gradient, so angle -> pi - angle
# which maps bucket i -> 24 - 1 - i (approximately, with wrap).
def _angle_remap_rot(angle, k):
    """Remap angle bucket after rotating image by k*90 degrees CCW.

    Orientation lives in [0, pi) over QANGLE buckets, so pi == QANGLE buckets.
    A 90-degree rotation shifts orientation by pi/2 == QANGLE//2 buckets
    (12 for QANGLE=24), NOT QANGLE//4. With QANGLE//2, a 180-degree rotation
    maps a bucket back to itself, as it must for undirected orientation.
    """
    return (angle + k * (QANGLE // 2)) % QANGLE

def _angle_remap_flip(angle):
    """Remap angle bucket after horizontal flip."""
    return (QANGLE - 1 - angle) % QANGLE

_ANGLE_REMAPS = [
    lambda a: _angle_remap_rot(a, 1),
    lambda a: _angle_remap_rot(a, 2),
    lambda a: _angle_remap_rot(a, 3),
    lambda a: _angle_remap_flip(a),
    lambda a: _angle_remap_flip(_angle_remap_rot(a, 1)),
    lambda a: _angle_remap_flip(_angle_remap_rot(a, 2)),
    lambda a: _angle_remap_flip(_angle_remap_rot(a, 3)),
]

_PERMS = [_make_patch_permutation(fn) for fn in _TRANSFORMS]


def augment_statistics(Q_orig, V_orig, counts_orig):
    """Apply eight-fold augmentation to Q/V/counts and return augmented totals.

    The original (identity) transform is included, so the result is 8× the
    original counts. Q_orig / V_orig are NOT modified.
    """
    Q_aug = Q_orig.copy()
    V_aug = V_orig.copy()
    counts_aug = counts_orig.copy()

    for perm, angle_remap in zip(_PERMS, _ANGLE_REMAPS):
        for a in range(QANGLE):
            a2 = angle_remap(a)
            for s in range(QSTRENGTH):
                for c in range(QCOHERENCE):
                    Q_src = Q_orig[a, s, c]     # (121,121)
                    V_src = V_orig[a, s, c]     # (121,)
                    cnt   = counts_orig[a, s, c]
                    if cnt == 0:
                        continue
                    # Permute rows and columns of Q, elements of V
                    Q_t = Q_src[np.ix_(perm, perm)]
                    V_t = V_src[perm]
                    Q_aug[a2, s, c] += Q_t
                    V_aug[a2, s, c] += V_t
                    counts_aug[a2, s, c] += cnt

    return Q_aug, V_aug, counts_aug


# ---------------------------------------------------------------------------
# Sanity checks on solved filters
# ---------------------------------------------------------------------------
def check_filters(h):
    ok = True
    if not np.isfinite(h).all():
        print("[WARN] Some filter coefficients are not finite.")
        ok = False
    coeff_sums = h.sum(axis=-1)   # shape (24,3,3)
    min_s, max_s = coeff_sums.min(), coeff_sums.max()
    print(f"  Filter coefficient sums: min={min_s:.4f}, max={max_s:.4f} (expect near 1.0)")
    if abs(min_s - 1.0) > 0.2 or abs(max_s - 1.0) > 0.2:
        print("[WARN] Some filter sums deviate notably from 1.0; brightness shift possible.")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Solve all buckets
# ---------------------------------------------------------------------------
def solve_filters(Q, V, counts, regularization):
    h0 = np.zeros(PATCH_AREA)
    h0[PATCH_AREA // 2] = 1.0

    h = np.zeros((QANGLE, QSTRENGTH, QCOHERENCE, PATCH_AREA))
    empty_buckets = 0

    total = QANGLE * QSTRENGTH * QCOHERENCE
    done = 0
    for a in range(QANGLE):
        for s in range(QSTRENGTH):
            for c in range(QCOHERENCE):
                print_progress(done, total)
                done += 1
                if counts[a, s, c] == 0:
                    h[a, s, c] = h0
                    empty_buckets += 1
                    continue
                try:
                    h[a, s, c] = cgls(Q[a, s, c], V[a, s, c], regularization)
                except Exception:
                    h[a, s, c] = h0
                    empty_buckets += 1
    print()
    return h, empty_buckets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    sigma_f = args.sigma / 255.0
    os.makedirs(args.output_dir, exist_ok=True)

    # Weighting matrix for structure tensor (same convention as train.py)
    weighting = gaussian2d([GRADSIZE, GRADSIZE], 2)
    weighting = np.diag(weighting.ravel())

    # Load training image list from split manifest if present, else scan trainpath
    imagelist = []
    for parent, _, filenames in os.walk("train"):
        for fn in filenames:
            if fn.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                imagelist.append(os.path.join(parent, fn))
    imagelist = sorted(imagelist)

    if args.max_images > 0:
        imagelist = imagelist[:args.max_images]

    n_images = len(imagelist)
    print(f"Training images : {n_images}")
    print(f"sigma           : {args.sigma} ({sigma_f:.5f} in float)")
    print(f"hash mode       : {args.hash_mode}" +
          (f"  guide_sigma={args.guide_sigma}" if args.hash_mode == "guided" else ""))
    print(f"samples/image   : {args.samples_per_image or 'all'}")
    print(f"output dir      : {args.output_dir}")

    # Resume: load existing Q/V or allocate fresh
    Q = np.zeros((QANGLE, QSTRENGTH, QCOHERENCE, PATCH_AREA, PATCH_AREA), dtype=np.float64)
    V = np.zeros((QANGLE, QSTRENGTH, QCOHERENCE, PATCH_AREA),             dtype=np.float64)
    counts = np.zeros((QANGLE, QSTRENGTH, QCOHERENCE),                    dtype=np.int64)

    completed_images = set()
    if args.qmatrix and args.vmatrix:
        print("Resuming from existing Q/V ...")
        with open(args.qmatrix, "rb") as f:
            qdata = pickle.load(f)
        with open(args.vmatrix, "rb") as f:
            vdata = pickle.load(f)
        Q      = qdata["Q"]
        V      = vdata["V"]
        counts = qdata["counts"]
        completed_images = set(qdata.get("completed_images", []))
        print(f"  Loaded {len(completed_images)} already-completed images.")

    # Determine strength and coherence thresholds.
    # On resume, reuse the thresholds stored with the statistics so that new
    # samples land in the same buckets as the ones already accumulated.
    SR_DEFAULT_STRENGTH   = (0.0001, 0.001)
    SR_DEFAULT_COHERENCE  = (0.25, 0.5)
    resumed_s_thresh = None
    resumed_c_thresh = None
    if args.qmatrix and args.vmatrix:
        resumed_s_thresh = qdata.get("strength_thresholds")
        resumed_c_thresh = qdata.get("coherence_thresholds")

    if resumed_s_thresh is not None:
        strength_thresholds  = tuple(resumed_s_thresh)
        coherence_thresholds = tuple(resumed_c_thresh) if resumed_c_thresh else SR_DEFAULT_COHERENCE
        print(f"Reusing stored thresholds: "
              f"strength=({strength_thresholds[0]:.6f}, {strength_thresholds[1]:.6f})  "
              f"coherence=({coherence_thresholds[0]:.4f}, {coherence_thresholds[1]:.4f})")
    elif args.no_calibrate:
        strength_thresholds  = SR_DEFAULT_STRENGTH
        coherence_thresholds = SR_DEFAULT_COHERENCE
        print(f"Calibration disabled; using SR defaults: "
              f"strength={SR_DEFAULT_STRENGTH}  coherence={SR_DEFAULT_COHERENCE}")
    else:
        strength_thresholds, coherence_thresholds = calibrate_strength_thresholds(
            imagelist, weighting, sigma_f, args)

    # Noise RNG base: per-image seeds are derived from the fixed list index
    # via SeedSequence spawn_key, so each image's noise is stable and depends
    # only on its position in the manifest, independent of processing order or
    # which images are skipped during resume.
    noise_base_seed = 1234

    t_start = time.time()

    for img_idx, image_path in enumerate(imagelist):
        print('\r', end='')
        print(' ' * 60, end='')
        print('\rProcessing image ' + str(img_idx+1) + ' of ' + str(n_images) + ' (' + image_path + ')')

        if image_path in completed_images:
            print('  [skip] already accumulated')
            continue

        # Stable per-image noise seed derived from the fixed list index.
        # spawn_key ties the seed to img_idx only, so resume/skip cannot shift it.
        img_ss = np.random.SeedSequence(noise_base_seed, spawn_key=(img_idx,))
        rng = np.random.default_rng(img_ss)

        try:
            clean = load_gray(image_path)
        except IOError as e:
            print(f'  [WARN] {e}')
            continue

        H, W = clean.shape

        # Add AWGN
        noisy = clean + rng.normal(0.0, sigma_f, size=clean.shape)

        # Guide for hash (if requested)
        if args.hash_mode == "guided":
            hash_src = make_guide(noisy, args.guide_sigma)
        else:
            hash_src = noisy

        # Reflect-pad both noisy (for patches) and hash source (for gradient blocks)
        noisy_pad    = reflect_pad(noisy,    PATCHMARGIN)
        hash_pad     = reflect_pad(hash_src, GRADMARGIN)

        # Select center positions (all valid original-image coordinates)
        all_rows = np.arange(H)
        all_cols = np.arange(W)
        row_grid, col_grid = np.meshgrid(all_rows, all_cols, indexing='ij')
        all_centers = np.stack([row_grid.ravel(), col_grid.ravel()], axis=1)  # (H*W, 2)

        n_pix = len(all_centers)
        if args.samples_per_image > 0 and n_pix > args.samples_per_image:
            chosen = rng.choice(n_pix, size=args.samples_per_image, replace=False)
            centers = all_centers[chosen]
        else:
            centers = all_centers

        center_rows = centers[:, 0]
        center_cols = centers[:, 1]

        accumulate(noisy_pad, hash_pad, clean,
                   center_rows, center_cols,
                   Q, V, counts,
                   weighting, args, strength_thresholds)

        completed_images.add(image_path)

    t_accum = time.time() - t_start
    print(f"\nAccumulation done in {t_accum:.1f}s")
    print(f"Total samples (approx): {counts.sum():,}")
    sys.stdout.flush()

    # Save raw (pre-augmentation) statistics
    _save_stats(Q, V, counts, completed_images, args, sigma_f,
                strength_thresholds, coherence_thresholds, suffix="")

    # Eight-fold augmentation
    print("\nApplying eight-fold statistical augmentation ...")
    t_aug = time.time()
    Q_aug, V_aug, counts_aug = augment_statistics(Q, V, counts)
    print(f"  done in {time.time()-t_aug:.1f}s  "
          f"(equiv. samples: {counts_aug.sum():,})")

    # Solve
    print("\nSolving filters ...")
    h, empty_buckets = solve_filters(Q_aug, V_aug, counts_aug, args.regularization)
    print(f"  empty/fallback buckets: {empty_buckets}/{N_FILTERS}")

    # Checks
    check_filters(h)

    # Save filter model
    model = {
        "task":              "denoise",
        "format_version":   1,
        "patchsize":        PATCHSIZE,
        "gradientsize":     GRADSIZE,
        "Qangle":           QANGLE,
        "Qstrength":        QSTRENGTH,
        "Qcoherence":       QCOHERENCE,
        "hash_mode":        args.hash_mode,
        "guide_sigma":      args.guide_sigma if args.hash_mode == "guided" else None,
        "strength_thresholds":  list(strength_thresholds),
        "coherence_thresholds": list(coherence_thresholds),
        "calib_method":     (None if args.no_calibrate else args.calib_method),
        "sigma":            args.sigma,
        "sigma_float":      sigma_f,
        "clip_noise":       False,
        "gray_conversion":  "BGR->YCrCb[:,0]/255",
        "padding":          "reflect",
        "regularization":   args.regularization,
        "augmentation":     "8fold_matrix",
        "n_raw_samples":    int(counts.sum()),
        "n_equiv_samples":  int(counts_aug.sum()),
        "n_images":         n_images,
        "empty_buckets":    empty_buckets,
        "h":                h,
    }
    filter_path = os.path.join(args.output_dir,
                               f"denoise_sigma{int(args.sigma)}_filter.p")
    with open(filter_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nFilter model saved to {filter_path}")

    # Plot filters
    if args.plot:
        # filterplot expects h with shape (Qangle, Qstrength, Qcoherence, R*R, patch*patch)
        # For denoising there is no pixel-type dimension; wrap with a length-1 axis.
        h_plot = h[:, :, :, np.newaxis, :]   # (24,3,3,1,121)
        filterplot(h_plot, R=1, Qangle=QANGLE, Qstrength=QSTRENGTH,
                   Qcoherence=QCOHERENCE, patchsize=PATCHSIZE)


def _save_stats(Q, V, counts, completed_images, args, sigma_f,
                strength_thresholds, coherence_thresholds, suffix=""):
    """Write Q/V/counts to disk using a tmp-then-rename pattern."""
    base = os.path.join(args.output_dir, f"denoise_sigma{int(args.sigma)}")
    meta = {
        "completed_images":    list(completed_images),
        "sigma":               args.sigma,
        "sigma_float":         sigma_f,
        "hash_mode":           args.hash_mode,
        "strength_thresholds": list(strength_thresholds),
        "coherence_thresholds": list(coherence_thresholds),
        "augmented":           False,
    }
    for name, payload in [
        (f"{base}_q{suffix}.p",      {**meta, "Q": Q, "counts": counts}),
        (f"{base}_v{suffix}.p",      {**meta, "V": V}),
    ]:
        tmp = name + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, name)


if __name__ == "__main__":
    main()
