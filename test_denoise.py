"""RAISR denoising inference.

Applies learned per-bucket Wiener filters to a noisy input image.
Usage:
    python test_denoise.py --help
"""
import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np

from gaussian2d import gaussian2d
from hashkey import hashkey
from train_denoise import (PATCHSIZE, GRADSIZE, default_guide_sigma,
                           default_nlm_h, make_guide, print_progress)


# ---------------------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------------------
PATCHMARGIN = PATCHSIZE // 2   # 5
GRADMARGIN  = GRADSIZE  // 2   # 4


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Apply RAISR denoising filters")
    p.add_argument("-f", "--filter", required=True,
                   help="Path to denoise_filter.p model file")
    p.add_argument("--input",        default="D:\\raisr\\p0_samples\\noisy\\noisy_25",
                   help="Input image path (noisy) or .npy float file")
    p.add_argument("--reference",    default=None,
                   help="Optional clean reference image for PSNR/SSIM")
    p.add_argument("--output",       default="D:\\raisr\\p0_samples\\results",
                   help="Output path for denoised image (PNG). "
                        "Default: <input>_denoised.png next to input.")
    p.add_argument("-p", "--plot",   action="store_true",
                   help="Show four-panel comparison figure")
    p.add_argument("--blend-width",  type=float, default=0.0,
                   help="Soft-blend width as a fraction of the bucket span "
                        "(0=hard switch, 0.3=recommended starting point). "
                        "Linearly interpolates filter weights near strength "
                        "and coherence thresholds.")
    p.add_argument("--share-flat",   action="store_true",
                   help="In low-coherence (flat/isotropic) regions, average "
                        "the 24 direction filters instead of picking one. "
                        "Reduces direction-bucket switching artefacts in "
                        "featureless areas.")
    p.add_argument("--sigma",        type=float, default=None,
                   help="Noise sigma of the input in 8-bit gray-level units. "
                        "Selects (or interpolates) the matching filter bank "
                        "among the model's noise levels. Required unless "
                        "--blind.")
    p.add_argument("--blind",        action="store_true",
                   help="Estimate the input noise level per image (median "
                        "Immerkaer with clipping compensation; wavelet MAD and "
                        "textbook Immerkaer printed as cross-checks) and select "
                        "the matching filter bank. Enables blind denoising.")
    p.add_argument("--interp-noise",
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Blend filter coefficients between the two noise "
                        "levels bracketing the selected sigma instead of hard "
                        "nearest-level selection. Bucketing still uses the "
                        "nearest level's thresholds. On by default: on the "
                        "acceptance set this was worth +0.9 dB at sigma=20 and "
                        "+1.3 dB at sigma=35 versus hard selection. Use "
                        "--no-interp-noise for the hard nearest-level baseline.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Noise estimation and noise-level selection
# ---------------------------------------------------------------------------
def estimate_sigma_mad(noisy):
    """Wavelet-MAD noise estimate (skimage), in 8-bit gray-level units.

    Falls back to a Haar-wavelet MAD when PyWavelets is unavailable: the
    median absolute deviation of the diagonal detail coefficients divided by
    0.6745 (Donoho-Johnstone), which is robust to sparse edges.
    """
    try:
        from skimage.restoration import estimate_sigma
        try:
            s = estimate_sigma(noisy, channel_axis=None)
        except TypeError:   # older skimage: multichannel instead of channel_axis
            s = estimate_sigma(noisy, multichannel=False)
        return float(s) * 255.0
    except ImportError:
        h, w = noisy.shape
        h2, w2 = h - h % 2, w - w % 2
        d = (noisy[:h2:2, :w2:2] - noisy[:h2:2, 1:w2:2]
             - noisy[1:h2:2, :w2:2] + noisy[1:h2:2, 1:w2:2])
        # std(d) = 2*sigma (four +-1 taps), MAD of a Gaussian = 0.6745*std.
        return float(np.median(np.abs(d)) / (0.6745 * 2.0)) * 255.0


def estimate_sigma_immerkaer(noisy):
    """Immerkaer (1996) Laplacian noise estimate, in 8-bit gray-level units.

    sigma_hat = sqrt(pi/2) / 6 * mean(|I * L|), L the 3x3 Laplacian kernel.
    Fast cross-check for the wavelet-MAD estimate; overestimates on textured
    images because edges leak into the mean of the high-pass response.
    """
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(_lap_response(noisy)))
                 / 6.0) * 255.0


def _lap_response(noisy):
    """3x3 valid response of L = [[1,-2,1],[-2,4,-2],[1,-2,1]] (std 6*sigma)."""
    return (noisy[:-2, :-2] - 2.0 * noisy[:-2, 1:-1] + noisy[:-2, 2:]
            - 2.0 * noisy[1:-1, :-2] + 4.0 * noisy[1:-1, 1:-1] - 2.0 * noisy[1:-1, 2:]
            + noisy[2:, :-2] - 2.0 * noisy[2:, 1:-1] + noisy[2:, 2:])


def clip_rail_fraction(noisy):
    """Fraction of samples pinned at the display rails (0 or 1).

    Returns 0 for unclipped float data (values strictly outside [0, 1]) so the
    compensation below only fires on quantised/clipped 8-bit inputs.
    """
    if float(noisy.min()) < 0.0 or float(noisy.max()) > 1.0:
        return 0.0
    return float(np.mean((noisy <= 0.0) | (noisy >= 1.0)))


# Variance-loss coefficient for the rail-occupancy compensation below,
# calibrated on 250 crops of runs/eval/sigmahat (see verify_sigmahat.py).
CLIP_COMP_COEF = 1.5


def estimate_sigma_robust(noisy):
    """Primary blind estimator: median Immerkaer + clipping compensation.

    Two systematic errors dominate a plain high-pass MAD on 8-bit inputs:

    - Texture leaks into the high-pass response and inflates the *mean*, so a
      median replaces it (std of the 3x3 Laplacian response is 6*sigma).
    - Dynamic-range clipping destroys real noise variance at high sigma (at
      sigma=50 a plain estimate runs ~10% low). The lost variance grows with
      the fraction of rail-pinned samples, so scale it back by
      1/sqrt(1 - CLIP_COMP_COEF * q). Unclipped float data gives q = 0.
    """
    resp = _lap_response(noisy)
    s0 = float(np.median(np.abs(resp)) / (0.6745 * 6.0)) * 255.0
    q = clip_rail_fraction(noisy)
    return s0 / np.sqrt(max(1e-3, 1.0 - CLIP_COMP_COEF * q))


def resolve_noise_bank(levels, sigma, interp=False):
    """Map a (possibly continuous) sigma onto the model's noise levels.

    Returns (lo, hi, t, near): the effective filter bank is
    (1-t)*h[lo] + t*h[hi]; `near` is the level whose thresholds are
    authoritative for bucketing (a discrete decision - only the filter
    coefficients blend across levels). With interp=False, lo == hi == near
    and t == 0, i.e. hard nearest-level selection. Levels may be unsorted.
    """
    arr = np.asarray(levels, dtype=np.float64)
    order = np.argsort(arr)              # ascending, maps to original indices
    sarr = arr[order]
    near = int(order[int(np.argmin(np.abs(sarr - sigma)))])
    if not interp or arr.size == 1:
        return near, near, 0.0, near
    if sigma <= sarr[0]:
        lo = hi = int(order[0])
        return lo, hi, 0.0, near
    if sigma >= sarr[-1]:
        lo = hi = int(order[-1])
        return lo, hi, 0.0, near
    k = int(np.searchsorted(sarr, sigma))
    lo = int(order[k - 1])
    hi = int(order[k])
    t = float((sigma - sarr[k - 1]) / (sarr[k] - sarr[k - 1]))
    return lo, hi, t, near

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_gray_float(path):
    """Load an image as float64 Y-channel in [0,1], or a .npy file directly."""
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D float array in {path}, got shape {arr.shape}")
        return arr.astype(np.float64)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Cannot read image: {path}")
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0].astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def denoise(noisy, model, blend_width=0.0, blend_flat=False,
            h_filters=None, strength_thresholds=None,
            coherence_thresholds=None, nlm_h=None, guide_sigma=None):
    """Apply RAISR denoising filters to a float64 grayscale image.

    Parameters
    ----------
    noisy : ndarray, shape (H, W), float64
    model : dict loaded from filter pickle
    h_filters / strength_thresholds / coherence_thresholds / nlm_h /
        guide_sigma : optional overrides for multi-noise models. Callers pick
        (or blend) the noise-level bank first and inject the level's values
        here; when omitted the model's own (first-level) values are used.

    Returns
    -------
    denoised : ndarray, shape (H, W), float64  (unclipped)
    """

    Qangle     = model["Qangle"]
    hash_mode  = model["hash_mode"]
    strength_thresholds  = tuple(strength_thresholds)
    coherence_thresholds = tuple(coherence_thresholds)

    patchsize  = model["patchsize"]
    gradsize   = model["gradientsize"]
    pmargin    = patchsize  // 2
    gmargin    = gradsize   // 2

    H, W = noisy.shape

    # Lightweight preprocessing for the hash (must match training exactly)
    hash_src = make_guide(noisy, hash_mode, guide_sigma, nlm_h)

    # Reflect-pad
    noisy_pad = np.pad(noisy,    pmargin, mode='reflect')
    hash_pad  = np.pad(hash_src, gmargin, mode='reflect')

    # Weighting matrix
    weighting = gaussian2d([gradsize, gradsize], 2)
    weighting = np.diag(weighting.ravel())

    denoised = np.zeros((H, W), dtype=np.float64)
    total = H * W
    done = 0

    # Precompute angle-averaged filters for the share-flat mode (per strength/coherence).
    # These are used when the local coherence is below the low threshold and
    # the user requested --share-flat; computing them once avoids per-pixel overhead.
    Qangle = model["Qangle"]
    if blend_flat:
        flat_filters = h_filters.mean(axis=0)   # (3,3,121): averaged over angle dim
    else:
        flat_filters = None

    s_lo, s_hi = strength_thresholds
    c_lo, c_hi = coherence_thresholds
    s_span = max(s_hi - s_lo, 1e-12)
    c_span = max(c_hi - c_lo, 1e-12)

    from math import atan2, floor, pi as _pi
    import numpy.linalg as _la

    for row in range(H):
        for col in range(W):
            if done % 1000 == 0:
                print_progress(done, total)

            # patch from noisy
            rp, cp = row + pmargin, col + pmargin
            patch = noisy_pad[rp - pmargin: rp + pmargin + 1,
                              cp - pmargin: cp + pmargin + 1].ravel()

            # gradient block from hash source
            rg, cg = row + gmargin, col + gmargin
            gblock = hash_pad[rg - gmargin: rg + gmargin + 1,
                              cg - gmargin: cg + gmargin + 1]

            if blend_width > 0.0 or blend_flat:
                # Need continuous lambda and u values for blending decisions.
                gy, gx = np.gradient(gblock)
                G = np.vstack((gx.ravel(), gy.ravel())).T
                GTWG = G.T.dot(weighting).dot(G)
                eigs = np.real(_la.eigvals(GTWG))
                eigs = np.sort(eigs)[::-1]
                lamda = float(eigs[0])
                sqrtl1 = float(np.sqrt(max(eigs[0], 0.0)))
                sqrtl2 = float(np.sqrt(max(eigs[1], 0.0)))
                u = ((sqrtl1 - sqrtl2) / (sqrtl1 + sqrtl2)
                     if sqrtl1 + sqrtl2 > 0 else 0.0)
                theta = atan2(float(G[:, 1].sum()), float(G[:, 0].sum()))
                if theta < 0:
                    theta += _pi
                if theta >= _pi:
                    theta = _pi * (1.0 - 1e-10)
                angle = min(int(floor(theta / _pi * Qangle)), Qangle - 1)

                # Quantise for the primary bucket
                strength  = 0 if lamda < s_lo else (2 if lamda > s_hi else 1)
                coherence = 0 if u < c_lo else (2 if u > c_hi else 1)

                # --- share-flat: replace direction with angle-averaged filter ---
                if blend_flat and u < c_lo:
                    filt = flat_filters[strength, coherence]
                    denoised[row, col] = np.dot(filt, patch)
                    done += 1
                    continue

                # --- soft blend across strength threshold ---
                filt = h_filters[angle, strength, coherence].copy()
                if blend_width > 0.0:
                    half = blend_width * s_span
                    # Near low threshold (strength 0 <-> 1)
                    if strength == 0 and lamda > s_lo - half:
                        t = (lamda - (s_lo - half)) / (2 * half)
                        t = max(0.0, min(1.0, t))
                        filt = ((1 - t) * h_filters[angle, 0, coherence]
                                + t       * h_filters[angle, 1, coherence])
                    elif strength == 1 and lamda < s_lo + half:
                        t = (lamda - (s_lo - half)) / (2 * half)
                        t = max(0.0, min(1.0, t))
                        filt = ((1 - t) * h_filters[angle, 0, coherence]
                                + t       * h_filters[angle, 1, coherence])
                    # Near high threshold (strength 1 <-> 2)
                    elif strength == 1 and lamda > s_hi - half:
                        t = (lamda - (s_hi - half)) / (2 * half)
                        t = max(0.0, min(1.0, t))
                        filt = ((1 - t) * h_filters[angle, 1, coherence]
                                + t       * h_filters[angle, 2, coherence])
                    elif strength == 2 and lamda < s_hi + half:
                        t = (lamda - (s_hi - half)) / (2 * half)
                        t = max(0.0, min(1.0, t))
                        filt = ((1 - t) * h_filters[angle, 1, coherence]
                                + t       * h_filters[angle, 2, coherence])

                    # Near coherence low threshold (0 <-> 1)
                    half_c = blend_width * c_span
                    if coherence == 0 and u > c_lo - half_c:
                        tc = (u - (c_lo - half_c)) / (2 * half_c)
                        tc = max(0.0, min(1.0, tc))
                        filt = ((1 - tc) * h_filters[angle, strength, 0]
                                + tc       * h_filters[angle, strength, 1])
                    elif coherence == 1 and u < c_lo + half_c:
                        tc = (u - (c_lo - half_c)) / (2 * half_c)
                        tc = max(0.0, min(1.0, tc))
                        filt = ((1 - tc) * h_filters[angle, strength, 0]
                                + tc       * h_filters[angle, strength, 1])

                denoised[row, col] = np.dot(filt, patch)
            else:
                # Fast path: hard bucket selection, no blending
                angle, strength, coherence = hashkey(gblock, Qangle, weighting,
                                                     mode='denoise',
                                                     strength_thresholds=strength_thresholds,
                                                     coherence_thresholds=coherence_thresholds)
                filt = h_filters[angle, strength, coherence]
                denoised[row, col] = np.dot(filt, patch)
            done += 1
    return denoised


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def psnr(ref, pred, data_range=1.0):
    mse = np.mean((ref - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def ssim(ref, pred, data_range=1.0):
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(ref, pred, data_range=data_range)
    except ImportError:
        return float('nan')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    # Load model
    print(f"Loading model: {args.filter}")
    with open(args.filter, "rb") as f:
        model = pickle.load(f)

    if model.get("task") != "denoise":
        raise ValueError("Model was not trained for denoising (task != 'denoise')")

    qnoise = model["Qnoise"]
    levels = model["noise_levels"]
    print(f"  noise levels={','.join(f'{s:g}' for s in levels)} "
          f"hash_mode={model['hash_mode']}, "
          f"patchsize={model['patchsize']}")

    # Noise-level selection mode
    if args.sigma is None and not args.blind:
        raise SystemExit("pass --sigma (known noise level) or --blind (estimate it)")

    # Load noisy input. For a regular image also keep the original Cr/Cb
    # channels so the denoised Y can be recombined into a color output that
    # matches the input; a .npy float input carries luminance only.
    print(f"Loading input: {args.input}")
    imagelist = []
    for parent, dirnames, filenames in os.walk(args.input):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
    clean = []
    if args.reference:
        print(f"Loading reference: {args.reference}")
        for parent, dirnames, filenames in os.walk(args.reference):
            for filename in filenames:
                if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    clean.append(os.path.join(parent, filename))

    os.makedirs(args.output, exist_ok=True)
    imagecount = 1
    for image in imagelist:
        print('\r', end='')
        print(' ' * 60, end='')
        print('\rDenoising image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
        bgr = cv2.imread(image, cv2.IMREAD_COLOR)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        noisy = ycrcb[:, :, 0].astype(np.float64) / 255.0
        crcb = ycrcb[:, :, 1:3].copy()   # (H, W, 2) uint8: Cr, Cb
        H, W = noisy.shape

        # --- pick the noise-level bank for this image ---
        if args.blind:
            sigma_sel = estimate_sigma_robust(noisy)
            sigma_mad = estimate_sigma_mad(noisy)
            sigma_imm = estimate_sigma_immerkaer(noisy)
            print(f"  sigma_hat: robust={sigma_sel:.2f}  mad={sigma_mad:.2f}  "
                  f"immerkaer={sigma_imm:.2f}  rail_frac="
                  f"{clip_rail_fraction(noisy) * 100.0:.2f}%")
        else:
            sigma_sel = args.sigma

        lo, hi, t, near = resolve_noise_bank(levels, sigma_sel,
                                             interp=args.interp_noise)
        nlm_by = model.get("nlm_h_by_level", [default_nlm_h(s) for s in levels])
        gs_by  = model.get("guide_sigma_by_level",[default_guide_sigma(s) for s in levels])
        if hi != lo and t > 0.0:
            h_bank = ((1.0 - t) * model["h"][lo] + t * model["h"][hi])
            # Guide strengths follow the same blend as the filter bank; under
            # the default scaling rules this is exactly 0.8*sigma_sel (nlm) and
            # 0.8*sigma_sel/25 (gaussian), i.e. continuous in sigma.
            nlm_h_sel      = (1.0 - t) * nlm_by[lo] + t * nlm_by[hi]
            guide_sigma_sel = (1.0 - t) * gs_by[lo]  + t * gs_by[hi]
            blend_msg = f"interp {levels[lo]:g}->{levels[hi]:g} t={t:.2f}"
        else:
            # Hard selection uses the bank's trained values so the hash source
            # matches training exactly.
            h_bank = model["h"][near]
            nlm_h_sel      = nlm_by[near] if nlm_by is not None else default_nlm_h(levels[near])
            guide_sigma_sel = gs_by[near] if gs_by is not None else default_guide_sigma(levels[near])
            blend_msg = f"bank {near} (sigma={levels[near]:g})"
        s_thr = model["strength_thresholds_by_level"][near]
        c_thr = model["coherence_thresholds_by_level"][near]
        print(f"  noise select: sigma={sigma_sel:.2f} -> {blend_msg}")

        denoised = denoise(noisy, model,
                        blend_width=args.blend_width,
                        blend_flat=args.share_flat,
                        h_filters=h_bank,
                        strength_thresholds=s_thr,
                        coherence_thresholds=c_thr,
                        nlm_h=nlm_h_sel,
                        guide_sigma=guide_sigma_sel)

        # Determine output path
        out_path = args.output + '/' + os.path.splitext(os.path.basename(image))[0] + '_result.png'
        # Save denoisy result
        if crcb is not None:
            merged = np.zeros((H, W, 3), dtype=np.uint8)
            merged[:, :, 0] = (denoised * 255).clip(0, 255).astype(np.uint8)
            merged[:, :, 1] = crcb[:, :, 0]
            merged[:, :, 2] = crcb[:, :, 1]
            color_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
            cv2.imwrite(out_path, color_bgr)
        else:
            uint8 = (denoised * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(out_path, uint8)

        # Metrics
        clean_ref = None
        if clean:
            clean_ref = load_gray_float(clean[imagecount - 1])
            if clean_ref.shape != noisy.shape:
                raise ValueError(f"Reference shape {clean_ref.shape} != input shape {noisy.shape}")    
            denoised_clipped = denoised.clip(0, 1)
            noisy_clipped    = noisy.clip(0, 1)
            p_noisy    = psnr(clean_ref, noisy_clipped)
            p_denoised = psnr(clean_ref, denoised_clipped)
            s_noisy    = ssim(clean_ref, noisy_clipped)
            s_denoised = ssim(clean_ref, denoised_clipped)
            print(f"\nMetrics (data_range=1, clipped output):")
            print(f"  Noisy    PSNR={p_noisy:.2f} dB   SSIM={s_noisy:.4f}")
            print(f"  Denoised PSNR={p_denoised:.2f} dB   SSIM={s_denoised:.4f}")

        # Four-panel plot
        if args.plot:
            import matplotlib.pyplot as plt
            if clean_ref is not None:
                imgs   = [clean_ref, noisy, denoised.clip(0,1),
                        np.abs(clean_ref - denoised.clip(0,1))]
                titles = ["Clean", "Noisy", "Denoised", "Absolute error"]
            else:
                guide = make_guide(noisy,
                                   model["hash_mode"],
                                   guide_sigma_sel,
                                   nlm_h_sel)
                residual = noisy - denoised.clip(0, 1)
                imgs   = [noisy, guide, denoised.clip(0,1), residual]
                titles = ["Noisy", "Guide", "Denoised", "Noisy - Denoised (residual)"]

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for ax, img, title in zip(axes, imgs, titles):
                vmin = 0 if "error" not in title.lower() and "residual" not in title.lower() else None
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=1 if vmin == 0 else None,
                        interpolation='nearest')
                ax.set_title(title)
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        imagecount += 1


if __name__ == "__main__":
    main()
