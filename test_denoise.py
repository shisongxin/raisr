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
from scipy.ndimage import gaussian_filter

from gaussian2d import gaussian2d
from hashkey import hashkey
from train_denoise import PATCHSIZE, GRADSIZE, print_progress


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
                   help="Path to denoise_sigma*_filter.p model file")
    p.add_argument("--input",        default="D:\\raisr\\p0_samples\\noisy",
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
    return p.parse_args()


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
def denoise(noisy, model, blend_width=0.0, blend_flat=False):
    """Apply RAISR denoising filters to a float64 grayscale image.

    Parameters
    ----------
    noisy : ndarray, shape (H, W), float64
    model : dict loaded from filter pickle

    Returns
    -------
    denoised : ndarray, shape (H, W), float64  (unclipped)
    """
    h_filters = model["h"]           # (24,3,3, 121)
    Qangle     = model["Qangle"]
    hash_mode  = model.get("hash_mode", "noisy")
    guide_sigma = model.get("guide_sigma") or 0.8
    # Use the thresholds the model was trained with; fall back to SR defaults
    # only for legacy models that predate calibration.
    strength_thresholds  = tuple(model.get("strength_thresholds",  (0.0001, 0.001)))
    coherence_thresholds = tuple(model.get("coherence_thresholds", (0.25, 0.5)))

    patchsize  = model["patchsize"]
    gradsize   = model["gradientsize"]
    pmargin    = patchsize  // 2
    gmargin    = gradsize   // 2

    H, W = noisy.shape

    # Build guide if needed
    if hash_mode == "guided":
        hash_src = gaussian_filter(noisy, sigma=guide_sigma, mode='reflect')
    else:
        hash_src = noisy

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

    print(f"  sigma={model['sigma']}, hash_mode={model.get('hash_mode','noisy')}, "
          f"patchsize={model['patchsize']}")

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

        # Load clean reference if provided
        
        denoised = denoise(noisy, model,
                        blend_width=args.blend_width,
                        blend_flat=args.share_flat)

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
                guide_sigma = model.get("guide_sigma") or 0.8
                from scipy.ndimage import gaussian_filter as _gf
                guide = _gf(noisy, sigma=guide_sigma, mode='reflect')
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
