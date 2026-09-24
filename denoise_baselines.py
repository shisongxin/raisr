"""Baseline denoising: bilateral filter and Non-Local Means (NLM).

Reads the noisy images in p0_samples/noisy, denoises each with both methods,
and writes the results to p0_samples/bilateral and p0_samples/nlm.

The noisy PNGs are color images with AWGN added on the Y channel only, so each
method denoises the Y channel and recombines it with the original Cr/Cb — the
same convention as test_denoise.py — to keep the output comparable.

Usage:
    python denoise_baselines.py                 # sigma=25 defaults
    python denoise_baselines.py --sigma 25
"""
import argparse
import os

import cv2
import numpy as np

from train_denoise import PATCHSIZE

NOISY_DIR     = os.path.join("p0_samples", "noisy")
BILATERAL_DIR = os.path.join("p0_samples", "bilateral")
NLM_DIR       = os.path.join("p0_samples", "nlm")
IMG_EXTS      = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def get_args():
    p = argparse.ArgumentParser(description="Bilateral / NLM baseline denoising")
    p.add_argument("--sigma", type=float, default=25.0,
                   help="Noise sigma in [0,255] scale, used to set filter "
                        "strengths (default 25).")
    p.add_argument("--input-dir",     default=NOISY_DIR)
    p.add_argument("--bilateral-dir", default=BILATERAL_DIR)
    p.add_argument("--nlm-dir",       default=NLM_DIR)
    return p.parse_args()


def bilateral_y(y_u8, sigma_noise):
    """Bilateral filter on a uint8 Y channel.

    d           : neighbourhood diameter, tied to RAINR's PATCHSIZE so both
                  methods draw on the same-size local neighbourhood
    sigmaColor  : intensity window, scaled to the noise level (in [0,255])
    sigmaSpace  : spatial extent, matched to d
    """
    d = PATCHSIZE
    sigma_color = float(2.5 * sigma_noise)   # [0,255] scale
    sigma_space = float(d)
    return cv2.bilateralFilter(y_u8, d, sigma_color, sigma_space)


def nlm_y(y_u8, sigma_noise):
    """Non-Local Means on a uint8 Y channel.

    h controls filter strength; ~1.1x sigma is a good starting point.
    templateWindowSize is tied to RAISR's PATCHSIZE (same local neighbourhood
    used per pixel); searchWindowSize follows the classic NLM default since
    RAISR has no equivalent non-local search parameter to match.
    """
    h = float(1.1 * sigma_noise)
    return cv2.fastNlMeansDenoising(y_u8, None, h=h,
                                    templateWindowSize=PATCHSIZE,
                                    searchWindowSize=21)


def process_one(path, sigma, bilat_dir, nlm_dir):
    """Denoise one color noisy image with both methods and save results."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"  [WARN] cannot read {path}")
        return
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]

    base = os.path.splitext(os.path.basename(path))[0]

    for method_fn, out_dir in [(bilateral_y, bilat_dir), (nlm_y, nlm_dir)]:
        y_denoised = method_fn(y, sigma)
        merged = ycrcb.copy()
        merged[:, :, 0] = y_denoised
        out_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(os.path.join(out_dir, base + "_result.png"), out_bgr)


def main():
    args = get_args()
    os.makedirs(args.bilateral_dir, exist_ok=True)
    os.makedirs(args.nlm_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.input_dir)
                   if f.lower().endswith(IMG_EXTS))
    if not files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Denoising {len(files)} images (sigma={args.sigma})")
    print(f"  bilateral -> {args.bilateral_dir}")
    print(f"  NLM       -> {args.nlm_dir}")
    for i, fn in enumerate(files, 1):
        path = os.path.join(args.input_dir, fn)
        print(f"  [{i}/{len(files)}] {fn}")
        process_one(path, args.sigma, args.bilateral_dir, args.nlm_dir)
    print("Done.")


if __name__ == "__main__":
    main()
