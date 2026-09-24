import os
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--clean', default="D:\\raisr\\p0_samples\\clean", help='Path to the directory containing clean images')
parser.add_argument('--denoised', default="D:\\raisr\\p0_samples\\results", help='Path to the directory containing denoised images')
args = parser.parse_args()

def psnr(ref, pred, data_range=1.0):
    ref = ref.astype(np.float64)
    pred = pred.astype(np.float64)
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

# ============ 示例调用 ============
if __name__ == "__main__":
    gt_list = []
    denoised_list = []
    for parent, dirnames, filenames in os.walk(args.clean):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                gt_list.append(os.path.join(parent, filename))

    for parent, dirnames, filenames in os.walk(args.denoised):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                denoised_list.append(os.path.join(parent, filename))
    for gt_path, denoised_path in zip(gt_list, denoised_list):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        denoised = cv2.imread(denoised_path, cv2.IMREAD_GRAYSCALE)
        
        p = psnr(gt, denoised, data_range=255.0)
        s = ssim(gt, denoised, data_range=255.0)
        print(f"Comparing {os.path.basename(gt_path)} and {os.path.basename(denoised_path)}:")
        print(f"PSNR = {p:.4f} dB")
        print(f"SSIM = {s:.4f}")