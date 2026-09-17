import os
import sys
import cv2
import numpy as np
from scipy import interpolate

SUPPORTED_EXTS = (
    '.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'
)


def bilinear_grid_interpolate(values, x_coords, y_coords):
    """Bilinear interpolation on a regular 2D grid, compatible with SciPy >= 1.14."""
    y_points = np.asarray(y_coords)
    x_points = np.asarray(x_coords)
    y_grid = np.linspace(0, values.shape[0] - 1, values.shape[0])
    x_grid = np.linspace(0, values.shape[1] - 1, values.shape[1])
    interpolator = interpolate.RegularGridInterpolator(
        (y_grid, x_grid),
        values,
        method='linear',
        bounds_error=False,
        fill_value=None,
    )
    yy, xx = np.meshgrid(y_points, x_points, indexing='ij')
    pts = np.stack([yy.ravel(), xx.ravel()], axis=1)
    return interpolator(pts).reshape(len(y_points), len(x_points))


def upscale_ycrcb_bilinear(image):
    """
    Upscale an image by 2x using YCrCb color space with bilinear interpolation.
    - Convert BGR -> YCrCb
    - Bilinear upscale each channel
    - Convert back to BGR
    """
    R = 2
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype('float')

    heightLR, widthLR = ycrcb.shape[:2]
    heightgridHR = np.linspace(0, heightLR - 0.5, heightLR * R)
    widthgridHR = np.linspace(0, widthLR - 0.5, widthLR * R)

    result = np.zeros((heightLR * R, widthLR * R, 3))
    for ch in range(3):
        result[:, :, ch] = bilinear_grid_interpolate(ycrcb[:, :, ch], widthgridHR, heightgridHR)

    result = np.clip(result, 0, 255)
    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2BGR)
    return result


def list_images(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(SUPPORTED_EXTS):
                files.append(os.path.join(root, filename))
    return files


def main():
    input_dir = 'test'
    output_dir = 'bilinear_results'
    os.makedirs(output_dir, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        print(f'No images found in {input_dir}. Please put your test images there first.')
        return

    for idx, image_path in enumerate(images):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        print(f'\rProcessing image {idx + 1}/{len(images)}: {image_path}')
        sys.stdout.flush()

        img = cv2.imread(image_path)
        if img is None:
            print(f'  Warning: cannot read {image_path}, skipping.')
            continue

        print('  Bilinear upscaling...')
        result = upscale_ycrcb_bilinear(img)

        output_path = os.path.join(output_dir, f'{basename}_bilinear.bmp')
        cv2.imwrite(output_path, result)
        print(f'  Saved to {output_path}')

    print('\rDone.')


if __name__ == '__main__':
    main()
