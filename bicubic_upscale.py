import cv2
import numpy as np
import os
import sys
from scipy import interpolate


def bicubic_grid_interpolate(values, x_coords, y_coords):
    """Bicubic interpolation on a regular 2D grid."""
    y_points = np.asarray(y_coords)
    x_points = np.asarray(x_coords)
    y_grid = np.linspace(0, values.shape[0] - 1, values.shape[0])
    x_grid = np.linspace(0, values.shape[1] - 1, values.shape[1])
    interpolator = interpolate.RegularGridInterpolator(
        (y_grid, x_grid),
        values,
        method='cubic',
        bounds_error=False,
        fill_value=None,
    )
    yy, xx = np.meshgrid(y_points, x_points, indexing='ij')
    pts = np.stack([yy.ravel(), xx.ravel()], axis=1)
    return interpolator(pts).reshape(len(y_points), len(x_points))


def upscale_ycrcb_bicubic(image):
    """
    Upscale an image by 2x using YCrCb color space with bicubic interpolation.
    - Convert BGR -> YCrCb
    - Bicubic upscale each channel
    - Convert back to BGR
    """
    R = 2
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype('float')

    heightLR, widthLR = ycrcb.shape[:2]
    heightgridHR = np.linspace(0, heightLR - 0.5, heightLR * R)
    widthgridHR = np.linspace(0, widthLR - 0.5, widthLR * R)

    result = np.zeros((heightLR * R, widthLR * R, 3))
    for ch in range(3):
        result[:, :, ch] = bicubic_grid_interpolate(ycrcb[:, :, ch], widthgridHR, heightgridHR)

    result = np.clip(result, 0, 255)
    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2BGR)
    return result


def main():
    testpath = 'test'
    outputpath = 'bicubic_results'
    os.makedirs(outputpath, exist_ok=True)

    imagelist = []
    for parent, dirnames, filenames in os.walk(testpath):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg',
                                          '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))

    if not imagelist:
        print('No images found in', testpath)
        return

    for idx, image in enumerate(imagelist):
        basename = os.path.splitext(os.path.basename(image))[0]
        print(f'\rProcessing image {idx + 1}/{len(imagelist)}: {image}')
        sys.stdout.flush()

        origin = cv2.imread(image)
        if origin is None:
            print(f'  Warning: cannot read {image}, skipping.')
            continue

        print('  Bicubic upscaling...')
        bicubic_result = upscale_ycrcb_bicubic(origin)

        cv2.imwrite(os.path.join(outputpath, f'{basename}_bicubic.bmp'), bicubic_result)
        print(f'  Saved to {outputpath}/{basename}_bicubic.bmp')

    print('\rDone.')


if __name__ == '__main__':
    main()