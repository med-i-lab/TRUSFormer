import numpy as np
from itertools import product


def split_into_patches_pixels(
    img, patch_height_pixels, patch_width_pixels, discard_from="bottomleft"
):
    """
    Splits into patches of size patch_height, patch_width
    returns an array of shape (grid_h, grid_w, patch_h, patch_w)
    whose [i,j]th entry is the patch at position i, j
    """

    dw = patch_width_pixels
    dh = patch_height_pixels

    w = int(img.shape[1] / dw)
    h = int(img.shape[0] / dh)

    grid = np.zeros((h, w, dh, dw, *(img.shape[2:]))).astype(img.dtype)

    origin = [0, 0]
    if "bottom" in discard_from:
        origin[0] = 0
    elif "top" in discard_from:
        origin[0] = img.shape[0] - (h * dh)
    if "right" in discard_from:
        origin[1] = 0
    elif "left" in discard_from:
        origin[1] = img.shape[1] - (w * dw)

    for i, j in product(range(h), range(w)):

        window = img[
            dh * i + origin[0] : dh * i + dh + origin[0],
            dw * j + origin[1] : dw * j + dw + origin[1],
        ]
        grid[i, j] = window

    return grid
