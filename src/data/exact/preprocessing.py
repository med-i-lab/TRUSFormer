from math import floor
import einops
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import pkg_resources
import pandas as pd
from itertools import product
from skimage.transform import resize
from scipy.signal import lfilter


# ===================================================
# DEFAULT PARAMETERS TAKEN FROM THE MATLAB SCRIPT
# ===================================================

# Setup the Rx frequency
# from @moon matlab code

F0 = 17.5e6 * 2  # Rx Quad 2x
DEFAULT_RECONSTRUCTION_FREQ = F0 / 2

# Stitch params
DEFAULT_PARAMS = {
    "Depth": np.array([7, 16, 22]),
    "Boundaries": np.array([11.5000, 19]),
    "Corrections": np.array([8, 0.0400, 14, 0.0200, 17, 0.0600, 21, 0.0350]),
}

AXIAL_EXTENT = 28  # mm, length of image along axial dimention
LATERAL_EXTENT = 46.08  # mm, length of image along lateral dimention


# Interpolation filter
FILTER = [
    -2.2737367544323206e-13,
    -2.4312130025236911e-03,
    -4.5593193912054630e-03,
    -6.0867394810202313e-03,
    -6.7775138305705696e-03,
    -6.4858523272732782e-03,
    -5.1777598602029684e-03,
    -2.9436086272198736e-03,
    0.0000000000000000e00,
    1.3571640261943685e-02,
    2.5784400964312226e-02,
    3.4906633308764867e-02,
    3.9457774229958886e-02,
    3.8379988421638700e-02,
    3.1186609072392457e-02,
    1.8075864528327656e-02,
    -9.0949470177292824e-13,
    -4.4992860027377901e-02,
    -8.7701098444085801e-02,
    -1.2218482731623226e-01,
    -1.4265809342759894e-01,
    -1.4396752281299996e-01,
    -1.2204652839409391e-01,
    -7.4315408141046646e-02,
    0.0000000000000000e00,
    1.3696806975167419e-01,
    2.9100578523139120e-01,
    4.5221821498853387e-01,
    6.0983636066157487e-01,
    7.5295791229928000e-01,
    8.7130541983333387e-01,
    9.5595626632984931e-01,
    1.0000000000009095e00,
    9.5595626632984931e-01,
    8.7130541983333387e-01,
    7.5295791229928000e-01,
    6.0983636066157487e-01,
    4.5221821498853387e-01,
    2.9100578523139120e-01,
    1.3696806975167419e-01,
    0.0000000000000000e00,
    -7.4315408141046646e-02,
    -1.2204652839409391e-01,
    -1.4396752281299996e-01,
    -1.4265809342759894e-01,
    -1.2218482731623226e-01,
    -8.7701098444085801e-02,
    -4.4992860027377901e-02,
    -9.0949470177292824e-13,
    1.8075864528327656e-02,
    3.1186609072392457e-02,
    3.8379988421638700e-02,
    3.9457774229958886e-02,
    3.4906633308764867e-02,
    2.5784400964312226e-02,
    1.3571640261943685e-02,
    0.0000000000000000e00,
    -2.9436086272198736e-03,
    -5.1777598602029684e-03,
    -6.4858523272732782e-03,
    -6.7775138305705696e-03,
    -6.0867394810202313e-03,
    -4.5593193912054630e-03,
    -2.4312130025236911e-03,
    -2.2737367544323206e-13,
]

DELAY = 32


# def interp(ys, mul):
#    # linear extrapolation for last (mul - 1) points
#    ys = list(ys)
#    ys.append(2 * ys[-1] - ys[-2])
#    # make interpolation function
#    xs = np.arange(len(ys))
#    fn = interp1d(xs, ys, kind="cubic")
#    # call it on desired data points
#    new_xs = np.arange(len(ys) - 1, step=1.0 / mul)
#    return fn(new_xs)


def upsample(array, factor):
    """upsample by filling between with zeros in the 0'th dimension"""

    out_shape = list(array.shape)
    out_shape[0] *= factor

    out = np.zeros(out_shape)

    out[::factor] = array

    return out


def interpolate(array, factor=8):
    """interpolates the given array along axis 0"""

    if not factor == 8:
        raise NotImplementedError("Factors other than 8 are not supported")

    # upsample the array by adding zeros
    array = upsample(array, factor)

    # apply the interpolation filter
    array = lfilter(FILTER, 1, array, axis=0)

    # adjust for the delay
    array = array[DELAY:]

    return array


def iq_to_rf(Q, I):

    reconstruction_freq = DEFAULT_RECONSTRUCTION_FREQ

    fs = reconstruction_freq
    f_rf = fs  # reconstruct to the original Rx freq in the param file

    fs = fs * 2  # actual Rx freq is double because of Quad 2x
    IntFac = 8
    fs_int = fs * IntFac

    bmode_n_samples = Q.shape[0]

    interpolation_factor = 8

    t = np.arange(
        0, (bmode_n_samples * interpolation_factor) / fs_int, 1 / fs_int
    ).reshape(-1, *((1,) * (Q.ndim - 1)))

    t = t[:-DELAY]

    Q_interp = interpolate(Q, interpolation_factor)
    I_interp = interpolate(I, interpolation_factor)

    rf = np.real(
        np.sqrt(I_interp**2 + Q_interp**2)  # type:ignore
        * np.sin(2 * np.pi * f_rf * t + np.arctan2(Q_interp, I_interp))
    )

    return rf


def stitch_focal_zones(img, depth=30, offset=2, params=DEFAULT_PARAMS):

    imgout = np.zeros((img.shape[0], img.shape[1] // 3))
    bound1 = round((params["Boundaries"][0] - offset) / (depth - offset) * img.shape[0])
    bound2 = round((params["Boundaries"][1] - offset) / (depth - offset) * img.shape[0])

    imgout[: bound1 - 1, :] = img[: bound1 - 1, 0 : img.shape[1] : 3]
    imgout[bound1 : bound2 - 1, :] = img[bound1 : bound2 - 1, 1 : img.shape[1] : 3]
    imgout[bound2:, :] = img[bound2:, 2 : img.shape[1] : 3]

    gaincurve = np.zeros(img.shape[0])
    depthvals = np.round(
        (params["Corrections"][0::2] - offset) / (depth - offset) * img.shape[0]
    ).astype("int")
    corrvals = params["Corrections"][1::2]

    samples_per_mm = img.shape[0] / (depth - offset)

    gaincurve[depthvals[0] : bound1 + 1] = (
        np.arange(1, (bound1 - depthvals[0] + 1) + 1) * corrvals[0] / samples_per_mm
    )
    gaincurve[bound1 + 1 : depthvals[1] + 1] = (
        np.arange(1, (depthvals[1] - bound1) + 1) * corrvals[1] / samples_per_mm
    )
    gaincurve[depthvals[2] : bound2 + 1] = (
        np.arange(1, (bound2 - depthvals[2] + 1) + 1) * corrvals[2] / samples_per_mm
    )
    gaincurve[bound2 + 1 : depthvals[3] + 1] = (
        np.arange(1, depthvals[3] - bound2 + 1) * corrvals[3] / samples_per_mm
    )

    imgout = imgout * einops.repeat(
        10 ** (gaincurve / 20), "axial -> axial lateral", lateral=imgout.shape[1]
    )

    return imgout


def stack_focal_zones(img):
    return einops.rearrange(img, "axial (lateral zone) -> axial lateral zone", zone=3)


def to_bmode(rf):
    return np.log(1 + np.abs(hilbert(rf)))


def _sliding_window_stacked(img, axial_size, lateral_size, axial_step, lateral_step):
    """
    Returns an array of shape (N, H, W) corresponding to patches of data
    from the RF image obtained by taking a sliding window with specified
    width and step size in milimeters. The windows are stacked along the
    first dimension.

    img: the source image, dimensions (AXIAL_DIM, LATERAL_DIM)
    axial_size: the target size of the sliding window along the axial direction
        in milimeters
    lateral_size: the target size of the sliding window along the lateral direction
        in milimeters
    ...
    lateral_step: the target step size of the sliding window along the lateral direction
        in milimeters
    """

    axial_dim, lateral_dim = img.shape
    axial_pixel_spacing = AXIAL_EXTENT / axial_dim  # mm per pixel
    lateral_pixel_spacing = LATERAL_EXTENT / lateral_dim  # mm per pixel

    axial_window_npixels = floor(axial_size / axial_pixel_spacing)
    lateral_window_npixels = floor(lateral_size / lateral_pixel_spacing)

    axial_step_npixels = floor(axial_step / axial_pixel_spacing)
    lateral_step_npixels = floor(lateral_step / lateral_pixel_spacing)

    def windows_topleft(dims, sizes, steps):
        """
        Returns the top left (i, j) coord and corresponding grid index for all
        windows with size (sizes) that fit in the dimenstions
        (dims) when using a sliding window with step steps
        """
        i = 0
        while i + sizes[0] <= dims[0]:
            j = 0
            while j + sizes[1] <= dims[1]:
                yield i, j
                j += steps[1]
            i += steps[0]

    windows = []
    for i, j in windows_topleft(
        (axial_dim, lateral_dim),
        (axial_window_npixels, lateral_window_npixels),
        (axial_step_npixels, lateral_step_npixels),
    ):
        window = img[i : i + axial_window_npixels, j : j + lateral_window_npixels]
        windows.append(window)

    return np.stack(windows)


from itertools import product


def patch_size_mm_to_pixels(
    img_shape,
    patch_height_mm,
    patch_width_mm,
    img_extent_mm=(AXIAL_EXTENT, LATERAL_EXTENT),
):
    """returns the pixel sizes that would correspond to the corresponding size
    in milimeters for the given ultrasound image,

    Args:
        img (np.ndarray): the ultrasound image, covering the full extent of the tissue
        patch_height_mm (float): desired patch height
        patch_width_mm (float): desired patch width
    """

    dw = int(patch_width_mm / (img_extent_mm[1] / img_shape[1]))
    dh = int(patch_height_mm / (img_extent_mm[0] / img_shape[0]))

    return dh, dw


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


def get_positions_for_split_into_patches(
    img_shape, patch_height_pixels, patch_width_pixels, discard_from="bottomleft"
):

    dw = patch_width_pixels
    dh = patch_height_pixels

    w = int(img_shape[1] / dw)
    h = int(img_shape[0] / dh)

    grid = np.zeros((h, w, 4)).astype("int")

    origin = [0, 0]
    if "bottom" in discard_from:
        origin[0] = 0
    elif "top" in discard_from:
        origin[0] = img_shape[0] - (h * dh)
    if "right" in discard_from:
        origin[1] = 0
    elif "left" in discard_from:
        origin[1] = img_shape[1] - (w * dw)

    for i, j in product(range(h), range(w)):

        slice = (
            dh * i + origin[0],
            dh * i + dh + origin[0],
            dw * j + origin[1],
            dw * j + dw + origin[1],
        )

        grid[i, j] = slice

    return grid


def split_into_patches(img, width_mm, height_mm):
    """
    Splits into patches of size height_mm, width_mm.
    returns an array of shape (grid_h, grid_w, patch_h, patch_w ...)
    whose [i,j]th entry is the patch at position i, j
    """

    height_pixels, width_pixels = patch_size_mm_to_pixels(
        img.shape, height_mm, width_mm
    )

    return split_into_patches_pixels(img, height_pixels, width_pixels)


def downsample_axial(img, factor):
    return resize(img, (img.shape[0] // factor, img.shape[1]), anti_aliasing=True)


def DEFAULT_PREPROCESS_TRANSFORM(iq) -> np.ndarray:
    """Default preprocessing - turns iq to rf, selects last
    frame only, decimates the signal in the axial direction,
    and stitches focal zones if necessary
    """

    from .preprocessing import iq_to_rf, stitch_focal_zones
    from scipy.signal import decimate

    # first frame only
    rf = iq_to_rf(iq["Q"][..., 0], iq["I"][..., 0])

    if rf.shape[1] > 512:
        rf = stitch_focal_zones(rf)

    # decimation by factor of 4 does not lose frequency information
    rf = decimate(rf, 4, axis=0)

    return rf
