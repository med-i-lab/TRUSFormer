import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter
from scipy import signal
from skimage.transform import resize
from scipy.interpolate import interp1d


def get_phase(x, axis=-1):
    analytic_rep = signal.hilbert(x, axis=axis)
    return np.arctan2(np.imag(analytic_rep), np.real(analytic_rep))


def get_envelope(x, axis=-1):
    return np.abs(signal.hilbert(x, axis=axis))


def shift_phase_by_constant_value(x, theta, axis=-1):
    """Imparts a constant phase shift to the signal which maintains
    the same envelope."""

    shift = np.exp(1j * 2 * np.pi * theta)
    analytic_rep = signal.hilbert(x, axis=axis)
    analytic_rep *= shift
    return np.real(analytic_rep)


def noise(
    shape,
    strength=0.1,
    freq_limit=0.5,
):
    noise = np.random.randn(*shape)
    b, a = signal.butter(3, freq_limit, "low")
    noise = signal.lfilter(b, a, noise)
    return noise * strength  # type:ignore


def phase_distort(x, strength=0.1, freq_limit=0.5, constant_along_lateral_axis=True):
    analytic_rep = signal.hilbert(x, axis=0)
    noise_shape = (x.shape[0],) if constant_along_lateral_axis else x.shape
    _noise = noise(noise_shape, strength=strength, freq_limit=freq_limit)
    if x.ndim == 2 and constant_along_lateral_axis:
        _noise = np.expand_dims(_noise, -1)
    distortion = np.exp(2 * np.pi * 1j * _noise)
    analytic_rep *= distortion
    return np.real(analytic_rep)


def envelope_distort(x, strength=0.1, freq_limit=0.5, constant_along_lateral_axis=True):
    analytic_rep = signal.hilbert(x, axis=0)
    noise_shape = (x.shape[0],) if constant_along_lateral_axis else x.shape
    _noise = noise(noise_shape, strength=strength, freq_limit=freq_limit)
    if x.ndim == 2 and constant_along_lateral_axis:
        _noise = np.expand_dims(_noise, -1)
    distortion = 1 + _noise
    analytic_rep *= distortion
    return np.real(analytic_rep)


def bandstop(x, critical_freqs, filter_degree=3):
    b, a = signal.butter(filter_degree, critical_freqs, "stop")
    filtered = signal.lfilter(b, a, x, axis=0)
    return filtered


def random_bandstop(x, width=0.1, filter_degree=3):
    if filter_degree != 3:
        from warnings import warn

        warn(
            "risk of numerical instability with larger filter size. only 3 has been tested."
        )

    low = np.random.rand() + 0.01
    high = min(low + width, 0.99)

    return bandstop(x, (low, high), filter_degree=filter_degree)


def freq_stretch(x, factor=0.99, shift=0.0):
    envelope = get_envelope(x, axis=0)
    phase = x / envelope
    old_x = np.arange(0, len(x))

    endpoint = (len(x) - 1) * factor
    diff = (len(x) - 1) - endpoint

    new_x = np.linspace(0, endpoint, len(x), endpoint=True)
    new_x += shift * diff

    f = interp1d(old_x, phase, axis=0)
    new_phase = f(new_x)

    return envelope * new_phase


def random_freq_stretch(x, _range=(0.98, 1.0)):
    rng = np.random.rand()
    factor = _range[0] + rng * (_range[1] - _range[0])
    shift = np.random.rand()
    return freq_stretch(x, factor, shift)


def show_augmentations(patch, augmented, resize_to=(256, 256)):

    if resize_to is not None:
        patch = resize(patch, resize_to)
        augmented = resize(augmented, resize_to)

    vmin = min(np.min(augmented), np.min(patch))  # type:ignore
    vmax = max(np.max(augmented), np.max(patch))  # type:ignore

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(augmented, aspect="auto", vmin=vmin, vmax=vmax)
    ax[0].set_title("augmented")
    ax[1].imshow(patch, aspect="auto", vmin=vmin, vmax=vmax)
    ax[1].set_title("original")

    fig, ax = plt.subplots(1, 1)
    ax.plot(patch[:100, 0], label="original")
    ax.plot(augmented[:100, 0], label="augmented")
    ax.set_title("Single RF line")
    ax.legend()

    fig.tight_layout()


class UltrasoundArrayAugmentation:
    def __init__(
        self,
        random_phase_shift=False,
        random_phase_distort=False,
        random_phase_distort_strength=0.1,
        random_phase_distort_freq_limit=0.3,
        random_envelope_distort=False,
        random_envelope_distort_strength=0.2,
        random_envelope_distort_freq_limit=0.1,
        random_bandstop=False,
        random_bandstop_width=0.1,
        random_freq_stretch=False,
        random_freq_stretch_range=(0.98, 1.0),
    ):

        self.random_phase_shift = random_phase_shift
        self.random_phase_distort = random_phase_distort
        self.random_phase_distort_strength = random_phase_distort_strength
        self.random_phase_distort_freq_limit = random_phase_distort_freq_limit
        self.random_envelope_distort = random_envelope_distort
        self.random_envelope_distort_strength = random_envelope_distort_strength
        self.random_envelope_distort_freq_limit = random_envelope_distort_freq_limit
        self.random_bandstop = random_bandstop
        self.random_bandstop_width = random_bandstop_width
        self.random_freq_stretch = random_freq_stretch
        self.random_freq_stretch_range = random_freq_stretch_range

    def __call__(self, x):

        if self.random_phase_shift:
            shift = np.random.rand()
            x = shift_phase_by_constant_value(x, shift, axis=0)

        if self.random_phase_distort:
            x = phase_distort(
                x,
                strength=self.random_phase_distort_strength,
                freq_limit=self.random_phase_distort_freq_limit,
                constant_along_lateral_axis=True,
            )

        if self.random_envelope_distort:
            x = envelope_distort(
                x,
                strength=self.random_envelope_distort_strength,
                freq_limit=self.random_envelope_distort_freq_limit,
                constant_along_lateral_axis=True,
            )

        if self.random_bandstop:
            x = random_bandstop(x, width=self.random_bandstop_width, filter_degree=3)

        if self.random_freq_stretch:
            x = random_freq_stretch(x, _range=self.random_freq_stretch_range)

        return x
