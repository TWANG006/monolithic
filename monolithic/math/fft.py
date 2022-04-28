"""This sub-module wrapps the pyFFTW rountines for computing FFT.

The pyfftw lib uses the FFTW library so it is much faster than both
numpy.fft and scipy.fft, especially when FFTW_ESTIMATE is used as
the computation method and no complicated planning is needed.
"""

import numpy as np
from pyfftw.builders import fft, fft2, ifft, ifft2


def fft_1d(a: np.ndarray, n: int = None, axis: int = -1):
    """Compute the 1-D discrete Fourier transform using the FFTW lib.

    Args:
        a (numpy.ndarray): input array, can be complex.
        n (int): length of the transformed axis of the output.
        axis (int): axis over which to compute the FFT. The last axis is used by default.

    Returns:
        complex numpy.ndarray: transformed input along the axis.
    """
    fft_obj = fft(a=a, n=n, axis=-1, planner_effort='FFTW_ESTIMATE')
    return fft_obj()


def ifft_1d(a: np.ndarray, n: int = None, axis: int = -1):
    """Compute the 1-D inverse discrete Fourier transform using the FFTW lib.

    Args:
        a (numpy.ndarray): input array, can be complex.
        n (int): length of the transformed axis of the output.
        axis (int): axis over which to compute the FFT. The last axis is used by default.

    Returns:
        complex numpy.ndarray: transformed input along the axis.
    """
    ifft_obj = ifft(a=a, n=n, axis=axis, planner_effort='FFTW_ESTIMATE')
    return ifft_obj()


def fft_2d(a: np.ndarray, s=None, axes=(-2, -1)):
    """Compute the 2-D discrete Fourier transform using the FFTW lib.

    Args:
        a (numpy.ndarray): input array, can be complex.
        s (sequence of ints): shape in each axis of the output.
        axes (sequence of ints): axes over which to compute the FFT.

    Returns:
        complex numpy.ndarray: transformed input along the axes.
    """
    fft_obj = fft2(a=a, s=s, axes=axes, planner_effort='FFTW_ESTIMATE')
    return fft_obj()


def ifft_2d(a: np.ndarray, s=None, axes=(-2, -1)):
    """Compute the 2-D inverse discrete Fourier transform using the FFTW lib.

    Args:
        a (numpy.ndarray): input array, can be complex.
        s (sequence of ints): shape in each axis of the output.
        axes (sequence of ints): axes over which to compute the FFT.

    Returns:
        complex numpy.ndarray: transformed input along the axes.
    """
    ifft_obj = ifft2(a=a, s=s, axes=axes, planner_effort='FFTW_ESTIMATE')
    return ifft_obj()
