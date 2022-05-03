"""Implement the 1D and 2D Power Spectral Density (PSD) for a surface height profile.

Examples:
    q, c = psd_1d(z, pixel_width, 'x')
    q, c = psd_1d(z, pixel_width, 'y')
    q, c, _ = psd_2d(z, pixel_width)

References:
    [1] Jacobs, T. D., Junge, T., & Pastewka, L. (2017). Metrology and Properties, 5(1), 013001.
    [2] Mona Mahboob Kanafi (2022). 1-dimensional surface roughness power spectrum of a profile or topography
    [3] Mona Mahboob Kanafi (2022). Radially averaged surface roughness/topography power spectrum (PSD)

"""

from typing import Tuple

import numpy as np


def window_function(n: int, win_type: str = 'hann') -> np.ndarray:
    """Generate a window function (row vector) in the range of [0, n  - 1].

    Args:
        n (int): number of elements for the window
        win_type (str): type of the window function. Supported functions are 'none', 'hann' and 'welch' for now.

    Returns:
        numpy.ndarray: the window funtion containing n elements.
    """
    # make sure n is non-negative
    if n < 0:
        raise ValueError(f'The number of elements cannot be {n}.')

    # validate the window function type
    win_type = win_type.lower()
    if win_type == 'hann':
        return np.hanning(n)
    elif win_type == 'welch':
        return 1 - ((np.arange(0, n) - (n - 1) * 0.5) / ((n - 1) * 0.5)) ** 2
    elif win_type == 'none':
        return np.ones(n)
    else:
        raise ValueError(f'Invalude window type {win_type}.')


def psd_1d(z: np.ndarray, pixel_size: float, dim: str = 'x', win_type: str = 'welch') -> Tuple:
    """Calculates the averaged 1D PSD along `x` or `y` direction or simply for multiple 1D line profiles.

    Args:
        z (numpy.ndarray): height topography, SI units, i.e., meters or 1D profile(s).
        pixel_size (float): size of each pixel, SI units, i.e., meters.
        dim (str): `x` or `y` dimension, where `x` takes z[:, 1: n] and `y` takes z[1: m, :].
        win_type (str): window function, can be 'welch', 'hann', or 'none'.

    Returns:
        (tuple): tuple containing
            q (numpy.ndarray): wavevectors, which is 1/lambda, where lambda is the wavelength.
            c_1d (numpy.ndarray): averaged 1D PSD profile.
            int_c_1d (numpy.ndarray): integration of the averaged 1D PSD profile.
    """
    pass
    # # convert z to be at least 2D
    # z_2d = np.atleast_2d(z)

    # # deal with dim
    # dim = dim.lower()
    # if dim == 'x':
    #     z_2d = z_2d
    # elif dim == 'y':
    #     z_2d = z_2d.T
    # else:
    #     raise ValueError(f'Dimension {dim} is invalide.')
    # m, n = z_2d.shape

    # # generate a window function
    # win = np.ones((m, 1)) * np.atleast_2d(n, win_type.lower())
