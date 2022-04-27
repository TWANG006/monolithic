"""Utility functions that may be generally used across the package."""

import numpy as np
import numpy.typing as npt


def fwhm_2_sigma(fwhm: npt.ArrayLike) -> npt.ArrayLike:
    """Convert the Full Width at Half Magnitude (FWHM) to simga.

    Args:
        fwhm (array_like): the input FWHM value.

    Returns:
        array_like: the simga converted from FWHM.
    """
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def sigma_2_fwhm(sigma: npt.ArrayLike) -> npt.ArrayLike:
    """Convert the simga of a Gaussian to FWHM.

    Args:
        fwhm (array_like): the input sigma value.

    Returns:
        array_like: the FWHM converted from sigma.
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma
