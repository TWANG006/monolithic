# -*- coding: utf-8 -*-
"""Statistics of the general metrology data.

This module gives the statistical calculation of surface data, such as rms, rmse, pv, etc..

"""

import numpy as np


def rmse(x: np.ndarray) -> float:
    """Calculate the RMS error of all the element in the input data.

    Args:
        x (numpy.ndarray): the input `numpy` array of data.

    Returns:
        float: the RMS error of all the element in x.
    """
    return np.sqrt(np.nanmean(np.square(x.ravel() - np.nanmean(x.ravel()))))


def pv(x: np.ndarray) -> float:
    """Calculate the PV of all the elements in the input data.

    Args:
        x (numpy.ndarray): the input `numpy` array of data.

    Returns:
        float: the RMS error of all the elements in x.
    """
    return np.nanmax(x.ravel()) - np.nanmin(x.ravel())


def prr(x: np.ndarray) -> float:
    """Calculate the Peak Removal Rate of the TIF-like `x`.

    Args:
        x (numpy.ndarray): the input `numpy` array of data.

    Returns:
        float: the PRR of x.
    """
    return np.nanmax(x.ravel())


def vrr(z: np.ndarray, dx: float, dy: float) -> float:
    """Calculate the Volumetric Removal Rate of the TIF-like `x`.

    Args:
        z (numpy.ndarray): the input `numpy` array of data.
        dx (float): the sampling interval of `z` in x axis.
        dy (float): the sampling interval of `z` in y axis.

    Returns:
        float: the VRR of z.
    """
    return np.nansum(z.ravel()) * dx * dy
