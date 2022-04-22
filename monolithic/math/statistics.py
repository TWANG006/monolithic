# -*- coding: utf-8 -*-
"""Statistics of the general metrology data.

This module gives the statistical calculation of surface data, such as rms, rmse, pv, etc..

"""

import numpy as np


def rmse(x: np.ndarray) -> float:
    """Calculate the RMS error of all the element in the input data.

    Args:
        x (numpy.ndarray): the input `numpy` array of data

    Returns:
        float: the RMS error of all the element in x

    """
    return np.sqrt(np.nanmean(np.square(x.ravel() - np.nanmean(x.ravel()))))


def pv(x: np.ndarray) -> float:
    """Calculate the PV of all the elements in the input data.

    Args:
        x (numpy.ndarray): the input `numpy` array of data

    Returns:
        float: the RMS error of all the elements in x

    """
    return np.nanmax(x.ravel()) - np.nanmin(x.ravel())
