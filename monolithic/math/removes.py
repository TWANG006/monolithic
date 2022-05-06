"""Implements the removal of 1D and 2D piston, tilt, power, etc."""

from typing import Tuple

import numpy as np
from scipy.linalg import lstsq


def remove_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple:
    """Fit and remove a surface from the input surface height map.

    Args:
        X (numpy.ndarray): x coordinates.
        Y (numpy.ndarray): y coordinates.
        Z (numpy.ndarray): surface heights.

    Returns:
        (tutple): tutple containing
            Z (numpy.ndarray): the surface-removed height map.
            Z_fit (numpy.ndarray): the removed surface.
            fit_func (function): the fitting function f(X, Y).
            coeffs (numpy.ndarray): the fitting coefficients.
    """
    # only fit the valid entries in Z
    id = np.isfinite(Z)
    x = X[id].reshape(-1, 1)
    y = Y[id].reshape(-1, 1)
    z = Z[id].reshape(-1, 1)

    # build the design matrix for the fitting
    H = np.column_stack((np.ones_like(x), x, y))
    coeffs, _, _, _ = lstsq(H, z, check_finite=False)

    # buid the output
    def fit_func(X, Y):
        return coeffs[0] + coeffs[1] * X + coeffs[2] * Y

    Z_fit = fit_func(X, Y)
    Z = Z - Z_fit

    return Z, Z_fit, fit_func
