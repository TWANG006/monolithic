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
            Z_res (numpy.ndarray): the surface-removed height map.
            Z_fit (numpy.ndarray): the removed surface.
            fit_func (function): the fitting function f(X, Y).
    """
    # only fit the valid entries in Z
    id = np.isfinite(Z)
    x = X[id].reshape(-1, 1)
    y = Y[id].reshape(-1, 1)
    z = Z[id].reshape(-1, 1)

    # build the design matrix for the fitting
    H = np.column_stack((np.ones_like(x), x, y))

    # solve
    coeffs, _, _, _ = lstsq(H, z, check_finite=False)

    # buid the output
    def fit_func(X, Y):
        return coeffs[0] + coeffs[1] * X + coeffs[2] * Y

    Z_fit = fit_func(X, Y)
    Zres = Z - Z_fit

    return Zres, Z_fit, fit_func


def remove_polynomials(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, p: int = 1):
    """Fit and remove an p-th order polynomial from the surface height map.

    Args:
        X (numpy.ndarray): x coordinates.
        Y (numpy.ndarray): y coordinates.
        Z (numpy.ndarray): surface heights.
        p (int): the highest order of the polynomial.

    Returns:
        (tutple): tutple containing
            Z_res (numpy.ndarray): the surface-removed height map.
            Z_fit (numpy.ndarray): the removed surface.
            fit_func (function): the fitting function f(X, Y).
    """
    # only fit the valid entries in Z
    id = np.isfinite(Z)
    x = X[id]
    y = Y[id]
    z = Z[id]
    H = np.ones(shape=(z.size, (p + 1) * (p + 2) // 2))

    # build the design matrix for the fitting
    k = 0
    for s in range(p + 1):
        for a in range(s, -1, -1):
            b = s - a
            H[:, k] = x**a * y**b
            k += 1

    # solve
    coeffs, _, _, _ = lstsq(H, z, check_finite=False)

    # fitting
    def fit_func(X, Y):
        Zf = 0
        k = 0
        for s in range(p + 1):
            for a in range(s, -1, -1):
                b = s - a
                Zf += coeffs[k] * X**a * Y**b
                k += 1
        return Zf

    Z_fit = fit_func(X, Y)
    Z_res = Z - Z_fit

    return Z_res, Z_fit, fit_func
