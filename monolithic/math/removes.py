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

    return Zres, Z_fit, fit_func, coeffs


def remove_polynomials(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, p: int = 1) -> Tuple:
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

    return Z_res, Z_fit, fit_func, coeffs


def remove_sphere(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple:
    """Fit and remove a sphere from the input surface height map.

    Args:
        X (numpy.ndarray): x coordinates.
        Y (numpy.ndarray): y coordinates.
        Z (numpy.ndarray): surface heights.

    Returns:
        (tutple): tutple containing
            Z_res (numpy.ndarray): the sphere-removed height map.
            Z_fit (numpy.ndarray): the removed sphere.
            fit_func (function): the fitting function f(X, Y).
    """
    # only fit the valid entries in Z
    id = np.isfinite(Z)
    x = X[id].reshape(-1, 1)
    y = Y[id].reshape(-1, 1)
    z = Z[id].reshape(-1, 1)

    # build the design matrix for the fitting
    H = np.column_stack((np.ones_like(x), x, y, x * x, y * y))

    # solve
    coeffs, _, _, _ = lstsq(H, z, check_finite=False)

    # buid the output
    def fit_func(X, Y):
        return coeffs[0] + coeffs[1] * X + coeffs[2] * Y + coeffs[3] * X * X + coeffs[3] * Y * Y

    Z_fit = fit_func(X, Y)
    Zres = Z - Z_fit

    return Zres, Z_fit, fit_func, coeffs


def remove_tilt(x: np.ndarray, z: np.ndarray):
    """Removes a 2D tilt from the surface profile.

    Args:
        x (numpy.ndarray): x coordinates.
        z (numpy.ndarray): height profile.

    Returns:
        (tuple): tuple containing:
            z_res (nupmy.ndarray): tilt-removed height profile.
            z_fit (numpy.ndarray): fitted height profile.
            fit_func (function): the fitting function.
            coeffs (numpy.ndarray): fitting coefficients.
    """
    # remove the invalid entries
    id = np.isfinite(z)
    z_to_fit = z[id].reshape(-1, 1)
    x_to_fit = x[id].reshape(-1, 1)

    # build the matrix
    H = np.column_stack((np.ones_like(x_to_fit), x_to_fit))

    # solv the linear system
    coeffs, _, _, _ = lstsq(H, z_to_fit, check_finite=False)

    def fit_func(x):
        return coeffs[0] + coeffs[1] * x

    z_fit = fit_func(x)
    z_res = z - z_fit

    return z_res, z_fit, fit_func, coeffs
