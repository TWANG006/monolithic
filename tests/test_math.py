"""Tests for the `math` sub-module."""

import numpy as np

from monolithic.math import fft_1d, fft_2d, fwhm_2_sigma, ifft_1d, ifft_2d, prr, pv, rmse, sigma_2_fwhm, vrr


def test_rmse():
    """Test the rmse function."""
    a_test = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    assert a_test.shape == (2, 5)
    assert np.round(rmse(a_test), 2) == 2.87


def test_pv():
    """Test the pv function."""
    a_test = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    assert a_test.shape == (2, 5)
    assert pv(a_test) == 9


def test_prr():
    """Test the prr function."""
    a_test = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    assert prr(a_test) == 10


def test_vrr():
    """Test the vrr function."""
    a_test = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    assert vrr(a_test, 0.1, 0.1) == 0.55


def test_fwhm_2_sigma():
    """Test the fwhm_2_sigma function."""
    fwhm = np.array([5, 5])
    assert np.round(fwhm_2_sigma(fwhm), 2)[1] == 2.12


def test_sigma_2_fwhm():
    """Test the sigma_2_fwhm function."""
    sigma = np.array([2.12, 2.12])
    assert np.round(sigma_2_fwhm(sigma))[1] == 5


def test_ftt_ifft_1d():
    """Test the forward and inverse Fourier transform."""
    a_test = np.array([1, 2, 3, 4, 5])
    assert (a_test - ifft_1d(fft_1d(a_test))).all() <= 1e-7


def test_ftt_ifft_2d():
    """Test the forward and inverse Fourier transform."""
    a_test = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    assert (a_test - np.real(ifft_2d(fft_2d(a_test)))).all() <= 1e-7
