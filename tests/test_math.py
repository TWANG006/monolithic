"""Tests for the `math` sub-module."""

# import matplotlib.pyplot as plt
import numpy as np

from monolithic.io import read_zygo_binary
from monolithic.math import (
    fft_1d,
    fft_2d,
    fwhm_2_sigma,
    ifft_1d,
    ifft_2d,
    prr,
    psd_1d,
    pv,
    remove_polynomials,
    remove_surface,
    rmse,
    sigma_2_fwhm,
    vrr,
    window_function,
)


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


def test_psd_window_function():
    """Test the window function."""
    n = 5
    win_welch = np.array([0, 0.75, 1, 0.75, 0])
    win_hann = np.array([0, 0.5, 1, 0.5, 0])
    win_none = np.ones(n)

    win_welch_calc = window_function(n, 'welch')
    win_hann_calc = window_function(n, 'hann')
    win_none_calc = window_function(n, 'none')

    assert np.abs((win_welch - win_welch_calc)).all() <= 1e-7
    assert np.abs((win_hann - win_hann_calc)).all() <= 1e-7
    assert np.abs((win_none - win_none_calc)).all() <= 1e-7


def test_psd_1d():
    """Test the psd_1d function."""
    _, _, _, Xca, _, Zca = read_zygo_binary('./data/zygo_test.dat')
    pixel_size = np.median(np.diff(Xca[1, :]))
    q, cq_1d, int_cq_1d = psd_1d(Zca[300:401, 300:401], pixel_size, 'x', 'welch')

    # _, ax = plt.subplots(1, 2)
    # ax[0].loglog(q * 1e-3, cq_1d * 1e21)
    # ax[1].loglog(q * 1e-3, int_cq_1d * 1e9)
    # plt.show()


def test_remove_surface():
    """Test the `remove_surface` function."""
    X, Y, Z, _, _, _ = read_zygo_binary('./data/zygo_test.dat')
    (
        Z,
        _,
        _,
    ) = remove_surface(X, Y, Z)
    assert np.nanmean(Z) <= 1e-15


def test_remove_polynomials():
    """Test the `remove_polynomials` function."""
    X, Y, Z, _, _, _ = read_zygo_binary('./data/zygo_test.dat')
    (
        Z,
        _,
        _,
    ) = remove_polynomials(X, Y, Z, p=1)
    assert np.nanmean(Z) <= 1e-15
