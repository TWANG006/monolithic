"""Tests for the `math` sub-module."""

import numpy as np

from monolithic.math import pv, rmse


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
