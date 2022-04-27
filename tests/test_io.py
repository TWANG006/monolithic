"""Tests for the `io` sub-module."""

from monolithic.io import read_zygo_binary, read_zygo_dat, read_zygo_datx


def test_read_zygo_binary():
    """Test the `read_zygo_binary` function."""
    X, Y, Z, Xca, Yca, Zca = read_zygo_binary('./data/zygo_test.dat')
    assert X is not None and Y is not None and Z is not None
    assert Xca.shape == Yca.shape == Zca.shape

    X, Y, Z, Xca, Yca, Zca = read_zygo_binary('./data/zygo_test.datx')
    assert X is None and Y is None and Z is None
    assert Xca.shape == Yca.shape == Zca.shape


def test_read_datx():
    """Test the `read_zygo_datx` function."""
    h = read_zygo_datx('./data/zygo_test.datx')

    assert h['intensity'] is None
    assert 'phase' in h
    assert 'meta' in h
    assert 'lateral_res' in h['meta']


def test_read_dat():
    """Test the `read_zygo_dat` function."""
    h = read_zygo_dat('./data/zygo_test.dat')

    assert h['intensity'] is not None
    assert 'phase' in h
    assert 'meta' in h
    assert 'lateral_res' in h['meta']
