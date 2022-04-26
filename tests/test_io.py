"""Tests for the `io` sub-module."""

from monolithic.io import read_zygo_dat, read_zygo_datx


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
