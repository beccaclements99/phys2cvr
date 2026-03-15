#!/usr/bin/env python3
"""Tests for io."""

import os
from pathlib import PosixPath
from unittest.mock import patch

import numpy as np
import pytest

from phys2cvr import signal

# ## Unit tests


def test_spc():
    # Test case 1: test with a single time series
    ts = np.array([1, 2, 3, 4, 5])
    expected_result = np.array([-0.666666, -0.333333, 0, 0.333333, 0.666666])
    assert np.allclose(signal.spc(ts), expected_result)

    # Test case 2: test with multiple time series
    ts = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    expected_result = np.array(
        [
            [-0.666666, -0.333333, 0, 0.333333, 0.666666],
            [0.666666, 0.333333, 0, -0.333333, -0.666666],
        ]
    )
    assert np.allclose(signal.spc(ts), expected_result)

    # Test case 3: test with input ts having mean 0
    ts = np.array([[-3, 0, 3], [3, 0, -3]])
    expected_result = np.array([[-3, 0, 3], [3, 0, -3]])
    assert np.allclose(signal.spc(ts), expected_result)

    # Test case 4: test with input ts having nan values
    ts = np.array([[1, 2, np.nan], [-1, np.nan, -3]])
    expected_result = np.array([[0, 0, 0], [0, 0, 0]])
    assert np.allclose(signal.spc(ts), expected_result)


def test_create_hrf():
    # Test case 1: test with default frequency
    expected_length = 1281
    assert len(signal.create_hrf()) == expected_length

    # Test case 2: test with custom frequency
    freq = 100
    expected_length = 3201
    assert len(signal.create_hrf(freq)) == expected_length

    # Test case 4: test that the maximum value of HRF is 1
    assert signal.create_hrf().max() == 1

    # Test with custom frequency of 5
    freq = 5
    hrf = signal.create_hrf(freq)

    # Test that the length of the HRF is correct
    expected_length = 161
    assert len(hrf) == expected_length

    # Test that the first peak of the HRF is at the correct time point
    expected_peak_index = 25
    actual_peak_index = np.argmax(hrf)
    assert actual_peak_index == expected_peak_index


def test_filter_signal():
    x = np.sin(np.linspace(0, 10, 200))
    out = signal.filter_signal(x, tr=1.0)
    assert out.shape == x.shape

    x = np.random.randn(5, 200)
    out = signal.filter_signal(x, tr=1.0, axis=-1)
    assert out.shape == x.shape


def test_endtidal_interpolation():
    sig = np.array([0, 1, 0, 2, 0])
    peaks = [1, 3]
    out = signal.endtidal_interpolation(sig, peaks)
    assert out.shape == sig.shape
    assert np.allclose(out[[1, 3]], sig[[1, 3]])

    sig = np.arange(10.0)
    peaks = [8, 2, 5]
    out = signal.endtidal_interpolation(sig, peaks)
    assert out.shape == sig.shape
    assert np.allclose(out[[2, 5, 8]], sig[[2, 5, 8]])


def test_convolve_signal(testdir):
    x = np.ones(20)
    out = signal.convolve_signal(x, freq=10, response_function='hrf')
    assert out.ndim == 1
    assert out.shape[0] > x.shape[0]
    # Add assertion on first 20 points of hrf at 10 Hz)

    x = np.arange(10.0)
    out = signal.convolve_signal(x, freq=10, response_function=None)
    assert np.allclose(out, x)

    x = np.arange(5.0)
    rf = [1, 0, -1]
    out = signal.convolve_signal(x, freq=10, response_function=rf)
    assert out.ndim == 1

    p = os.path.join(testdir, 'rf.1D')
    np.savetxt(p, np.array([1.0, 2.0, 3.0]))
    x = np.ones(10)
    with patch('phys2cvr.io.load_array', return_value=np.array([1.0, 2.0, 3.0])):
        out = signal.convolve_signal(x, 10, response_function=str(p))
    assert out.ndim == 1


def test_resample_signal_samples_basic():
    x = np.arange(10.0)
    out = signal.resample_signal_samples(x, 20)
    assert out.shape[0] == 20

    x = np.random.randn(5, 10)
    out = signal.resample_signal_samples(x, 15, axis=-1)
    assert out.shape == (5, 15)


def test_resample_signal_freqs_basic():
    x = np.arange(10.0)
    out = signal.resample_signal_freqs(x, 1, 2)
    assert out.shape[0] > x.shape[0]

    x = np.random.randn(4, 10)
    out = signal.resample_signal_freqs(x, 1, 3, axis=-1)
    assert out.shape[1] > 10


# ## Break tests


def test_break_convolve_signal_invalid_array(testdir):
    x = np.arange(5.0)
    rf = ['a', 'b']
    with pytest.raises(ValueError):
        signal.convolve_signal(x, freq=10, response_function=rf)

    x = np.zeros((3, 3, 3))
    with pytest.raises(NotImplementedError):
        signal.convolve_signal(x, freq=10)

    p = os.path.join(testdir, 'does_not_exist.1D')
    x = np.ones(10)
    with pytest.raises(OSError):
        signal.convolve_signal(x, 10, response_function=PosixPath(p))

    p = os.path.join(testdir, 'does_not_exist.1D')
    x = np.ones(10)
    with pytest.raises(NotImplementedError):
        signal.convolve_signal(x, 10, response_function=str(p))

    x = np.ones(10)
    with pytest.raises(NotImplementedError):
        signal.convolve_signal(x, 10, response_function='not_a_rf')
