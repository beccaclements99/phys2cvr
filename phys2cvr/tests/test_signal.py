#!/usr/bin/env python3
"""Tests for io."""

import os

import numpy as np

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
