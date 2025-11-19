#!/usr/bin/env python3
"""
Plotting module for phys2cvr.

Attributes
----------
FIGSIZE : tuple
    Figure size
SET_DPI : int
    DPI of the figure
LGR :
    Logger
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore as zs

SET_DPI = 100
FIGSIZE = (18, 10)

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


def _time_axis(array, freq=None):
    """
    Prepare time axis for the plots.

    Parameters
    ----------
    freq : None, optional
        Sampling frequency of array. If specified, plot against time.

    Returns
    -------
    time_axis : np.ndarray
        Time in seconds if freq is not None or number of samples if it is.
    """
    if freq is not None:
        # Preparing time axis for plots
        return np.linspace(0, (len(array) - 1) / freq, len(array))
    else:
        return np.arange(0, array.size)


def plot_two_timeseries(
    ts1, ts2, outname, ts1_name=None, ts2_name=None, freq=None, zscore=False
):
    """
    Plot two timeseries.

    Parameters
    ----------
    ts1 : np.ndarray-like
        First timeseries to be plotted.
    ts2 : np.ndarray-like
        Second tiemseries to be plotted.
    outname : str or path
        Filename (and path to) to save the image.
    ts1_name : str or None, optional
        Name of the first timeseries, for legend and title.
    ts2_name : str or None, optional
        Name of the second timeseries, for legend and title.
    freq : None, optional
        Frequency of Xcorr samples. If specified, it's used to plot against time.
    zscore : bool, optional
        If True, zscore timeseries before plot. Default is False.
    """
    ts1 = zs(ts1) if zscore else ts1
    ts2 = zs(ts2) if zscore else ts2

    plt.figure(figsize=FIGSIZE, dpi=SET_DPI)
    time_axis = _time_axis(ts1, freq)
    plt.plot(time_axis, ts1, '-')
    time_axis = _time_axis(ts2, freq)
    plt.plot(time_axis, ts2, '-')
    plt.title(f'{ts1_name} vs {ts2_name}')
    plt.legend([ts1_name, ts2_name])
    plt.tight_layout()
    plt.savefig(outname, dpi=SET_DPI)
    plt.close()


def plot_xcorr(xcorr, outprefix, freq=None):
    """
    Plot Cross Correlation max and abs max values.

    Parameters
    ----------
    xcorr : np.ndarray-like
        Cross correlation array.
    outprefix : str or path
        Prefix of filename (and path to) to save the image.
    freq : None, optional
        Frequency of Xcorr samples. If specified, it's used to plot against time.
    """
    time_axis = _time_axis(xcorr, freq)

    plt.figure(figsize=FIGSIZE, dpi=SET_DPI)
    plt.plot(time_axis, xcorr)
    plt.plot(time_axis[xcorr.argmax()], xcorr[xcorr.argmax()], 'd')
    plt.plot(time_axis[np.abs(xcorr).argmax()], xcorr[np.abs(xcorr).argmax()], 'x')
    plt.legend(['Cross correlation value', 'Max Xcorr', 'Max absolute Xcorr'])
    plt.title('Cross correlation and optimal shift')
    plt.tight_layout()
    plt.savefig(f'{outprefix}_optshift.png', dpi=SET_DPI)
    plt.close()


"""
Copyright 2021, Stefano Moia.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
