#!/usr/bin/env python3
"""
Signal analysis module for phys2cvr.

Attributes
----------
LGR :
    Logger
"""

import logging
from copy import deepcopy
from pathlib import Path, PosixPath

import numpy as np
import scipy.interpolate as spint
import scipy.stats as sct
from scipy.signal import butter, filtfilt

from phys2cvr.io import load_array
from phys2cvr.viz import plot_two_timeseries

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


def spc(ts):
    """
    Compute signal percentage change over time series (ts).

    Timeseries are divided by the mean.
    Timeseries that have a mean of 0 are divided by 1 instead.

    Parameters
    ----------
    ts : numpy.ndarray
        A timeseries or set of timeseries; last dimension is assumed to be time.

    Returns
    -------
    numpy.ndarray
        Signal percentage change version of the input ts.
    """
    m = ts.mean(axis=-1)[..., np.newaxis]
    md = deepcopy(m)
    md[md == 0] = 1
    ts = (ts - m) / md
    ts[np.isnan(ts)] = 0
    return ts


def create_hrf(freq=40):
    """
    Create a canonical haemodynamic response function sampled at the given freq.

    Parameters
    ----------
    freq : float
        Sampling frequency used to resample the haemodynamic response function.

    Returns
    -------
    hrf : np.ndarray
        Haemodynamic response function.
    """
    # Create HRF
    RT = 1 / freq
    fMRI_T = 16
    p = [6, 16, 1, 1, 6, 0, 32]

    # Modelled hemodynamic response function - {mixture of Gammas}
    dt = RT / fMRI_T
    u = np.arange(0, p[6] / dt + 1, 1) - p[5] / dt

    a1 = p[0] / p[2]
    b1 = 1 / p[3]
    a2 = p[1] / p[3]
    b2 = 1 / p[3]

    hrf = (
        sct.gamma.pdf(u * dt, a1, scale=b1) - sct.gamma.pdf(u * dt, a2, scale=b2) / p[4]
    ) / dt

    time_axis = np.arange(0, int(p[6] / RT + 1), 1) * fMRI_T
    hrf = hrf[time_axis]

    min_hrf = 1e-9 * min(hrf[hrf > 10 * np.finfo(float).eps])
    if min_hrf < 10 * np.finfo(float).eps:
        min_hrf = 10 * np.finfo(float).eps

    hrf[hrf == 0] = min_hrf
    hrf = hrf / max(hrf)
    return hrf


def filter_signal(data, tr, lowcut=0.02, highcut=0.04, order=9, axis=-1):
    """
    Create a bandpass filter within lower and upper threshold, then filter.

    Parameters
    ----------
    data : np.ndarray
        Data to filter.
    tr : float
        TR of functional files.
    lowcut : float
        Low frequency threshold.
    highcut : float
        High frequency threshold.
    order : int
        Butterworth filter order.
    axis : int
        The axis along which the filter is applied.

    Returns
    -------
    filt_data : np.ndarray
        Bandpass-filtered data.
    """
    nyq = (1 / tr) / 2
    low = lowcut / nyq
    high = highcut / nyq
    a, b = butter(int(order), [low, high], btype='band')
    return filtfilt(a, b, data, axis=axis)


def compute_petco2hrf(
    co2, pidx, freq, outprefix, comp_endtidal=True, response_function='hfr', mode='full'
):
    """
    Create the PetCO2 trace from CO2 trace, then convolve it to obtain the PetCO2hrf.

    Parameters
    ----------
    co2 : np.ndarray
        CO2 (or physiological) regressor
    pidx : np.ndarray
        Indices of peaks
    freq : str, int, or float
        Sample frequency of the CO2 regressor
    outprefix : str
        Prefix of the output file (i.e., the regressor of interest).
    comp_endtidal : bool
        If True, interpolate input regressor
    response_function : {`hrf`, `rrf`, `crf`}, None, str, path, or 1D array-like , optional
        Name of the response function to be used in the convolution of the regressor of
        interest or path to a 1D file containing one. Default is `hrf`.
        For `rrf` and `crf`, `phys2denoise` must be installed (see extra installs).
            - `None` will skip the convolution
            - `hrf` will use an internally generated canonical HRF
            - `rrf` will use `phys2denoise`'s RRF
            - `crf` will use `phys2denoise`'s CRF
            - Alternatively, specify a valid path file containing a custom one.
            - Alternatively, specify a custom one directly as np.ndarray or list.
    mode : {'full', 'valid', 'same'} str, optional
        Convolution mode, see numpy.convolve.

    Returns
    -------
    petco2hrf : np.ndarray
        Convolved PetCO2 trace.

    Raises
    ------
    NotImplementedError
        If the provided CO2 is not a 1D array.
        If the provided response function is not a supported option.
    ValueError
        if the provided response function is not a numeric ndarray-like variable.
    """
    co2 = co2.squeeze()
    if co2.ndim > 1:
        raise NotImplementedError(
            'Arrays with more than 2 dimensions are not supported.'
        )

    if comp_endtidal:
        nx = np.linspace(0, co2.size, co2.size)
        f = spint.interp1d(pidx, co2[pidx], fill_value='extrapolate')
        petco2 = f(nx)

        # Plot PetCO2 vs CO2
        plot_two_timeseries(
            petco2, co2, f'{outprefix}_co2_vs_petco2.png', 'PetCO2', 'CO2', freq
        )

        # Demean and export
        petco2 = petco2 - petco2.mean()
        np.savetxt(f'{outprefix}_petco2.1D', petco2, fmt='%.18f')
    else:
        LGR.info(
            'Skipping End Tidal interpolation of PetCO2 trace (if you provided raw CO2 '
            'data, then you probably should not be doing this)'
        )
        petco2 = co2 - co2.mean()

    if type(response_function) is str:
        if Path(response_function).exists():
            response_function = Path(response_function)
            LGR.debug(f'{response_function} is a valid file')
        else:
            response_function = response_function.lower()
            response_function = (
                None if response_function == 'none' else response_function
            )

    if type(response_function) in [list, np.ndarray]:
        convolving_function = (
            np.array(response_function)
            if type(response_function) is list
            else response_function
        )
        if not np.issubdtype(convolving_function.dtype, np.number):
            raise ValueError(
                'Provided function is not a numeric ndarray-like variable.'
            )
    elif response_function is None:
        LGR.info(
            'Computing PetCO2 trace but skipping convolution with response function'
        )
    elif response_function == 'hrf':
        convolving_function = create_hrf(freq)
    elif response_function == 'rrf':
        try:
            from phys2denoise.metrics.responses import rrf
        except ImportError:
            raise ImportError(
                'phys2denoise is required for the use of RRF response functions. '
                'Please see install instructions.'
            )
        convolving_function = rrf
    elif response_function == 'crf':
        try:
            from phys2denoise.metrics.responses import crf
        except ImportError:
            raise ImportError(
                'phys2denoise is required for the use of CRF response functions. '
                'Please see install instructions.'
            )
        convolving_function = crf
    elif type(response_function) is PosixPath:
        if not Path(response_function).exists():
            raise OSError(f'{response_function} not found in system.')
        convolving_function = load_array(response_function)
    else:
        raise NotImplementedError(
            f'Response function "{response_function}" is not supported yet or file was '
            'not found.'
        )

    # Plot convolved PetCO2 vs PetCO2
    if response_function is not None:
        petco2hrf = np.convolve(petco2, convolving_function, mode=mode)
        petco2hrf = np.interp(
            petco2hrf, (petco2hrf.min(), petco2hrf.max()), (petco2.min(), petco2.max())
        )
        plot_two_timeseries(
            petco2hrf,
            petco2,
            f'{outprefix}_petco2_vs_petco2hrf.png',
            'Convolved PetCO2',
            'PetCO2',
            freq,
        )
    else:
        LGR.info('Skipping convolution of PetCO2 trace')
        petco2hrf = petco2

    return petco2hrf


def resample_signal_samples(ts, samples, axis=-1):
    """
    Resample timeseries based on desired number of samples.

    Parameters
    ----------
    ts : np.ndarray
        The timeseries to be resampled
    samples : int
        Desired number of samples.
    axis : int
        The axis (dimension) over which the interpolation should be applied - by default it's
        -1, i.e., the last dimension.

    Returns
    -------
    numpy.ndarray
        Resampled timeseries.
    """
    len_tp = ts.shape[axis]
    regr_t = np.linspace(0, len_tp - 1, samples)
    time_t = np.linspace(0, len_tp - 1, len_tp)
    f = spint.interp1d(time_t, ts, fill_value='extrapolate', axis=axis)
    return f(regr_t)


def resample_signal_freqs(ts, freq1, freq2, axis=-1):
    """
    Resample timeseries based on current and desired frequency.

    Parameters
    ----------
    ts : np.ndarray
        The timeseries to be resampled.
    freq1 : float
        Current frequency.
    freq2 : float
        Desired frequency.
    axis : int
        The axis (dimension) over which the interpolation should happen - by default it's
        -1, i.e. the last dimension.

    Returns
    -------
    numpy.ndarray
        Resampled timeseries.
    """
    len_tp = ts.shape[axis]
    len_s = (len_tp - 1) / freq1
    regr_t = np.linspace(0, len_s, int(len_s * freq2) + 1)
    time_t = np.linspace(0, len_s, len_tp)
    f = spint.interp1d(time_t, ts, fill_value='extrapolate', axis=axis)
    return f(regr_t)


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
