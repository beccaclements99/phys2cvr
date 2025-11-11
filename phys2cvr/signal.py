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

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spint
import scipy.stats as sct
from scipy.signal import butter, filtfilt

from phys2cvr.io import FIGSIZE, SET_DPI, load_array

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


def spc(ts):
    """
    Compute signal percentage change of ts.

    Those ts that have mean 0 are divided by 1 instead.

    Parameters
    ----------
    ts : numpy.ndarray
        A timeseries or a set of timeseries - it is assumed that the last dimension is time.

    Returns
    -------
    numpy.ndarray
        The SPC version of ts.
    """
    m = ts.mean(axis=-1)[..., np.newaxis]
    md = deepcopy(m)
    md[md == 0] = 1
    ts = (ts - m) / md
    ts[np.isnan(ts)] = 0

    return ts


def create_hrf(freq=40):
    """
    Create a canonical haemodynamic response function sampled at the given frequency.

    Parameters
    ----------
    freq : float
        Sampling frequency of the haemodynamic response function.

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


def filter_signal(data, tr, lowcut=0.02, highcut=0.04, order=9):
    """
    Create a bandpass filter given a lowcut and a highcut, then filter data.

    Parameters
    ----------
    data : np.ndarray
        Data to filter (along last axis)
    tr : float
        TR of functional files
    lowcut : float
        Lower frequency in the bandpass
    highcut : float
        Higher frequency in the bandpass
    order : int
        The order of the butterworth filter

    Returns
    -------
    filt_data : np.ndarray
        Input `data`, but filtered.
    """
    nyq = (1 / tr) / 2
    low = lowcut / nyq
    high = highcut / nyq
    a, b = butter(int(order), [low, high], btype='band')
    filt_data = filtfilt(a, b, data, axis=-1)
    return filt_data


def compute_petco2hrf(co2, pidx, freq, outname, response_function='hfr', mode='full'):
    """
    Create PetCO2 trace from CO2 trace, then convolve to get PetCO2hrf.

    Parameters
    ----------
    co2 : np.ndarray
        CO2 (or physiological) regressor
    pidx : np.ndarray
        Index of peaks
    freq : str, int, or float
        Sample frequency of the CO2 regressor
    outname : str
        Prefix of the exported file
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
        Convolved CO2 trace.

    Raises
    ------
    NotImplementedError
        If the provided co2 is not a 1D array
        If the provided response function is not a supported option
    ValueError
        if the provided response function
    """
    co2 = co2.squeeze()
    if co2.ndim > 1:
        raise NotImplementedError('2+ D arrays are not supported.')

    if type(response_function) is str:
        if Path(response_function).exists():
            response_function = Path(response_function)
            LGR.debug(f'{response_function} found to be a valid file')
        else:
            response_function = response_function.lower()

    # Get response function
    if type(response_function) in [list, np.ndarray]:
        convolving_function = (
            np.array(response_function)
            if type(response_function) is list
            else response_function
        )
        if not np.issubdtype(convolving_function.dtype, np.number):
            raise ValueError
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

    nx = np.linspace(0, co2.size, co2.size)
    f = spint.interp1d(pidx, co2[pidx], fill_value='extrapolate')
    petco2 = f(nx)

    # Plot PETco2
    plt.figure(figsize=FIGSIZE, dpi=SET_DPI)
    plt.title('CO2 and PetCO2')
    plt.plot(co2, '-', petco2, '-')
    plt.legend(['CO2', 'PetCO2'])
    plt.tight_layout()
    plt.savefig(f'{outname}_petco2.png', dpi=SET_DPI)
    plt.close()

    # Demean and export
    petco2 = petco2 - petco2.mean()
    np.savetxt(f'{outname}_petco2.1D', petco2, fmt='%.18f')

    # Convolve, and then rescale to have same amplitude (?)
    petco2hrf = (
        np.convolve(petco2, convolving_function, mode=mode)
        if response_function is not None
        else petco2
    )
    petco2hrf = np.interp(
        petco2hrf, (petco2hrf.min(), petco2hrf.max()), (petco2.min(), petco2.max())
    )

    if response_function is not None:
        plt.figure(figsize=FIGSIZE, dpi=SET_DPI)
        plt.title('PetCO2 and convolved PetCO2 (PetCO2hrf)')
        plt.plot(petco2hrf, '-', petco2, '-')
        plt.tight_layout()
        plt.savefig(f'{outname}_petco2hrf.png', dpi=SET_DPI)
        plt.close()

        np.savetxt(f'{outname}_petco2hrf.1D', petco2hrf, fmt='%.18f')

    return petco2hrf


def resample_signal_samples(ts, samples, axis=-1):
    """
    Upsample or downsample a given timeseries based on samples.

    This program brings ts at freq1 to a new timeseries at freq2

    Parameters
    ----------
    ts : numpy.ndarray
        The timeseries to resample.
    samples : int
        The new desired amount od samples
    axis : int
        The axis over with the interpolation should happen - by default it's
        -1, i.e. the last dimension.

    Returns
    -------
    numpy.ndarray
        The resampled timeseries
    """
    # Upsample functional signal
    len_tp = ts.shape[axis]
    regr_t = np.linspace(0, len_tp - 1, samples)
    time_t = np.linspace(0, len_tp - 1, len_tp)
    f = spint.interp1d(time_t, ts, fill_value='extrapolate', axis=axis)

    return f(regr_t)


def resample_signal_freqs(ts, freq1, freq2, axis=-1):
    """
    Upsample or downsample a given timeseries based on frequencies.

    This program brings ts at freq1 to a new timeseries at freq2

    Parameters
    ----------
    ts : numpy.ndarray
        The timeseries to resample.
    freq1 : float
        The frequency of the timeseries to resample
    freq2 : float
        The new desired frequency
    axis : int
        The axis over with the interpolation should happen - by default it's
        -1, i.e. the last dimension.

    Returns
    -------
    numpy.ndarray
        The resampled timeseries
    """
    # Upsample functional signal
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
