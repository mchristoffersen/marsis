"""Utility functions"""

import numpy as np


def arangeT(tstart, tstop, fs):
    """Generate a uniformly sampled timeseries.

    Generate a uniformly sampled timeseries from tstart to tstop
    sampled at frequency fs.

    Parameters
    ----------
    tstart: float
        Starting time
    path: str
        Path to directory where the MARSIS data files will be downloaded to

    Returns
    -------
    seq: array_like of double
        Array containing timeseries values.

    Notes
    -----
    The timeseries generated is a half open interval - tstart is included but
    never tstop.
    """

    tTot = tstop - tstart
    nsamp = int(tTot * fs)
    seq = np.arange(nsamp, dtype=np.double)
    seq = seq / fs + tstart
    return seq


def refChirp(f0=-0.5e6, f1=0.5e6, m=4.0e9, fs=1.4e6):
    """Generate a chirp with given parameters.

    Generate a linear frequency sweep (chirp) with the given frequency
    bound and sweep rate, sampled at a given frequency

    Parameters
    ----------
    f0: float
        Starting frequency of the chirp
    f1: float
        Final frequency of the chirp
    m: float
        Rate of frequency sweep, a positive value is an upsweep and a
        negative value is a downsweep
    fs: float
        Sampling rate of chirp

    Returns
    -------
    chirp: array_like of float
        Tuple containing the sampled chirp amplitude values.

    Notes
    -----
    The time length of the chirp is calculated from the input parameters,
    t = (f1-f0)/m. This function generates a signal with unit amplitude.
    """
    # TODO: Add windowing

    tlen = (f1 - f0) / m
    t = arangeT(0, tlen, fs)  # time vector

    if len(t) == 0:  # In case m is very quick, spit out one sample
        t = np.array([1.0 / fs])

    phi = np.pi * (2 * f0 * t + m * (t ** 2))  # phase
    chirp = -1 * np.sin(phi) + 1j * np.cos(phi)  # chirp
    return chirp
