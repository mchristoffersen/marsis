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

    phi = np.pi * (2 * f0 * t + m * (t**2))  # phase
    chirp = -1 * np.sin(phi) + 1j * np.cos(phi)  # chirp
    return chirp


def quadMixShift(x, fShift=-0.7e6, fs=1.4e6):
    """Frequency shift a signal through quadrature mixing.

    Parameters
    ----------
    x: array_like of complex float
        Time domain signal
    fShift: float
        Frequency shift to apply to x
    fs: float
        Sampling frequency of x

    Returns
    -------
    x_mix: array_like of complex float
        Time domain frequency shifted x

    Notes
    -----
    The data array can be 1D (a single trace) or 2D (many traces). In the 2D
    case each "column" (constnt axis 0) is treated independently.
    """
    # Handle single traces
    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    # Generate time array
    t = np.arange(x.shape[0]) / fs

    # Generate complex signal to mix with
    y = np.exp(2 * np.pi * 1j * fShift * t)[:, np.newaxis]

    # Mix
    x_sh = x * y

    # Do mixing
    return x_sh


def pulseCompressTrig(data, chirp, ttrig, fs=1.4e6):
    """Pulse compress a signal and remove trigger delay.

    Parameters
    ----------
    data: (n x m) array_like of float
        Baseband time domain data, size n x m
    chirp: array_like of float
        Baseband time domain chirp, size n
    ttrig: array_like of float
        Trigger delay in seconds
    fs: float
        Sampling frequency of data

    Returns
    -------
    pc: array_like of float
        Time domain pulse compressed data, size n x m

    Notes
    -----
    """
    # Check dimensions
    if len(data.shape) != 2:
        raise ValueError("data must be 2D")
    if len(chirp.shape) != 1:
        raise ValueError("chirp must be 1D")
    if len(ttrig.shape) != 1:
        raise ValueError("ttrig must be 1D")
    if ttrig.shape[0] != data.shape[1]:
        raise ValueError("ttrig must have entry for each column of data")
    if chirp.shape[0] > data.shape[0]:
        raise ValueError("chirp cannot be longer than number of rows in data")

    # Zero pad chirp
    chirp = np.append(chirp, np.zeros(data.shape[0] - len(chirp)))
    CHIRP = np.fft.fft(chirp)

    # Angular frequency vector for trigger removal
    w = (np.fft.fftfreq(len(chirp), d=1.0 / fs) * 2 * np.pi)[:, np.newaxis]
    ttrig = ttrig[np.newaxis, :]

    # Pulse compress
    PC = (
        np.fft.fft(data, axis=0)
        * np.conj(np.fft.fft(chirp))[:, np.newaxis]
        * np.exp(-1j * w * ttrig)
    )
    pc = np.fft.ifft(PC, axis=0)

    return pc


def trigDelay(edr, fs=1.4e6, c=299792458.0):
    """Calculate F1/F2 trigger delays for an EDR observation

    Parameters
    ----------
    edr: EDR object
        EDR observation for which F1F2 trigger delays should be calculated
    fs: float
        Sampling frequency of data
    c: float
        Speed of light (meters/second)

    Returns
    -------
    dlyF1: array_like of float
        F1 trigger delay
    dlyF2: array_like of foat
        F2 trigger delay
    """
    dt = 1.0 / fs

    dlyF1 = (
        edr.anc["RX_TRIG_SA_PROGR"][:, 0] * (1.0 / (2 * fs))  # programmed trigger delay
        - edr.geo["SPACECRAFT_ALTITUDE"] * 2e3 / c  # minus altitude
        + 256 * dt  # plus half of a trace
    )
    dlyF2 = (
        edr.anc["RX_TRIG_SA_PROGR"][:, 1] * (1.0 / (2 * fs))  # programmed trigger delay
        - edr.geo["SPACECRAFT_ALTITUDE"] * 2e3 / c  # minus altitude
        + 256 * dt  # plus half of a trace
        - 450e-6  # minus extra F2 offset
    )

    return dlyF1, dlyF2
