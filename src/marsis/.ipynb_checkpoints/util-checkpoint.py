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


def baseBand(DATA, bbShift=-0.7e6, fs=1.4e6):
    """Frequency shift a signal.

    Parameters
    ----------
    DATA: array_like of float
        Frequency domain data
    bbShift: float
        Frequency shift to apply to data
    fs: float
        Sampling frequency of data

    Returns
    -------
    SHIFT: array_like of float
        Frequency domain shifted data

    Notes
    -----
    The data array can be 1D (a single trace) or 2D (many traces). In the 2D
    case it is assumed that the first axis is the fast-time axis and is in the
    frequency domain.
    The data
    """
    # Handle single traces
    if(len(DATA.shape) == 1):
        DATA = DATA[:, np.newaxis]
        
    # Generate time array
    t = np.arange(DATA.shape[0]) * 1.0 / fs
    
    # Generate complex signal to mix with
    bb = np.exp(2 * np.pi * 1j * bbShift * t)
    
    # Do mixing
    data = np.fft.ifft(DATA, axis=0)
    SHIFT = np.fft.fft(data * bb[:, np.newaxis], axis=0)

    return SHIFT


def pulseCompressTrig(DATA, CHIRP, ttrig, fs=1.4e6):
    """Pulse compress a signal and remove trigger delay.

    Parameters
    ----------
    DATA: array_like of float
        Baseband frequency domain data
    CHIRP: array_like of float
        Baseband frequency domain chirp
    ttrig: array_like of float
        Trigger delay in seconds
    fs: float
        Sampling frequency of data

    Returns
    -------
    PC: array_like of float
        Frequency domain pulse compressed data

    Notes
    -----
    The data array can be 1D (a single trace) or 2D (many traces). In the 2D
    case it is assumed that the first axis is the fast-time axis and is in the
    frequency domain.
    """
    # Handle single traces
    if(len(DATA.shape) == 1):
        DATA = DATA[:, np.newaxis]
        
    # Zero pad chirp
    chirp = np.append(np.fft.ifft(CHIRP), np.zeros(DATA.shape[0] - len(CHIRP)))
    CHIRP = np.fft.fft(chirp)

    # Angular frequency vector for trigger removal
    w = (np.fft.fftfreq(len(chirp), d=1.0 / fs) * 2 * np.pi)[:, np.newaxis]
    ttrig = ttrig[np.newaxis, :]
    
    # Pulse compress
    PC = DATA * np.conj(CHIRP)[:, np.newaxis] * np.exp(-1j * w * ttrig)

    return PC


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
