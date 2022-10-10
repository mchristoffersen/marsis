import numpy as np
from tqdm import tqdm

from .util import refChirp


def pc(DATA, ttrig):
    # TRACE - freq domain data to pulse compress
    # ttrig - trigger time correction

    chirp = refChirp()

    # Zero pad chirp
    chirp = np.append(chirp, np.zeros(DATA.shape[0] - len(chirp)))

    # Shift data to baseband
    fs = 1.4e6
    t = np.arange(len(chirp)) * 1.0 / fs
    bb = np.exp(2 * np.pi * 1j * -0.7e6 * t)
    data = np.fft.ifft(DATA, axis=0)
    data = data * bb[:, np.newaxis]

    # FFT
    CHIRP = np.fft.fft(chirp)
    DATA = np.fft.fft(data, axis=0)

    # Pulse compress
    w = (np.fft.fftfreq(len(chirp), d=1.0 / fs) * 2 * np.pi)[
        :, np.newaxis
    ]  # Angular frequency
    ttrig = ttrig[np.newaxis, :]

    c = 299792458  # speed of light m/s
    pc = np.fft.ifft(
        DATA * np.conj(CHIRP)[:, np.newaxis] * np.exp(-1j * w * ttrig), axis=0
    )
    return np.abs(pc)


def plain(edr):
    c = 299792458
    dt = 1.0 / 1.4e6

    tshiftF1 = (
        edr.anc["RX_TRIG_SA_PROGR"][:, 0] * (1 / 2.8e6)
        - edr.geo["SPACECRAFT_ALTITUDE"] * 2e3 / c
        + 256 * dt
    )
    tshiftF2 = (
        edr.anc["RX_TRIG_SA_PROGR"][:, 1] * (1 / 2.8e6)
        - edr.geo["SPACECRAFT_ALTITUDE"] * 2e3 / c
        + 256 * dt
        - 450e-6
    )

    rgF1 = np.zeros(edr.data["ZERO_F1"].shape, dtype=np.float32)
    rgF1 += pc(edr.data["ZERO_F1"], tshiftF1)
    rgF1 += pc(edr.data["MINUS1_F1"], tshiftF1)
    rgF1 += pc(edr.data["PLUS1_F1"], tshiftF1)

    rgF2 = np.zeros(edr.data["ZERO_F2"].shape, dtype=np.float32)
    rgF2 += pc(edr.data["ZERO_F2"], tshiftF2)
    rgF2 += pc(edr.data["MINUS1_F2"], tshiftF2)
    rgF2 += pc(edr.data["PLUS1_F2"], tshiftF2)

    # Normalize to background
    kernel = np.ones(32)
    for i in range(rgF1.shape[1]):
        smooth = np.correlate(rgF1[:, i], kernel, mode="valid")
        ms = np.min(smooth)
        if ms == 0:
            ms = 1
        rgF1[:, i] = rgF1[:, i] / ms

    for i in range(rgF2.shape[1]):
        smooth = np.correlate(rgF2[:, i], kernel, mode="valid")
        ms = np.min(smooth)
        if ms == 0:
            ms = 1
        rgF2[:, i] = rgF2[:, i] / ms

    return rgF1, rgF2
