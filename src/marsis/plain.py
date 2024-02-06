import numpy as np
from tqdm import tqdm

from .util import refChirp, quadMixShift, pulseCompressTrig, trigDelay


def plain(edr):
    dlyF1, dlyF2 = trigDelay(edr)

    # Reference chirp
    chirp = refChirp()

    rgF1 = np.zeros(edr.data["ZERO_F1"].shape, dtype=np.float32)
    for filt in ["MINUS1_F1", "ZERO_F1", "PLUS1_F1"]:
        data_baseband = quadMixShift(edr.data[filt])
        rgF1 += np.abs(pulseCompressTrig(data_baseband, chirp, dlyF1))

    rgF2 = np.zeros(edr.data["ZERO_F2"].shape, dtype=np.float32)
    for filt in ["MINUS1_F2", "ZERO_F2", "PLUS1_F2"]:
        DATA_BASEBAND = quadMixShift(edr.data[filt])
        rgF2 += np.abs(pulseCompressTrig(data_baseband, chirp, dlyF2))

    # Normalize to background
    # TODO: why do I have to do this??
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
