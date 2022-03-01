import sys

import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio as rio
import scipy.signal.windows

from .util import genChirp


def lsindx(length):
    # Trace index for least squares
    # Divided by 100 to prevent numerical problems
    return np.arange(length) / 100 + 1


def contrastFit(DATA, ttrig, init=None):
    # Run contrast method over reasonable value and fit curve to it
    # as initial guess for optimization
    m = np.linspace(4e9, 8e9, 100)
    c = np.zeros((len(m), DATA.shape[1]))

    for j in range(DATA.shape[1]):
        for i in range(len(m)):
            c[i, j] = pc_Contrast([m[i]], DATA[:, j], ttrig)

    # Least squares fit
    b = m[np.argmin(c, axis=0)]
    indx = lsindx(len(b))
    A = np.vstack((indx ** 0, indx ** 1, indx ** 2, indx ** 3, indx ** 4, indx ** 5))
    # print(A.T)
    init = scipy.linalg.lstsq(A.T, b)[0]

    return init


def bcOptimize(DATA, ttrig, init=None):
    # Run contrast method over reasonable value and fit curve to it
    # as initial guess for optimization
    m = np.linspace(4e9, 8e9, 100)
    c = np.zeros((len(m), DATA.shape[1]))

    for j in range(DATA.shape[1]):
        for i in range(len(m)):
            c[i, j] = pc_Contrast([m[i]], DATA[:, j], ttrig)

    # Least squares fit
    b = m[np.argmin(c, axis=0)]
    indx = lsindx(len(b))
    A = np.vstack((indx ** 0, indx ** 1, indx ** 2, indx ** 3, indx ** 4, indx ** 5))
    # print(A.T)
    init = scipy.linalg.lstsq(A.T, b)[0]

    # Smooth function SNR based optimization
    res = scipy.optimize.minimize(
        pc_SNR,
        init,
        args=(DATA, ttrig),
        method="Nelder-Mead",
        options={"xatol": 1, "adaptive": True, "maxiter": 50},
    )
    return res.x


def pc_Contrast(b, DATA, ttrig):
    data = pc(b, DATA, ttrig)
    data = np.abs(data)
    contrast = -1 * (np.std(data) / np.mean(data))
    return contrast


def pc_SNR(b, DATA, ttrig):
    np.seterr(all="raise")
    data = pc(b, DATA, ttrig)

    # Calculate summed SNR for the track
    # Take N as the avg of the first 32 samples
    # Take S as the max value of the trace

    data = np.abs(data)
    # Normalize to low echo
    for i in range(data.shape[1]):
        kernel = np.ones(32)
        smooth = np.correlate(data[:, i], kernel, mode="valid")
        ms = np.min(smooth)
        if ms == 0:
            ms = 1
        data[:, i] = data[:, i] / ms
    # N = np.mean(np.abs(data[:32, :]), axis=0)
    # S = np.max(np.abs(data), axis=0)

    # SNR = S/N
    SNR = np.max(data, axis=0)
    sys.stdout.write("SNR=%f\r" % (-1 * np.sum(SNR)))
    return -1 * np.sum(SNR)


def pc(b, DATA, ttrig):
    # b is a vector of floats, coefficents for a polynomial
    # describing chirp rate vs trace index for the chirp
    # Default b for no chirp rate compensation would be [4.0e9]
    # This is not default val to enable plugging this method
    # into minimizers

    DATA = np.copy(DATA)

    if len(DATA.shape) == 1:
        DATA = DATA[:, np.newaxis]

    # Base band and filter
    t = np.arange(DATA.shape[0]) * 1.0 / 1.4e6
    bb = np.exp(2 * np.pi * 1j * -0.7e6 * t)
    data = np.fft.ifft(DATA, axis=0)
    data = data * bb[:, np.newaxis]

    # bfilt, afilt = scipy.signal.butter(4, 0.4e6, "low", fs=1.4e6)
    # data = scipy.signal.filtfilt(bfilt, afilt, data, axis=0)

    DATA = np.fft.fft(data, axis=0)

    ntrace = DATA.shape[1]
    indx = lsindx(ntrace)

    # Calculate phase distortion from given coefficents
    m = np.zeros(len(indx), dtype=np.float128)
    for j in range(len(b)):
        m += b[j] * (indx ** j)

    w = np.fft.fftfreq(DATA.shape[0], d=1.0 / 1.4e6) * 2 * np.pi  # Angular frequency

    # Pulse compress frame by frame
    for j in range(ntrace):
        chirp = genChirp(m=m[j])

        # window
        chirp = chirp * scipy.signal.windows.hann(len(chirp))

        # Handle chirp that is too long
        pad = DATA.shape[0] - len(chirp)
        if pad < 0:
            chirp = chirp[0 : DATA.shape[0]]
        else:
            chirp = np.append(chirp, np.zeros(pad))

        CHIRP = np.fft.fft(chirp)

        DATA[:, j] = DATA[:, j] * np.conj(CHIRP) * np.exp(-1j * w * ttrig[j])

    return np.fft.ifft(DATA, axis=0)


def campbell(edr, dem, cacheIono=False, cache="./", contrast=False):
    """Campbell style ionospheric correction.

    Fit a smooth function to quadratic phase distortion caused
    by the ionosphere, where the max SNR of each frame is a used
    as a goodness metric.

    Parameters
    ----------
    edr: EDR
        EDR object containing the EDR data 
    path: str
        Path to directory where the MARSIS data files will be downloaded to
    clobber: bool
        Whether to overwrite (re-download) files at path

    Returns
    -------
    files: list of str
        Tuple containing the paths of all files downloaded by this function

    Notes
    -----
    """
    c = 299792458
    dt = 1.0 / 1.4e6

    # RX window shift
    tshiftF1 = (
        edr.anc["RX_TRIG_SA_PROGR"][:, 0] * (1 / 2.8e6)
        - edr.geo["SPACECRAFT_ALTITUDE"] * 2e3 / c
        + 256 * dt
    )
    tshiftF2 = (
        (edr.anc["RX_TRIG_SA_PROGR"][:, 1]) * (1 / 2.8e6)
        - edr.geo["SPACECRAFT_ALTITUDE"] * 2e3 / c
        + 256 * dt
        - 450e-6 # Delay from xmit F1 to xmit F2
    )

    # Find when freq switch is (if any)
    dcg = edr.ost["DCG_CONFIGURATION_LO"].to_numpy()
    try:
        switchIdx = np.where(dcg[:-1] != dcg[1:])[0][0]
    except IndexError:
        switchIdx = len(dcg) - 4
        print("No freq switch - using last 3 frames as temp fix")

    f1 = edr.data["ZERO_F1"]
    f2 = edr.data["ZERO_F2"]

    cacheFile = "%s/%s_coeffs.txt" % (cache, edr.lbld["PRODUCT_ID"].lower())
    cmt = """
    if cacheIono:
        if(os.path.isfile(cacheFile)):
            fd = open(cacheFile, "r")
            coeffs = fd.read().split("\n")
            fd.close()
            coeffDict = {}
            for line in coeffs:
                line = line.split("=")
                coeffDict[line.]"""

    # Independently derive ionosphere correction for lo and hi bands on either
    # side of freq switch
    bF1B1 = bcOptimize(f1[:, 0 : switchIdx + 1], tshiftF1[0 : switchIdx + 1])
    bF1B2 = bcOptimize(f1[:, switchIdx + 1 :], tshiftF1[switchIdx + 1 :])
    bF2B1 = bcOptimize(f2[:, 0 : switchIdx + 1], tshiftF2[0 : switchIdx + 1])
    bF2B2 = bcOptimize(f2[:, switchIdx + 1 :], tshiftF2[switchIdx + 1 :])

    # Write coeffs to file
    if cacheIono:
        fd = open(cacheFile, "w")
        elems = "%e, " * len(bF1B1)
        bF1B1Line = "bF1B1 = [" + elems[:-2] + "]\n"
        elems = "%e, " * len(bF1B2)
        bF1B2Line = "bF1B2 = [" + elems[:-2] + "]\n"
        elems = "%e, " * len(bF2B1)
        bF2B1Line = "bF2B1 = [" + elems[:-2] + "]\n"
        elems = "%e, " * len(bF2B2)
        bF2B2Line = "bF2B2 = [" + elems[:-2] + "]\n"

        fd.write(bF1B1Line % tuple(bF1B1))
        fd.write(bF1B2Line % tuple(bF1B2))
        fd.write(bF2B1Line % tuple(bF2B1))
        fd.write(bF2B2Line % tuple(bF2B2))
        fd.close()

    # Make multilook images for f1 and f2
    rgF1 = np.zeros_like(f1, dtype=np.float32)
    rgF1[:, : switchIdx + 1] += np.abs(
        pc(bF1B1, edr.data["MINUS1_F1"][:, : switchIdx + 1], tshiftF1[: switchIdx + 1])
    )
    rgF1[:, : switchIdx + 1] += np.abs(
        pc(bF1B1, edr.data["ZERO_F1"][:, : switchIdx + 1], tshiftF1[: switchIdx + 1])
    )
    rgF1[:, : switchIdx + 1] += np.abs(
        pc(bF1B1, edr.data["PLUS1_F1"][:, : switchIdx + 1], tshiftF1[: switchIdx + 1])
    )
    rgF1[:, switchIdx + 1 :] += np.abs(
        pc(bF1B2, edr.data["MINUS1_F1"][:, switchIdx + 1 :], tshiftF1[switchIdx + 1 :])
    )
    rgF1[:, switchIdx + 1 :] += np.abs(
        pc(bF1B2, edr.data["ZERO_F1"][:, switchIdx + 1 :], tshiftF1[switchIdx + 1 :])
    )
    rgF1[:, switchIdx + 1 :] += np.abs(
        pc(bF1B2, edr.data["PLUS1_F1"][:, switchIdx + 1 :], tshiftF1[switchIdx + 1 :])
    )

    rgF2 = np.zeros_like(f2, dtype=np.float32)
    rgF2[:, : switchIdx + 1] += np.abs(
        pc(bF2B1, edr.data["MINUS1_F2"][:, : switchIdx + 1], tshiftF2[: switchIdx + 1])
    )
    rgF2[:, : switchIdx + 1] += np.abs(
        pc(bF2B1, edr.data["ZERO_F2"][:, : switchIdx + 1], tshiftF2[: switchIdx + 1])
    )
    rgF2[:, : switchIdx + 1] += np.abs(
        pc(bF2B1, edr.data["PLUS1_F2"][:, : switchIdx + 1], tshiftF2[: switchIdx + 1])
    )
    rgF2[:, switchIdx + 1 :] += np.abs(
        pc(bF2B2, edr.data["MINUS1_F2"][:, switchIdx + 1 :], tshiftF2[switchIdx + 1 :])
    )
    rgF2[:, switchIdx + 1 :] += np.abs(
        pc(bF2B2, edr.data["ZERO_F2"][:, switchIdx + 1 :], tshiftF2[switchIdx + 1 :])
    )
    rgF2[:, switchIdx + 1 :] += np.abs(
        pc(bF2B2, edr.data["PLUS1_F2"][:, switchIdx + 1 :], tshiftF2[switchIdx + 1 :])
    )

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

    # Correct to MOLA surface
    # srfF1 = np.argmax(rgF1 > thld, axis=0)
    # srfF2 = np.argmax(rgF2 > thld, axis=0)

    #srfF1 = np.argmax(rgF1, axis=0)
    #srfF2 = np.argmax(rgF2, axis=0)
    
    srfF1 = np.argmax(10*np.log(rgF1) > -15, axis=0)
    srfF2 = np.argmax(10*np.log(rgF2) > -10, axis=0)

    x = np.arange(len(srfF1))
    if(len(x[srfF1 != 0]) != 0):
        # Just skipping bad tracks for now
        srfF1 = np.interp(x, x[srfF1 != 0], srfF1[srfF1 != 0])
        srfF2 = np.interp(x, x[srfF2 != 0], srfF2[srfF2 != 0])


    # MOLA Surface
    xyzcrs = "+proj=geocent +a=3396190 +b=3376200 +no_defs"
    xyz = edr.geo["TARGET_SC_POSITION_VECTOR"] * 1e3
    dem = rio.open(dem, 'r')

    demX, demY, demZ = pyproj.transform(
        xyzcrs, dem.crs, xyz[:, 0], xyz[:, 1], xyz[:, 2]
    )

    iy, ix = dem.index(demX, demY)
    ix = np.array(ix)
    iy = np.array(iy)

    # Temp fix mola meters/pix issue
    ix[ix > dem.width - 1] = dem.width - 1
    ix[ix < 0] = 0

    mola = dem.read(1)[iy, ix] + 3396190

    dem.close()

    angle = np.abs(np.arctan(xyz[:, 2] / np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)))

    a = 3396190
    b = 3376200

    marsR = (a * b) / np.sqrt(
        (a ** 2) * (np.sin(angle) ** 2) + (b ** 2) * (np.cos(angle) ** 2)
    )

    c = 299792458
    molasample = 256 + (((marsR - mola) * 2 / c) * 1.4e6).astype(np.int32)

    dsF1 = (molasample - srfF1).astype(np.int32)
    dsF2 = (molasample - srfF2).astype(np.int32)
    
    for i in range(rgF1.shape[1]):
        rgF1[:, i] = np.roll(rgF1[:, i], dsF1[i])
        rgF2[:, i] = np.roll(rgF2[:, i], dsF2[i])
        
    return rgF1, rgF2
