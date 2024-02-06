import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.optimize
import tqdm

from marsis.util import arangeT, trigDelay, refChirp, pulseCompressTrig, quadMixShift
from marsis import log


def totalDelay(psis, band):
    # Total delay in samples
    key = int(band / 1e6)
    coeffs = {
        1: [8.71e-8, 2.43e-6, 4.50e-5],
        3: [3.04e-8, 2.83e-7, 1.69e-6],
        4: [1.7e-8, 8.75e-8, 2.85e-7],
        5: [1.08e-8, 3.54e-8, 7.30e-8],
    }
    coeff = coeffs[key]
    return coeff[0] * psis[0] + coeff[1] * psis[1] + coeff[2] * psis[2]


def distortion(psis, n, band, fs=1.4e6):
    # Calculate distortion
    f = (np.fft.fftfreq(n, d=1.0 / fs) + band) / 1e6  # Frequency in MHz

    c = 299792458  # speed of light m/s
    phase = np.exp(
        ((-1j * 2 * np.pi) / c)
        * (
            (((8.98**2) * psis[0]) / f)
            + (((8.98**4) * psis[1]) / (3 * (f**3)))
            + (((8.98**6) * psis[2]) / (8 * (f**5)))
        )
    )

    return phase


def obj_func(psis, trace, sim, band, plot=False):
    # Make sure dims match
    if len(sim.shape) == 1:
        sim = sim[:, np.newaxis]

    pc = np.fft.ifft(trace * distortion(psis, len(trace), band), axis=0)

    # Normalize simulation and pulse compressed trace
    sim = sim / np.max(sim)
    pc = np.abs(pc) / np.max(np.abs(pc))
    pc = pc[: len(sim)]

    if plot:
        # plt.figure()
        # print(np.abs(distortion(psis, len(trace), band)))
        plt.figure()
        plt.plot(pc)
        # ax2 = plt.gca().twinx()
        plt.plot(sim)
        plt.show()

    num = np.sum(np.squeeze(pc**2) * np.squeeze(sim**2))
    srf = np.argmax(sim)
    denom = np.sum(pc[0:srf]) ** 2

    cost = -num / (denom**2)

    # print(cost, dn, cost+dn)

    return cost


def find_distortion(trace, sim, band, delay, delayBound, delayPrev, steps=10):
    # Find min/max psi bounds based on delay
    delayMax = delay + delayBound
    delayMin = max(0, delay - delayBound)

    # McMichael 2017 Table 1
    psi1C = {1: 8.71e-8, 3: 3.04e-8, 4: 1.7e-8, 5: 1.08e-8}
    psi2C = {1: 2.43e-6, 3: 2.83e-7, 4: 8.75e-8, 5: 3.54e-8}
    psi3C = {1: 4.50e-5, 3: 1.69e-6, 4: 2.85e-7, 5: 7.30e-8}
    key = int(band / 1e6)
    psi1Max = delayMax / psi1C[key]
    psi2Max = delayMax / psi2C[key]
    psi3Max = delayMax / psi3C[key]

    # Constrained grid search by delayBound samples around delay
    psi1 = np.linspace(0, psi1Max, steps)
    psi2 = np.linspace(0, psi2Max, steps)
    psi3 = np.linspace(0, psi3Max, steps)

    res = np.zeros((steps, steps, steps))
    dly = np.zeros((steps, steps, steps))
    dly[:] = np.nan
    res[:] = np.nan
    trace = np.fft.fft(trace, axis=0)
    for i in range(len(psi1)):
        for j in range(len(psi2)):
            for k in range(len(psi3)):
                gDelay = totalDelay([psi1[i], psi2[j], psi3[k]], band)
                if np.abs(gDelay - delay) > delayBound:
                    continue
                else:
                    res[i, j, k] = obj_func(
                        [psi1[i], psi2[j], psi3[k]], trace, sim, band
                    )
                    dly[i, j, k] = gDelay

    if delayPrev is not None:
        # Weight objective function based on closeness to previous delay
        dDly = np.abs(dly - delayPrev)
        res += dDly * 1e-5

    # print(np.nanmin(res), np.nanmax(res))

    # nanpct = np.sum(np.isnan(res))/(res.shape[0]*res.shape[1]*res.shape[2])
    # print(nanpct)
    foc = np.where(res == np.nanmin(res))

    op1 = psi1[foc[0]]
    op2 = psi2[foc[1]]
    op3 = psi3[foc[2]]

    return np.array([op1, op2, op3])


def mcmichael(edr, sim):
    # Load clutter sim
    sim = np.fromfile(sim, dtype=np.float32).reshape(edr.data["ZERO_F1"].shape)

    # Trigger delay
    trig = {}
    trig["F1"], trig["F2"] = trigDelay(edr)

    # Reference chirp
    chirp = refChirp()

    outrg = {}
    outpsi = {}
    for f in ["F1", "F2"]:
        # Pulse commpress
        data_baseband = quadMixShift(edr.data["ZERO_" + f])
        data_baseband = np.vstack(
            (
                data_baseband,
                np.zeros((512, data_baseband.shape[1]), dtype=np.complex128),
            )
        )
        rg = pulseCompressTrig(data_baseband, chirp, trig[f])

        # Get bands
        bands = [1.8e6, 3e6, 4e6, 5e6]
        band = [bands[i] for i in edr.ost["DCG_CONFIGURATION_" + f]]

        # Rough delay estimate
        delayEst = np.argmax(rg, axis=0) - np.argmax(sim, axis=0)
        n = 10
        delayEst = (
            np.convolve(
                np.append(delayEst, np.ones(n) * delayEst[-1]), np.ones(n), mode="same"
            )
            / n
        )
        delayEst = delayEst[:-n]
        plt.plot(delayEst)
        plt.show()

        # Find phase distortion to correct each trace
        psis = [None] * rg.shape[1]
        dlyPrev = None
        for i in tqdm.tqdm(range(rg.shape[1]), disable=False):
            if band[i] != band[i - 1]:  # Reset prev delay if band chnage
                dlyPrev = None

            psi = find_distortion(
                rg[:, i], sim[:, i], band[i], delayEst[i], 100, dlyPrev, steps=10
            )
            delay = totalDelay(psi, band[i])[0]
            psis0 = psi[:]
            delay0 = delay

            psi = find_distortion(
                rg[:, i], sim[:, i], band[i], delay, 25, dlyPrev, steps=10
            )

            try:
                delay = totalDelay(psi, band[i])[0]
            except IndexError:
                print(psis0)
                print(delay0)
                print(psi)
                return 0, 0, 0, 0
            psi = find_distortion(
                rg[:, i], sim[:, i], band[i], delay, 5, dlyPrev, steps=20
            )
            delay = totalDelay(psi, band[i])[0]
            dlyPrev = delay
            psis[i] = psi

        pc = np.zeros_like(rg)
        rgs = {}
        for dop in ["MINUS1", "ZERO", "PLUS1"]:
            data_baseband = quadMixShift(edr.data[dop + "_" + f])
            data_baseband = np.vstack(
                (
                    data_baseband,
                    np.zeros((512, data_baseband.shape[1]), dtype=np.complex128),
                )
            )
            rgs[dop] = pulseCompressTrig(data_baseband, chirp, trig[f])

        # Make radargram
        for i in range(pc.shape[1]):
            for dop in ["MINUS1", "ZERO", "PLUS1"]:
                pc[:, i] += np.abs(
                    np.fft.ifft(
                        np.fft.fft(rgs[dop][:, i], axis=0)
                        * distortion(psis[i], rgs[dop].shape[0], band[i]),
                        axis=0,
                    )
                )

        outrg[f] = pc[:]
        outpsi[f] = psis[:]

    # Normalize to background (first 20 samples)
    for f in ["F1", "F2"]:
        outrg[f] = outrg[f][:512, :]  # Crop away padding
        mv = np.mean(outrg[f][:20, :], axis=0)
        outrg["f"] /= mv[np.newaxis, :]

    return outrg["F1"], outrg["F2"], outpsi["F1"], outpsi["F2"]
