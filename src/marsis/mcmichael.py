import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.optimize
import tqdm

from marsis.util import arangeT, trigDelay, refChirp, pulseCompressTrig, quadMixShift
import marsis.plain
from marsis import log


def delayCoeffs(band):
    # Return pure delay coefficents
    # Table I from McMichael
    key = int(band / 1e6)
    coeffs = {
        1: [8.71e-8, 2.43e-6, 4.50e-5],
        3: [3.04e-8, 2.83e-7, 1.69e-6],
        4: [1.7e-8, 8.75e-8, 2.85e-7],
        5: [1.08e-8, 3.54e-8, 7.30e-8],
    }
    return coeffs[key]


def psiScale(psis):
    # Apply scaling to normalized psis
    return psis*1e9


def totalDelay(psis, band):
    # Total delay in samples
    coeff = delayCoeffs(band)
    psis = psiScale(psis)
    return np.squeeze(coeff[0] * psis[0] + coeff[1] * psis[1] + coeff[2] * psis[2])


def distortion(psis, n, band, fs=1.4e6):
    # Calculate distortion
    f = (np.fft.fftfreq(n, d=1.0 / fs) + band) / 1e6  # Frequency in MHz

    psis = psiScale(psis)

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


def obj_func(psis, trace, sim, band, delayPrev, plot=False):
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

    if(delayPrev is not None):
        #print("Penalty", np.abs(totalDelay(psis, band)-delayPrev))
        #print(cost)
        cost += np.abs(totalDelay(psis, band)-delayPrev)*1e-6
        #print(cost)
    # print(cost, dn, cost+dn)

    return cost*1e5


def find_distortion(trace, sim, band, delay, delayBound, delayPrev, steps=10):
    # Find min/max psi bounds based on delay
    delayMax = delay + delayBound
    delayMin = max(0, delay - delayBound)

    coeffs = delayCoeffs(band)

    psiBounds = []
    for i in range(3):
        coeffScale = psiScale(coeffs[i])
        psiBounds.append((delayMin / coeffScale, delayMax / coeffScale))

    psiSpace = []
    for i in range(3):
        nstep = int(psiBounds[i][1] / ((psiBounds[i][1]-psiBounds[i][0])/steps))
        #print(nstep)
        psiSpace.append(np.linspace(0, psiBounds[i][1], nstep))

    # print("Delay bounds:", delayMin, delayMax)
    # Get 10 steps between max and min

    trace = np.fft.fft(trace, axis=0)

    # Find parameter sets with valid delay
    pp1, pp2, pp3 = np.meshgrid(psiSpace[0], psiSpace[1], psiSpace[2], indexing="ij")
    delayGrid = totalDelay(np.array([pp1, pp2, pp3]), band)
    delayMask = np.abs(delayGrid - delay) <= delayBound
    iv, jv, kv = np.where(delayMask)

    # print("Parameter sets:", np.sum(delayMask))
    # Only loop over those
    res = np.zeros_like(delayGrid)
    res[:] = np.nan
    for l in range(len(iv)):
        i = iv[l]
        j = jv[l]
        k = kv[l]
        res[i, j, k] = obj_func(
            np.array([pp1[i, j, k], pp2[i, j, k], pp3[i, j, k]]), trace, sim, band, delayPrev
        )

    #nanpct = np.sum(np.isnan(res)) / (res.shape[0] * res.shape[1] * res.shape[2])

    foc = np.where(res == np.nanmin(res))
    if(len(foc[0]) > 1):
        print(foc)

    op1 = pp1[foc]
    op2 = pp2[foc]
    op3 = pp3[foc]

    # Plot three slices going through min
    if False:
        #print(op1, op2, op3)
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        axs[0].pcolormesh(psiSpace[2], psiSpace[1], res[foc[0][0], :, :])
        axs[0].plot(op3, op2, "r.")
        axs[0].set(xlabel="$\psi_3$", ylabel="$\psi_2$")


        axs[1].pcolormesh(psiSpace[2], psiSpace[0], res[:, foc[1][0], :])
        axs[1].plot(op3, op1, "r.")
        axs[1].set(xlabel="$\psi_3$", ylabel="$\psi_1$")

        axs[2].pcolormesh(psiSpace[1], psiSpace[0], res[:, :, foc[2][0]])
        axs[2].plot(op2, op1, "r.")
        axs[2].set(xlabel="$\psi_2$", ylabel="$\psi_1$")

        plt.show()

    return np.array([op1, op2, op3])


def mcmichael(edr, sim, progress_bar=True):
    # Load clutter sim
    sim = np.fromfile(sim, dtype=np.float32).reshape(edr.data["ZERO_F1"].shape)

    sim = np.roll(sim, 8, axis=0)

    # Trigger delay
    trig = {}
    trig["F1"], trig["F2"] = trigDelay(edr)

    # Reference chirp
    chirp = refChirp()

    # No correction multilook radargrams
    rgml = {}
    rgml["F1"], rgml["F2"] = marsis.plain(edr)

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
        delayEst = np.argmax(rgml[f], axis=0) - np.argmax(sim, axis=0)

        n = 10
        delayEst = (
            np.convolve(
                np.append(delayEst, np.ones(n) * delayEst[-1]), np.ones(n), mode="same"
            )
            / n
        )
        delayEst = delayEst[:-n]

        # Clip to reasonable bounds
        delayEst = np.maximum(0, delayEst)
        delayEst = np.minimum(512, delayEst)

        # Find phase distortion to correct each trace
        psis = [None] * rg.shape[1]
        dlyPrev = None

        delayHist = []
        for i in tqdm.tqdm(range(rg.shape[1]), disable=not progress_bar):
            if band[i] != band[i - 1]:  # Reset prev delay if band chnage
                dlyPrev = None

            delay = delayEst[i]
            delayBound = [125, 25, 5, 2]
            steps = [10, 10, 5, 5]

            for j in range(len(steps)):
                # print("Given delay:", delay)
                psi = find_distortion(
                    rg[:, i],
                    sim[:, i],
                    band[i],
                    delay,
                    delayBound[j],
                    dlyPrev,
                    steps=steps[j],
                )
                delay = totalDelay(psi, band[i])

            #plt.plot(sim[:,i], 'k')
            #ax2 = plt.gca().twinx()
            #ax2.plot(np.abs(rg[:,i]), 'r')
            #plt.show()
                # print("Best delay:", delay)
            # print("Final delay:", delay)
            # print()
            psi = np.squeeze(psi)
            if(len(psi.shape) > 1):
                print(psi.shape)
                print(psi)
            #bounds = scipy.optimize.Bounds(lb=(0, 0, 0), ub=(np.inf, np.inf, np.inf))
            #optim = scipy.optimize.minimize(
            #    obj_func,
            #    psi,
            #    args=(rg[:, i], sim[:, i], band[i], dlyPrev),
            #    method="L-BFGS-B",
            #    bounds=bounds,
            #    options={"eps": 1e-5},
            #)
            #psi = optim.x

            #plt.figure()
            #plt.plot(sim[:,i], "k")
            #corr = np.fft.ifft(np.fft.fft(rg[:, i])*distortion(psi, len(rg[:, i]), band[i]))
            #print(corr.shape)
            #ax2 = plt.gca().twinx()
            #ax2.plot(np.abs(corr), "")
            #plt.show()
            delay = totalDelay(psi, band[i])
            dlyPrev = delay
            delayHist.append(delay)
            psis[i] = psi
            #if(i == 800):
            #    break

        plt.plot(delayHist)
        plt.show()
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
        pc = np.zeros(rg.shape, np.float32)
        for i in range(pc.shape[1]):
            if psis[i] is None:
                psis[i] = (0, 0, 0)
            for dop in ["MINUS1", "ZERO", "PLUS1"]:
                pc[:, i] += np.abs(
                    np.fft.ifft(
                        np.fft.fft(rgs[dop][:, i], axis=0)
                        * distortion(np.array(psis[i]), rgs[dop].shape[0], band[i]),
                        axis=0,
                    )
                )

        # Normalize to background (first 20 samples)
        pc = pc[:512, :]  # Crop away padding
        mv = np.mean(pc[:20, :], axis=0)
        pc /= mv[np.newaxis, :]        
        plt.imshow(np.log10(pc[:512,:]), aspect="auto")
        plt.show()

        outrg[f] = pc[:]
        outpsi[f] = psis[:]

    return outrg["F1"], outrg["F2"], outpsi["F1"], outpsi["F2"]
