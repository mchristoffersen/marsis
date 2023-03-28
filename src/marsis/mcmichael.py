import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.optimize

from marsis.util import arangeT, trigDelay, refChirp, pulseCompressTrig, baseBand
from marsis import log


def psiGridSearch(TRACE, sim, tshift, band, dly, dlyBound, steps, plot=False):
    # Find min/max psi bounds based on delay
    dlyMax = dly + dlyBound
    dlyMin = max(0, dly - dlyBound)

    # McMichael 2017 Table 1
    psi1C = {1: 8.71e-8, 3: 3.04e-8, 4: 1.7e-8, 5: 1.08e-8}
    psi2C = {1: 2.43e-6, 3: 2.83e-7, 4: 8.75e-8, 5: 3.54e-8}
    psi3C = {1: 4.50e-5, 3: 1.69e-6, 4: 2.85e-7, 5: 7.30e-8}
    key = int(band / 1e6)
    psi1Max = dlyMax / psi1C[key]
    psi2Max = dlyMax / psi2C[key]
    psi3Max = dlyMax / psi3C[key]

    # Constrained grid search by dlyBound samples around dly
    # print(psi1Min, psi2Min, psi3Min)
    psi1 = np.linspace(0, psi1Max, steps)
    psi2 = np.linspace(0, psi2Max, steps)
    psi3 = np.linspace(0, psi3Max, steps)

    res = np.zeros((steps, steps, steps))
    for i in range(len(psi1)):
        for j in range(len(psi2)):
            for k in range(len(psi3)):
                delay = totalDelay([psi1[i], psi2[j], psi3[k]], band)
                if np.abs(delay - dly) > dlyBound:
                    res[i, j, k] = 0
                else:
                    res[i, j, k] = obj_func(
                        [psi1[i], psi2[j], psi3[[k]]], TRACE, tshift, band, sim
                    )

    foc = np.where(res == res.min())

    if plot:
        plt.figure()
        img = res[:, :, foc[2]].reshape((steps, steps))
        img = np.log(np.abs(img))
        plt.imshow(img, aspect="auto")
        plt.plot([foc[1]], [foc[0]], "r.")
    op1 = psi1[foc[0]]
    op2 = psi2[foc[1]]
    op3 = psi3[foc[2]]

    return [op1, op2, op3]


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


def obj_func(psis, TRACE, ttrig, band, sim):
    # Make sure dims match
    if len(sim.shape) == 1:
        sim = sim[:, np.newaxis]

    pc = pc_phase(psis, TRACE, np.array([ttrig]), band)

    # print(pc.shape, sim.shape)

    num = np.sum((np.abs(pc) ** 2) * (sim ** 2))
    srf = np.argmax(sim)
    denom = np.sum(np.abs(pc[0:srf]) ** 2)

    # plt.plot(np.abs(pc)**2/np.max(np.abs(pc)**2))
    # plt.plot(np.abs(sim)**2/np.max(np.abs(sim)**2))
    # plt.plot((np.abs(pc) ** 2) * (sim ** 2))
    # plt.title(num)

    # plt.show()

    cost = -num/(denom ** 2)

    if False:
        plt.figure()
        plt.plot(np.abs(pc))
        plt.plot(np.abs(sim) * (np.max(np.abs(pc)) / np.max(sim)))
        plt.title(psis)
        plt.figure()
        plt.plot(np.abs(pc[0:srf]))
        plt.title(psis)
        print(psis, num, denom)
    return cost #*1e32


def pc_phase(psis, TRACE, dly, band, fs=1.4e6):
    # psis - ionospheric distortion coeffs
    # TRACE - freq domain trace to pulse compress
    # ttrig - trigger time correction
    # f0 - carrier freq
    (psi1, psi2, psi3) = psis
    
    #psi1 *= 1e10
    #psi2 *= 1e10
    #psi3 *= 1e10

    # Reference chirp
    CHIRP = np.fft.fft(refChirp())

    # Shift data to baseband
    TRACE = baseBand(TRACE)

    # Pulse compress
    w = np.fft.fftfreq(len(TRACE), d=1.0 / fs) * 2 * np.pi  # Angular frequency
    f = (np.fft.fftfreq(len(TRACE), d=1.0 / fs) + band) / 1e6  # Frequency in MHz

    c = 299792458  # speed of light m/s
    PHASE = np.exp(
        ((-1j * 2 * np.pi) / c)
        * (
            (((8.98 ** 2) * psi1) / f)
            + (((8.98 ** 4) * psi2) / (3 * (f ** 3)))
            + (((8.98 ** 6) * psi3) / (8 * (f ** 5)))
        )
    )

    PC = pulseCompressTrig(TRACE, CHIRP, dly) * PHASE[:, np.newaxis]

    return np.fft.ifft(PC, axis=0)


def pc_clutter(DATA, sim, ttrig, dcg_config, psis=None):
    findPsis = False
    if psis is None:
        findPsis = True
        psi_t = np.dtype(
            [("psi1", np.float32), ("psi2", np.float32), ("psi3", np.float32)]
        )
        psis = np.empty(DATA.shape[1], dtype=psi_t)

    bands = [1.8e6, 3e6, 4e6, 5e6]
    pc = np.zeros(DATA.shape, dtype=np.float32)
    for i in tqdm(range(DATA.shape[1])):
        band = bands[dcg_config[i]]

        TRACE = DATA[:, i]
        simt = sim[:, i]
        ttrigt = ttrig[i]

        if findPsis:
            if(i == 0):
                x0 = [0,0,0]
            else:
                x0 = list(psis[i-1])
            
            # Grid search ionosphere
            psi = psiGridSearch(TRACE, simt, ttrigt, band, 0, 256, 5)
            dly = totalDelay(psi, band)[0]

            psi = psiGridSearch(TRACE, simt, ttrigt, band, dly, 50, 10)
            dly = totalDelay(psi, band)[0]

            psi = psiGridSearch(TRACE, simt, ttrigt, band, dly, 5, 20)
            dly = totalDelay(psi, band)[0]
                        
            psi = psiGridSearch(TRACE, simt, ttrigt, band, dly, 1, 40)

            #print()
            #print("Nit:", res.nit)
            #print("Hess:\n", res.hess_inv.todense())
            #print("Jac:\n", res.jac)
            #print("Psis", res.x)
        
            #print()
            #print("Grid", obj_func(psi, TRACE, ttrigt, band, simt))
            #print("Optim", obj_func(res.x, TRACE, ttrigt, band, simt))

            #dly = totalDelay(psi, band)

            psis[i] = tuple(psi)

        pc[:, i] = np.abs(pc_phase(psis[i], TRACE, np.array([ttrigt]), band))[:, 0]

    if findPsis:
        return pc, psis
    else:
        return pc


def mcmichael(edr, sim):
    # Load clutter sim
    sim = np.fromfile(sim, dtype=np.float32).reshape(edr.data["ZERO_F1"].shape)

    dlyF1, dlyF2 = trigDelay(edr)

    # F1
    # Find ionosphere coeffs with zero filt then apply to all filts
    rgF1, psisF1 = pc_clutter(
        edr.data["ZERO_F1"], sim, dlyF1, edr.ost["DCG_CONFIGURATION_F1"]
    )
    rgF1 += pc_clutter(
        edr.data["MINUS1_F1"],
        sim,
        dlyF1,
        edr.ost["DCG_CONFIGURATION_F1"],
        psis=psisF1,
    )
    rgF1 += pc_clutter(
        edr.data["PLUS1_F1"],
        sim,
        dlyF1,
        edr.ost["DCG_CONFIGURATION_F1"],
        psis=psisF1,
    )

    # F2
    # Find ionosphere coeffs with zero filt then apply to all filts
    rgF2, psisF2 = pc_clutter(
        edr.data["ZERO_F2"], sim, dlyF2, edr.ost["DCG_CONFIGURATION_F2"]
    )
    rgF2 += pc_clutter(
        edr.data["MINUS1_F2"],
        sim,
        dlyF2,
        edr.ost["DCG_CONFIGURATION_F2"],
        psis=psisF2,
    )
    rgF2 += pc_clutter(
        edr.data["PLUS1_F2"],
        sim,
        dlyF2,
        edr.ost["DCG_CONFIGURATION_F2"],
        psis=psisF2,
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

    return rgF1, rgF2, psisF1, psisF2
