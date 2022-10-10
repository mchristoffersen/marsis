import numpy as np
from tqdm import tqdm


def arangeT(start, stop, fs):
    # Function to generate set of
    # Args are start time, stop time, sampling frequency
    # Generates times within the closed interval [start, stop] at 1/fs spacing
    # Double precision floating point

    tTot = stop - start
    nsamp = int(tTot * fs)
    seq = np.arange(nsamp, dtype=np.double)
    seq = seq / fs + start

    return seq


def psiGridSearch(TRACE, sim, tshift, band, dly, dlyBound, steps, plot=False):
    # Find min/max psi bounds based on delay
    dlyMax = dly + dlyBound
    dlyMin = max(0, dly - dlyBound)

    psi1C = {3: 3.04e-8, 4: 1.7e-8, 5: 1.08e-8}
    psi2C = {3: 2.83e-7, 4: 8.75e-8, 5: 3.54e-8}
    psi3C = {3: 1.69e-6, 4: 2.85e-7, 5: 7.30e-8}
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
        3: [3.04e-8, 2.83e-7, 1.69e-6],
        4: [1.7e-8, 8.75e-8, 2.85e-7],
        5: [1.08e-8, 3.54e-8, 7.30e-8],
    }
    coeff = coeffs[key]
    return coeff[0] * psis[0] + coeff[1] * psis[1] + coeff[2] * psis[2]


def obj_func(psis, TRACE, ttrig, band, sim):
    pc = pc_phase(psis, TRACE, ttrig, band)

    num = np.sum((np.abs(pc) ** 2) * (sim ** 2))
    srf = np.argmax(sim)
    denom = np.sum(np.abs(pc[0:srf]) ** 2)

    cost = -num / (denom ** 2)

    if False:
        plt.figure()
        plt.plot(np.abs(pc))
        plt.plot(np.abs(sim) * (np.max(np.abs(pc)) / np.max(sim)))
        plt.title(psis)
        plt.figure()
        plt.plot(np.abs(pc[0:srf]))
        plt.title(psis)
        print(psis, num, denom)
    return cost


def pc_phase(psis, TRACE, ttrig, band):
    # psis - ionospheric distortion coeffs
    # TRACE - freq domain trace to pulse compress
    # ttrig - trigger time correction
    # f0 - carrier freq
    (psi1, psi2, psi3) = psis

    # Reference chirp
    fs = 1.4e6
    tlen = 250e-6
    f0 = -0.5e6
    m = 4.0e9
    t = arangeT(0, tlen, fs)  # time vector
    phi = np.pi * (2 * f0 * t + m * (t ** 2))  # phase
    chirp = -1 * np.sin(phi) + 1j * np.cos(phi)  # chirp
    # chirp = chirp*scipy.signal.windows.hann(len(chirp))

    # Zero pad data
    trace = np.fft.ifft(TRACE)
    # trace = np.append(trace, np.zeros(len(chirp)))

    # Zero pad chirp
    chirp = np.append(chirp, np.zeros(len(trace) - len(chirp)))

    # Shift data to baseband
    t = np.arange(len(chirp)) * 1.0 / fs
    bb = np.exp(2 * np.pi * 1j * -0.7e6 * t)
    trace = trace * bb

    # FFT
    CHIRP = np.fft.fft(chirp)
    TRACE = np.fft.fft(trace)

    # Pulse compress
    w = np.fft.fftfreq(len(chirp), d=1.0 / fs) * 2 * np.pi  # Angular frequency
    f = (np.fft.fftfreq(len(chirp), d=1.0 / fs) + band) / 1e6  # Frequency in MHz

    c = 299792458  # speed of light m/s
    PHASE = np.exp(
        ((-1j * 2 * np.pi) / c)
        * (
            (((8.98 ** 2) * psi1) / f)
            + (((8.98 ** 4) * psi2) / (3 * (f ** 3)))
            + (((8.98 ** 6) * psi3) / (8 * (f ** 5)))
        )
    )
    pc = np.fft.ifft(TRACE * np.conj(CHIRP) * np.exp(-1j * w * ttrig) * PHASE)
    return pc


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

        datat = DATA[:, i]
        simt = sim[:, i]
        ttrigt = ttrig[i]

        if findPsis:
            # Grid search ionosphere
            psi = psiGridSearch(datat, simt, ttrigt, band, 0, 256, 10)
            dly = totalDelay(psi, band)[0]

            psi = psiGridSearch(datat, simt, ttrigt, band, dly, 50, 10)
            dly = totalDelay(psi, band)[0]

            psi = psiGridSearch(datat, simt, ttrigt, band, dly, 5, 20)
            dly = totalDelay(psi, band)[0]

            psi = psiGridSearch(datat, simt, ttrigt, band, dly, 1, 40)
            dly = totalDelay(psi, band)[0]

            psis[i] = tuple(psi)

        pc[:, i] = np.abs(pc_phase(psis[i], datat, ttrigt, band))

    if findPsis:
        return pc, psis
    else:
        return pc


def mcmichael(edr, sim):
    c = 299792458
    dt = 1.0 / 1.4e6

    sim = np.fromfile(sim, dtype=np.float32).reshape(edr.data["ZERO_F1"].shape)

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

    # rgF1 = np.zeros(edr.data["ZERO_F1"].shape, dtype=np.float32)
    rgF1, psisF1 = pc_clutter(
        edr.data["ZERO_F1"], sim, tshiftF1, edr.ost["DCG_CONFIGURATION_F1"]
    )
    rgF1 += pc_clutter(
        edr.data["MINUS1_F1"],
        sim,
        tshiftF1,
        edr.ost["DCG_CONFIGURATION_F1"],
        psis=psisF1,
    )
    rgF1 += pc_clutter(
        edr.data["PLUS1_F1"],
        sim,
        tshiftF1,
        edr.ost["DCG_CONFIGURATION_F1"],
        psis=psisF1,
    )

    rgF2, psisF2 = pc_clutter(
        edr.data["ZERO_F2"], sim, tshiftF2, edr.ost["DCG_CONFIGURATION_F2"]
    )
    rgF2 += pc_clutter(
        edr.data["MINUS1_F2"],
        sim,
        tshiftF2,
        edr.ost["DCG_CONFIGURATION_F2"],
        psis=psisF2,
    )
    rgF2 += pc_clutter(
        edr.data["PLUS1_F2"],
        sim,
        tshiftF2,
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
