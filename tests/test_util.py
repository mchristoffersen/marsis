import marsis
import numpy as np
import pytest


class TestUtil:
    def test_arangeT(self):
        tstart = -2
        tstop = 3
        fs = 2

        assert np.all(
            marsis.util.arangeT(tstart, tstop, fs)
            == np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
        )

    def test_refChirp(self):
        rc = np.load(__file__.replace("test_util.py", "refChirp.npy"))
        assert marsis.util.refChirp(f0=-0.5e6, f1=0.5e6, m=4.0e9, fs=1.4e6) == pytest.approx(rc)

    def test_quadMixShift(self):
        t = np.arange(100)
        f = 0.1
        x = np.exp(2 * np.pi * 1j * f * t)
        x_sh = marsis.util.quadMixShift(x, fShift=-f, fs=1)
        assert x_sh == pytest.approx(
            np.ones(len(t), dtype=np.complex128)[:, np.newaxis]
        )

    def test_pulseCompressTrig(self):
        rc = np.load(__file__.replace("test_util.py", "refChirp.npy"))
        pc = marsis.util.pulseCompressTrig(rc[:,np.newaxis], rc, np.array([20e-6]), fs=1.4e6)
        bmark = np.load(__file__.replace("test_util.py", "refChirp_autocorr_20us_delay.npy"))
        assert pc == pytest.approx(bmark)