import matplotlib.pyplot as plt
import numpy as np

import marsis


def test_campbell():
    file = "./e_07244_ss3_trk_cmp_m.lbl"
    edr = marsis.EDR(file)
    f1, f2 = marsis.campbell(edr)
    plt.imshow(np.log(f1 ** 2))
    plt.figure()
    plt.imshow(np.log(f2 ** 2))
    plt.show()


test_campbell()
