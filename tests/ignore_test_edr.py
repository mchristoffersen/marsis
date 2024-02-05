import marsis


def test_edr():
    file = "./e_07244_ss3_trk_cmp_m.lbl"
    edr = marsis.EDR(file)
    print(edr.geo)


test_edr()
