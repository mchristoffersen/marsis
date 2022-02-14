import marsis


def test_fetch():
    file = ["E_07244_SS3_TRK_CMP_M"]
    path = "./"
    marsis.fetch(file, path)


test_fetch()
