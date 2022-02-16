import numpy as np
import PIL as pil


def gen_tiff(rg, name):
    # Log scaling
    rg = np.log(np.power(rg, 2))
    rg -= rg.min()
    rg *= 255 / rg.max()
    rg = rg.astype(np.uint8)
    img = pil.Image.fromarray(rg)
    img.save(name)


def gen_segy():
    pass
