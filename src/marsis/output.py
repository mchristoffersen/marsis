import PIL as pil
import numpy as np


def gen_tiff(rg, name):
    # Log scaling
    rg = np.log(np.power(rg, 2))
    rg -= rg.min()
    rg *= 255/rg.max()
    rg = rg.astype(np.uint8)
    img = pil.Image.fromarray(rg)
    img.save(name)


def gen_segy():
    pass
