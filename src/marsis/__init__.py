import logging

log = logging.getLogger("marsis")
log.setLevel("INFO")
log.propagate = 0
ch = logging.StreamHandler()
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
log.addHandler(ch)

from .campbell import campbell
from .mcmichael import mcmichael
from .plain import plain
from .edr import EDR
from .fetch import fetch
from .output import gen_segy, gen_tiff