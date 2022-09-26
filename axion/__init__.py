__version__ = "0.0.1a"

from . import lcm
from . import utils
from .lcm.skymaps.plotutils import plot_power, plot_sky_count

from matplotlib.pyplot import style, register_cmap
import os

# add my custom style to matplotlib's styles: (ripped off from arviz https://github.com/arviz-devs/arviz/blob/main/arviz/__init__.py)
_my_style_path = os.path.join(os.path.dirname(__file__), "styles")
style.core.USER_LIBRARY_PATHS.append(_my_style_path)
style.core.reload_library()

# import Planck colour map
# this was adapted from Andrea Zonca's code at https://zonca.dev/2013/09/Planck-CMB-map-at-high-resolution.html
from matplotlib.colors import ListedColormap
import requests
from io import StringIO
import numpy as np

txt = StringIO(
    requests.get(
        "https://raw.githubusercontent.com/zonca/paperplots/master/data/Planck_Parchment_RGB.txt"
    ).content.decode("utf-8")
)

planck_cmap = ListedColormap(np.loadtxt(txt) / 255.0)
planck_cmap.set_bad("gray")  # color of missing pixels
planck_cmap.set_under("white")  # color of background, necessary if you want to use
register_cmap("planck", planck_cmap)

del np
del style, register_cmap, ListedColormap, planck_cmap, txt
del os, requests, StringIO
