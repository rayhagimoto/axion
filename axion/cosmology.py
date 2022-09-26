"Physical constants and LCDM cosmology parameters"
import numpy as np
from scipy.integrate import quad

# ------------------------------------ UNITS ------------------------------------#


Km_per_Mpc = 3.089e19 # Km per Mpc
eV_per_J = 6.242e+18 # eV per Joule

h = 6.62607015e-34 # J s
hbar = h * eV_per_J / 2/ np.pi

# ---------------------------------- COSMOLOGY ----------------------------------#
# LCDM Parameters
OmegaL = 0.7
OmegaM = 0.3
OmegaR = 9e-5
hbarH0 = hbar * (70 / Km_per_Mpc)  # hbar times H0 in eV
zcmb = 1100

# Dimensionless Hubble parameter
H = lambda z: np.sqrt(OmegaR * (1 + z) ** 4 + OmegaM * (1 + z) ** 3 + OmegaL)

# ufunc for comoving distance between us and a comoving light source observed to have redshift z_upper
s = np.frompyfunc(
    lambda z_upper: quad(lambda z: 1 / H(z), 0.0, z_upper)[0], nin=1, nout=1
)
# -------------------------- OTHER PHYSICAL CONSTANTS ---------------------------#


alphaEM = 1 / 137  # electromagnetic fine structure constant