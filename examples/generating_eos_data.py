"""Generating particle fractions and effective masses for several sets of equation of state parameters
"""

import numpy as np
import pandas as pd
import ns_eos.equilibrium_comp as ec


# natural constants
c = 2.9979246e23  # speed of light in fm/s
hbar = 6.582120e-22  # hbar in MeV s
m_u = 1.0364270e-44  # atomic mass unit in MeV s^2/fm^2
m_u_cgs = 1.66053906660e-24  # atomic mass unit in g
m_mu = 0.113429 * m_u  # muon mass unit in MeV s^2/fm^2


# baryon number density and mass energy density in units of 1/fm**3 and g/cm**3, respectively
n_b = np.arange(0.06, 0.6, 0.01)
rho_b = n_b * m_u_cgs * 1e13 ** 3


# ---- NRAPR EoS - parameters are taken from Steiner et al. (2005) ---- #

eos_NRAPR = ec.EquationOfState(
    t0=-2719.7,
    t1=417.64,
    t2=-66.687,
    t3=15042.0,
    x0=0.16154,
    x1=-0.047986,
    x2=0.027170,
    x3=0.13611,
    alpha=0.14416,
)


# ---- LNS EoS - parameters are taken from Cao et al. (2006) ---- #

eos_LNS = ec.EquationOfState(
    t0=-2484.97,
    t1=266.735,
    t2=-337.135,
    t3=14588.2,
    x0=0.06277,
    x1=0.65845,
    x2=-0.95382,
    x3=-0.03413,
    alpha=0.16667,
)


# ---- SLy4 - parameters are taken from Chabanat et al. (1998) ---- #

eos_SLy4 = ec.EquationOfState(
    t0=-2488.91,
    t1=486.82,
    t2=-546.39,
    t3=13777.0,
    x0=0.834,
    x1=-0.344,
    x2=-1,
    x3=1.354,
    alpha=0.14416,
)
