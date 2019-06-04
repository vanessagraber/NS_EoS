"""Calculation of chemical potentials based on baryon conservation, charge neutrality, beta equilibrium
    and muon production rate for a given set of Skyrme parameters as done in Chamel (2008)
"""

import numpy as np
from scipy.optimize import newton
from typing import Tuple

# natural constants
c = 2.9979246e23  # speed of light in fm/s
hbar = 6.582120e-22  # hbar in MeV s
m_u = 1.0364270e-44  # atomic mass unit in MeV s^2/fm^2
m_mu = 0.113429 * m_u  # muon mass unit in MeV s^2/fm^2

# constant related to the appearance of muon
const = m_mu ** 2 * c ** 2 / (hbar ** 2 * (3 * np.pi ** 2) ** (2 / 3))


def relation_nmu_ne(n_e: float) -> float:
    """function relates the muon number density to a given electron number density;
    calculated from the equality of the muon and electron chemical potentials"""

    # for low electron number densities, no muons are present
    if n_e < const ** (3 / 2):
        return 0.0
    # above a critical electron number density muons appear
    else:
        n_mu = (n_e ** (2 / 3) - const) ** (3 / 2)
        return n_mu


def relation_np_ne(n_e: float) -> float:
    """function calculates the proton number density for a given electron number density;
    obtained from the charge neutrality condition"""

    # below first appearance of muons, electron and proton number density are equivalent
    if n_e < const ** (3 / 2):
        return n_e
    # above the critical electron number density, muons contribute
    else:
        return relation_nmu_ne(n_e) + n_e
