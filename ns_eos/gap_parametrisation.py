"""
Calculation of the superfluid neutron and superconducting proton gap
in the neutron star core following the parametrisation introduced in
Andersson et al. (2005) with the parameters given in Ho et al. (2015)

Authors:

        Vanessa Graber (graber@ice.csic.es)

    Copyright (c) Vanessa Graber
"""

from scipy.optimize import newton
from typing import TypedDict
import numpy as np

# fit parameters for the crustal 1S0 neutron gap
neutron_singlet_AWP2 = {
    "Delta_0": 28.00,
    "k_1": 0.20,
    "k_2": 1.50,
    "k_3": 1.70,
    "k_4": 2.50,
}
neutron_singlet_AWP3 = {
    "Delta_0": 50.00,
    "k_1": 0.20,
    "k_2": 2.00,
    "k_3": 1.40,
    "k_4": 2.00,
}
neutron_singlet_CCDK = {
    "Delta_0": 127.00,
    "k_1": 0.18,
    "k_2": 4.50,
    "k_3": 1.08,
    "k_4": 1.10,
}
neutron_singlet_CLS = {
    "Delta_0": 2.20,
    "k_1": 0.18,
    "k_2": 0.06,
    "k_3": 1.30,
    "k_4": 0.03,
}
neutron_singlet_GIPSF = {
    "Delta_0": 8.80,
    "k_1": 0.18,
    "k_2": 0.10,
    "k_3": 1.20,
    "k_4": 0.60,
}
neutron_singlet_MSH = {
    "Delta_0": 2.45,
    "k_1": 0.18,
    "k_2": 0.05,
    "k_3": 1.40,
    "k_4": 0.10,
}
neutron_singlet_SCLBL = {
    "Delta_0": 4.10,
    "k_1": 0.35,
    "k_2": 1.70,
    "k_3": 1.67,
    "k_4": 0.06,
}
neutron_singlet_SFB = {
    "Delta_0": 45.00,
    "k_1": 0.10,
    "k_2": 4.50,
    "k_3": 1.55,
    "k_4": 2.50,
}
neutron_singlet_WAP = {
    "Delta_0": 69.00,
    "k_1": 0.15,
    "k_2": 3.00,
    "k_3": 1.40,
    "k_4": 3.00,
}

# fit parameters for the core 1S0 proton gap
proton_singlet_AO = {
    "Delta_0": 14.00,
    "k_1": 0.15,
    "k_2": 0.22,
    "k_3": 1.05,
    "k_4": 3.80,
}
proton_singlet_BCLL = {
    "Delta_0": 1.69,
    "k_1": 0.05,
    "k_2": 0.07,
    "k_3": 1.05,
    "k_4": 0.16,
}
proton_singlet_BS = {
    "Delta_0": 17.00,
    "k_1": 0.00,
    "k_2": 2.90,
    "k_3": 0.80,
    "k_4": 0.08,
}
proton_singlet_CCDK = {
    "Delta_0": 102.00,
    "k_1": 0.00,
    "k_2": 9.00,
    "k_3": 1.30,
    "k_4": 1.50,
}
proton_singlet_CCYms = {
    "Delta_0": 35.00,
    "k_1": 0.00,
    "k_2": 5.00,
    "k_3": 1.10,
    "k_4": 0.50,
}
proton_singlet_CCYps = {
    "Delta_0": 34.00,
    "k_1": 0.00,
    "k_2": 5.00,
    "k_3": 0.95,
    "k_4": 0.30,
}
proton_singlet_EEHO = {
    "Delta_0": 4.50,
    "k_1": 0.00,
    "k_2": 0.57,
    "k_3": 1.20,
    "k_4": 0.35,
}
proton_singlet_EEHOr = {
    "Delta_0": 61.00,
    "k_1": 0.00,
    "k_2": 6.00,
    "k_3": 1.10,
    "k_4": 0.60,
}
proton_singlet_T = {
    "Delta_0": 48.00,
    "k_1": 0.15,
    "k_2": 2.10,
    "k_3": 1.20,
    "k_4": 2.80,
}

# fit parameters for the core 3P2 neutron gap
neutron_triplet_AO = {
    "Delta_0": 4.00,
    "k_1": 1.20,
    "k_2": 0.450,
    "k_3": 3.30,
    "k_4": 5.0000,
}
neutron_triplet_BEEHS = {
    "Delta_0": 0.45,
    "k_1": 1.00,
    "k_2": 0.400,
    "k_3": 3.20,
    "k_4": 0.2500,
}
neutron_triplet_EEHO = {
    "Delta_0": 0.48,
    "k_1": 1.28,
    "k_2": 0.100,
    "k_3": 2.37,
    "k_4": 0.0200,
}
neutron_triplet_EEHOr = {
    "Delta_0": 0.23,
    "k_1": 1.20,
    "k_2": 0.026,
    "k_3": 1.60,
    "k_4": 0.0080,
}
neutron_triplet_SYHHP = {
    "Delta_0": 1.00,
    "k_1": 2.08,
    "k_2": 0.040,
    "k_3": 2.70,
    "k_4": 0.0130,
}
neutron_triplet_T = {
    "Delta_0": 1.20,
    "k_1": 1.55,
    "k_2": 0.050,
    "k_3": 2.35,
    "k_4": 0.0700,
}
neutron_triplet_TTav = {
    "Delta_0": 3.00,
    "k_1": 1.10,
    "k_2": 0.600,
    "k_3": 2.92,
    "k_4": 3.0000,
}
neutron_triplet_TToa = {
    "Delta_0": 2.10,
    "k_1": 1.10,
    "k_2": 0.600,
    "k_3": 3.20,
    "k_4": 2.4000,
}


def gap_full(k_F: float, gap_parameters: TypedDict) -> float:
    """
    function calculates the parametrised energy gap in MeV
    for the full range (not cut off) of neutron Fermi number in 1/fm
    """

    Delta = (
        gap_parameters["Delta_0"]
        * (k_F - gap_parameters["k_1"]) ** 2
        / ((k_F - gap_parameters["k_1"]) ** 2 + gap_parameters["k_2"])
        * (k_F - gap_parameters["k_3"]) ** 2
        / ((k_F - gap_parameters["k_3"]) ** 2 + gap_parameters["k_4"])
    )

    return Delta


def gap_singlet_neutrons(k_F_n: float, gap_parameters: TypedDict) -> float:
    """function determines the neutron singlet energy gap in MeV as a function
    of proton Fermi number in 1/fm"""

    limit_left = newton(gap_full, 0.01, args=(gap_parameters,))
    limit_right = newton(gap_full, 0.9, args=(gap_parameters,))

    if k_F_n > limit_right:
        return np.nan
    else:
        if k_F_n < limit_left:
            return np.nan
        else:
            return gap_full(k_F_n, gap_parameters)


def gap_singlet_protons(k_F_p: float, gap_parameters: TypedDict) -> float:
    """function determines the proton singlet energy gap in MeV as a function
    of proton Fermi number in 1/fm"""

    limit_left = newton(gap_full, 0.01, args=(gap_parameters,))
    limit_right = newton(gap_full, 1.5, args=(gap_parameters,))

    if k_F_p > limit_right:
        return np.nan
    else:
        if k_F_p < limit_left:
            return np.nan
        else:
            return gap_full(k_F_p, gap_parameters)


def gap_triplet_neutrons(k_F_n: float, gap_parameters: TypedDict) -> float:
    """
    function determines the neutron triplet energy gap in MeV as a function
    of neutron Fermi number in 1/fm
    """

    limit_left = newton(gap_full, 1.1, args=(gap_parameters,))
    limit_right = newton(gap_full, 3.5, args=(gap_parameters,))

    if k_F_n > limit_right:
        return np.nan
    else:
        if k_F_n < limit_left:
            return np.nan
        else:
            return gap_full(k_F_n, gap_parameters)
