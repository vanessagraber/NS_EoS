"""Calculation of the superfluid neutron and superconducting proton gap in the neutron star core following the
    parametrisation introduced in Andersson et al. (2005) with the parameters given in Ho et al. (2015)
"""

from scipy.optimize import newton
import numpy as np

# fit parameters for the two superfluids using the CCDK gap for the protons and the TToa gap for the neutrons
proton_singlet = {"Delta_0": 102, "k_1": 0, "k_2": 9, "k_3": 1.3, "k_4": 1.5}
neutron_triplet = {"Delta_0": 2.1, "k_1": 1.1, "k_2": 0.6, "k_3": 3.2, "k_4": 2.4}


def gap_neutrons_full(k_F_n: float) -> float:
    """function calculates the parametrised neutron energy gap in MeV for the full range (not cut off)
        of neutron Fermi number in 1/fm """

    Delta_n = (
        neutron_triplet["Delta_0"]
        * (k_F_n - neutron_triplet["k_1"]) ** 2
        / ((k_F_n - neutron_triplet["k_1"]) ** 2 + neutron_triplet["k_2"])
        * (k_F_n - neutron_triplet["k_3"]) ** 2
        / ((k_F_n - neutron_triplet["k_3"]) ** 2 + neutron_triplet["k_4"])
    )

    return Delta_n


def gap_neutrons(k_F_n: float) -> float:
    """function determines the neutron energy gap in MeV as a function of neutron Fermi number in 1/fm """

    limit_left = newton(gap_neutrons_full, 1.1)
    limit_right = newton(gap_neutrons_full, 3.5)

    if k_F_n > limit_right:
        return np.nan
    else:
        if k_F_n < limit_left:
            return np.nan
        else:
            return gap_neutrons_full(k_F_n)


def gap_protons_full(k_F_p: float) -> float:
    """function calculates the parametrised proton energy gap in MeV for the full range (not cut off)
        of neutron Fermi number in 1/fm """

    Delta_p = (
        proton_singlet["Delta_0"]
        * (k_F_p - proton_singlet["k_1"]) ** 2
        / ((k_F_p - proton_singlet["k_1"]) ** 2 + proton_singlet["k_2"])
        * (k_F_p - proton_singlet["k_3"]) ** 2
        / ((k_F_p - proton_singlet["k_3"]) ** 2 + proton_singlet["k_4"])
    )

    return Delta_p


def gap_protons(k_F_p: float) -> float:
    """function determines the proton energy gap in MeV as a function of proton Fermi number in 1/fm """

    limit_left = newton(gap_protons_full, 0.01)
    limit_right = newton(gap_protons_full, 1.5)

    if k_F_p > limit_right:
        return np.nan
    else:
        if k_F_p < limit_left:
            return np.nan
        else:
            return gap_protons_full(k_F_p)
