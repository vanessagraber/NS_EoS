"""Calculation of the superfluid neutron and superconducting proton gap in the neutron star core following the
    parametrisation introduced in Andersson et al. (2005) and used in Ho et al. (2012)
"""

import numpy as np
import ns_eos.equilibrium_comp as ec

# fit parameters for the two superfluids

proton_singlet = {"Delta_0": 120, "k_1": 0, "k_2": 9, "k_3": 1.3, "k_4": 1.8}
neutron_triplet = {"Delta_0": 0.068, "k_1": 1.28, "k_2": 0.1, "k_3": 2.37, "k_4": 0.02}


def gap_neutrons(k_F_n: float) -> float:
    """function calculates the neutron energy gap in MeV as a function of neutron Fermi number in 1/fm """

    Delta_n = (
        neutron_triplet["Delta_0"]
        * (k_F_n - neutron_triplet["k_1"]) ** 2
        / ((k_F_n - neutron_triplet["k_1"]) ** 2 + neutron_triplet["k_2"])
        * (k_F_n - neutron_triplet["k_3"]) ** 2
        / ((k_F_n - neutron_triplet["k_3"]) ** 2 + neutron_triplet["k_4"])
    )

    return Delta_n


def gap_protons(k_F_p: float) -> float:
    """function calculates the proton energy gap in MeV as a function of proton Fermi number in 1/fm """

    Delta_p = (
        proton_singlet["Delta_0"]
        * (k_F_p - proton_singlet["k_1"]) ** 2
        / ((k_F_p - proton_singlet["k_1"]) ** 2 + proton_singlet["k_2"])
        * (k_F_p - proton_singlet["k_3"]) ** 2
        / ((k_F_p - proton_singlet["k_3"]) ** 2 + proton_singlet["k_4"])
    )

    return Delta_p
