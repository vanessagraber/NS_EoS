"""
Calculation of the neutron star composition based on baryon conservation,
charge neutrality, beta equilibrium and muon production rate for a given
set of Skyrme parameters as done in Chamel (2008). The superfluid neutron
and superconducting proton gap in the neutron star core are based on the
parametrisation introduced in Andersson et al. (2005) with parameters from
Ho et al. (2015).

Authors:

        Vanessa Graber (graber@ice.csic.es)

    Copyright (c) Vanessa Graber
"""

import numpy as np
from scipy.optimize import newton
from typing import Tuple
import ns_eos.gap_parametrisation as gp

# natural constants
c = 2.997925e23  # speed of light in fm/s
hbar = 6.582120e-22  # hbar in MeV s
m_u = 1.036427e-44  # atomic mass unit in MeV s**2/fm**2
m_u_cgs = 1.660539e-24  # atomic mass unit in g
m_mu = 0.113429 * m_u  # muon mass unit in MeV s**2/fm**2
q = 1.199985  # electric charge in (MeV fm)**1/2

# unit conversion factors
fm = 1e-13  # fm to cm
MeV = 1e6 * 1.782662e-33 * (c * fm) ** 2  # MeV to g*cm**2/s**2

# constant in units of 1/fm**2 related to the appearance of muon
muon_eqn_const = m_mu**2 * c**2 / (hbar**2 * (3 * np.pi**2) ** (2 / 3))


def relation_nmu_ne(n_e: float) -> float:
    """
    function relates the muon number density in 1/fm**3 to a given
    electron number density in 1/fm**3; calculated from the equality of
    the muon and electron chemical potentials
    """

    # for low electron number densities, no muons are present
    if n_e < muon_eqn_const ** (3 / 2):
        return 0.0
    # above a critical electron number density muons appear
    else:
        n_mu = (n_e ** (2 / 3) - muon_eqn_const) ** (3 / 2)
        return n_mu


def relation_np_ne(n_e: float) -> float:
    """
    function calculates the proton number density in 1/fm**3 for a
    given electron number density in 1/fm**3; obtained from the charge
    neutrality condition
    """

    # below first appearance of muons, electron and proton number density are equivalent
    if n_e < muon_eqn_const ** (3 / 2):
        return n_e
    # above the critical electron number density, muons contribute
    else:
        return relation_nmu_ne(n_e) + n_e


class EquationOfState:
    def __init__(
        self,
        t0=-2719.7,
        t1=417.64,
        t2=-66.687,
        t3=15042.0,
        t4=0.0,
        x0=0.16154,
        x1=-0.047986,
        x2=0.027170,
        x3=0.13611,
        x4=0.0,
        sigma=0.14416,
        sigma_2=0.0,
        W0=41.958,
    ):
        """
        predefined parameters are for the NRAPR equation of state

        :param t0: Skyrme parameter
        :type t0: float
        :param t1: Skyrme parameter
        :type t1: float
        :param t2: Skyrme parameter
        :type t2: float
        :param t3: Skyrme parameter
        :type t3: float
        :param t4: Skyrme parameter
        :type t4: float
        :param x0: Skyrme parameter
        :type x0: float
        :param x1: Skyrme parameter
        :type x1: float
        :param x2: Skyrme parameter
        :type x2: float
        :param x3: Skyrme parameter
        :type x3: float
        :param x4: Skyrme parameter
        :type x4: float
        :param sigma: Skyrme parameter
        :type sigma: float
        :param sigma_2: Skyrme parameter
        :type sigma_2: float
        :param W0: Skyrme parameter
        :type W0: float
        """

        # Skyrme parameters
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.sigma = sigma
        self.sigma_2 = sigma_2
        self.W0 = W0

        # modified Skyrme parameters
        self.C_0_tau = 3 * self.t1 / 16 + 1 * self.t2 / 4 * (5 / 4 + self.x2)
        self.C_1_tau = -1 * self.t1 / 8 * (1 / 2 + self.x1) + 1 * self.t2 / 8 * (
            1 / 2 + self.x2
        )
        self.C_0_Delta_n = -9 * self.t1 / 64 + 1 * self.t2 / 16 * (5 / 4 + self.x2)
        self.C_1_Delta_n = 3 * self.t1 / 32 * (1 / 2 + self.x1) + 1 * self.t2 / 32 * (
            1 / 2 + self.x2
        )
        self.C_0_Delta_J = -3 * self.W0 / 4
        self.C_1_Delta_J = -1 * self.W0 / 4

    def _parameters_hamiltonian(self) -> Tuple[float, ...]:
        """
        function calculates the parameters for the effective Skyrme Hamiltonian
        """

        B1 = (self.t0 / 2) * (1 + self.x0 / 2)
        B2 = -(self.t0 / 2) * (self.x0 + 1 / 2)
        B3 = (1 / 4) * (self.t1 * (1 + self.x1 / 2) + self.t2 * (1 + self.x2 / 2))
        B4 = -(1 / 4) * (self.t1 * (self.x1 + 1 / 2) - self.t2 * (self.x2 + 1 / 2))
        B5 = (self.t3 / 12) * (1 + self.x3 / 2)
        B6 = -(self.t3 / 12) * (self.x3 + 1 / 2)
        B7 = (self.t4 / 12) * (1 + self.x4 / 2)
        B8 = -(self.t4 / 12) * (self.x4 + 1 / 2)

        return B1, B2, B3, B4, B5, B6, B7, B8

    def H_parameters(self) -> Tuple[float, ...]:
        """
        function calculates the three H_i parameters of the energy
        density functional
        """

        H_1 = self.C_0_tau - self.C_1_tau
        H_2 = -4 * self.C_0_Delta_n + 4 * self.C_1_Delta_n
        H_3 = self.C_0_tau + self.C_1_tau - 4 * self.C_0_Delta_n - 4 * self.C_1_Delta_n

        return H_1, H_2, H_3

    def relative_chem_pot(self, n_b: float, n_p: float) -> float:
        """
        function determines the relative chemical potential in MeV of the
        protons and neutrons as a function of baryon and proton number
        densities (in 1/fm**3) using the parameters of the Skyrme Hamiltonian
        """

        b_vector = self._parameters_hamiltonian()

        del_mu = (
            (3 * np.pi**2) ** (2 / 3)
            * ((n_b - n_p) ** (2 / 3) - n_p ** (2 / 3))
            * (hbar**2 / (2 * m_u) + b_vector[2] * n_b)
            + 2
            * (n_b - 2 * n_p)
            * (
                b_vector[1]
                + b_vector[5] * n_b**self.sigma
                + b_vector[7] * n_b**self.sigma_2
            )
            + (8 / 5)
            * (3 * np.pi**2) ** (2 / 3)
            * b_vector[3]
            * ((n_b - n_p) ** (5 / 3) - n_p ** (5 / 3))
        )
        return del_mu

    def func_to_minimize(self, n_e: float, n_b: float) -> float:
        """
        based on beta equilibrium, the function equates the relative
        nucleon chemical potentials with the electron chemical potential
        for given electron and baryon number density in 1/fm**3
        """

        n_p = relation_np_ne(n_e)
        del_mu = self.relative_chem_pot(n_b, n_p)
        mu_e = c * hbar * (3 * np.pi**2 * n_e) ** (1 / 3)

        func_min = del_mu - mu_e

        return func_min

    def _relation_ne_nb(self, n_b: float) -> float:
        """
        function solves for the electron number density in 1/fm**3 for
        any given baryon number density in 1/fm**3
        """

        output = newton(self.func_to_minimize, 0.001, args=(n_b,))

        return output

    def relation_ne_nb(self, n_b: np.ndarray) -> np.ndarray:
        """
        function takes a numpy array of baryon number densities in 1/fm**3
        and calculates the electron number density in 1/fm**3
        """

        vect_func = np.vectorize(self._relation_ne_nb)

        n_e = vect_func(n_b)

        return n_e

    # particle fractions

    def x_e(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the electron fraction for any given
        baryon number density in 1/fm**3
        """

        n_e = self.relation_ne_nb(n_b)
        x_e = n_e / n_b

        return x_e

    def x_p(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the proton fraction for any given
        baryon number density in 1/fm**3
        """

        n_e = self.relation_ne_nb(n_b)
        vect_func = np.vectorize(relation_np_ne)
        x_p = vect_func(n_e) / n_b

        return x_p

    def x_mu(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the muon fraction for any given baryon
        number density in 1/fm**3
        """

        n_e = self.relation_ne_nb(n_b)
        vect_func = np.vectorize(relation_nmu_ne)
        x_mu = vect_func(n_e) / n_b

        return x_mu

    # number densities

    def n_p(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the proton number density in 1/fm**3
        for any given baryon number density in 1/fm**3
        """

        x_p = self.x_p(n_b)
        n_p = n_b * x_p

        return n_p

    def n_n(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the neutron number density in 1/fm**3
        for any given baryon number density in 1/fm**3
        """

        n_n = n_b - self.n_p(n_b)

        return n_n

    # dynamic effective masses caused by entrainment

    def m_eff_n(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the neutron effective mass in gram for a
        given baryon number density in 1/fm**3
        """

        B3 = self._parameters_hamiltonian()[2]
        beta_3 = (2 * m_u * B3) / (hbar**2)
        x_p = self.x_p(n_b)
        m_eff_n = m_u_cgs * (1 + beta_3 * n_b * (1 - x_p)) / (1 + beta_3 * n_b)

        return m_eff_n

    def m_eff_p(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the proton effective mass in gram for a
         baryon number density in 1/fm**3
        """

        B3 = self._parameters_hamiltonian()[2]
        beta_3 = (2 * m_u * B3) / (hbar**2)
        x_p = self.x_p(n_b)
        m_eff_p = m_u_cgs * (1 + beta_3 * n_b * x_p) / (1 + beta_3 * n_b)

        return m_eff_p

    # Landau effective masses characterising the static ground state

    def m_eff_L_n(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the neutron Landau effective mass in gram
        for a given baryon number density in 1/fm**3
        """

        m_eff_L_n = (
            1 / m_u_cgs
            + 2
            / hbar**2
            * fm**2
            / MeV
            * (n_b * (self.C_0_tau - self.C_1_tau) + 2 * self.n_n(n_b) * self.C_1_tau)
        ) ** (-1)

        return m_eff_L_n

    def m_eff_L_p(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the proton Landau effective mass in gram
        for a given baryon number density in 1/fm**3
        """

        m_eff_L_p = (
            1 / m_u_cgs
            + 2
            / hbar**2
            * fm**2
            / MeV
            * (n_b * (self.C_0_tau - self.C_1_tau) + 2 * self.n_p(n_b) * self.C_1_tau)
        ) ** (-1)

        return m_eff_L_p

    # Fermi wave numbers

    def k_F_n(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the neutron Fermi wave vector in 1/fm for a
        given baryon number density in 1/fm**3
        """

        k_F_n = (3 * np.pi**2 * self.n_n(n_b)) ** (1 / 3)

        return k_F_n

    def k_F_p(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the proton Fermi wave vector in 1/fm for a
        given baryon number density in 1/fm**3
        """

        k_F_p = (3 * np.pi**2 * self.n_p(n_b)) ** (1 / 3)

        return k_F_p

    # characteristic length scales

    def lambda_L(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the London penetration depth in cm for a
        given baryon number density in 1/fm**3
        """

        lambda_L = ((m_u * c**2) / (q**2 * 4 * np.pi * self.n_p(n_b))) ** (
            1 / 2
        ) * fm

        return lambda_L

    def xi_n(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the neutron coherence length in cm for a
        given baryon number density in 1/fm**3
        """

        v_F_n = hbar * self.k_F_n(n_b) / self.m_eff_L_n(n_b) * MeV / fm
        vect_func = np.vectorize(gp.gap_neutrons)
        Delta_n = vect_func(self.k_F_n(n_b))
        xi_n = hbar * v_F_n / (np.pi * Delta_n)

        return xi_n

    def xi_p(self, n_b: np.ndarray) -> np.ndarray:
        """
        function calculates the proton coherence length in cm for a
        given baryon number density in 1/fm**3
        """

        v_F_p = hbar * self.k_F_p(n_b) / self.m_eff_L_p(n_b) * MeV / fm
        vect_func = np.vectorize(gp.gap_protons)
        Delta_p = vect_func(self.k_F_p(n_b))
        xi_p = hbar * v_F_p / (np.pi * Delta_p)

        return xi_p
