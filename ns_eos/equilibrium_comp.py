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
muon_eqn_const = m_mu ** 2 * c ** 2 / (hbar ** 2 * (3 * np.pi ** 2) ** (2 / 3))


def relation_nmu_ne(n_e: float) -> float:
    """function relates the muon number density to a given electron number density;
    calculated from the equality of the muon and electron chemical potentials"""

    # for low electron number densities, no muons are present
    if n_e < muon_eqn_const ** (3 / 2):
        return 0.0
    # above a critical electron number density muons appear
    else:
        n_mu = (n_e ** (2 / 3) - muon_eqn_const) ** (3 / 2)
        return n_mu


def relation_np_ne(n_e: float) -> float:
    """function calculates the proton number density for a given electron number density;
    obtained from the charge neutrality condition"""

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
        x0=0.16154,
        x1=-0.047986,
        x2=0.027170,
        x3=0.13611,
        alpha=0.14416,
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
        :param x0: Skyrme parameter
        :type x0: float
        :param x1: Skyrme parameter
        :type x1: float
        :param x2: Skyrme parameter
        :type x2: float
        :param x3: Skyrme parameter
        :type x3: float
        :param alpha: Skyrme parameter
        :type alpha: float
        """

        # Skyrme parameters
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.alpha = alpha

    def _parameters_hamiltonian(self) -> Tuple[float, ...]:
        """function calculates the parameters for the effective Skyrme Hamiltonian"""

        B1 = (self.t0 / 2) * (1 + self.x0 / 2)
        B2 = -(self.t0 / 2) * (self.x0 + 1 / 2)
        B3 = (1 / 4) * (self.t1 * (1 + self.x1 / 2) + self.t2 * (1 + self.x2 / 2))
        B4 = -(1 / 4) * (self.t1 * (self.x1 + 1 / 2) - self.t2 * (self.x2 + 1 / 2))
        B5 = (self.t3 / 12) * (1 + self.x3 / 2)
        B6 = -(self.t3 / 12) * (self.x3 + 1 / 2)

        return B1, B2, B3, B4, B5, B6

    def relative_chem_pot(self, n_b: float, n_p: float) -> float:
        """function determines the relative chemical potential of the protons and neutrons as a
        function of baryon and proton number densities using the parameters of the Skyrme Hamiltonian"""

        b_vector = self._parameters_hamiltonian()

        del_mu = (
            (3 * np.pi ** 2) ** (2 / 3)
            * ((n_b - n_p) ** (2 / 3) - n_p ** (2 / 3))
            * (hbar ** 2 / (2 * m_u) + b_vector[2] * n_b)
            + 2 * (n_b - 2 * n_p) * (b_vector[1] + b_vector[5] * n_b ** self.alpha)
            + (8 / 5)
            * (3 * np.pi ** 2) ** (2 / 3)
            * b_vector[3]
            * ((n_b - n_p) ** (5 / 3) - n_p ** (5 / 3))
        )
        return del_mu

    def func_to_minimize(self, n_e: float, n_b: float) -> float:
        """based on beta equilibrium, the function equates the relative nucleon chemical potentials
         with the electron chemical potential"""

        n_p = relation_np_ne(n_e)
        del_mu = self.relative_chem_pot(n_b, n_p)
        mu_e = c * hbar * (3 * np.pi ** 2 * n_e) ** (1 / 3)

        func_min = del_mu - mu_e

        return func_min

    def relation_ne_nb(self, n_b: np.ndarray) -> np.ndarray:
        """function solves for the electron number density for any given baryon number density"""

        vect_func = np.vectorize(self.func_to_minimize)

        output = np.array([newton(vect_func, 0.001, args=(n_b,))])

        return output

    def x_e(self, n_b: np.ndarray) -> np.ndarray:
        """function calculates the electron fraction for any given baryon number density"""

        n_e = self.relation_ne_nb(n_b)
        x_e = n_e/n_b

        return x_e

    def x_p(self, n_b: np.ndarray) -> np.ndarray:
        """function calculates the proton fraction for any given baryon number density"""

        n_e = self.relation_ne_nb(n_b)
        vect_func = np.vectorize(relation_np_ne)
        x_p = vect_func(n_e)/n_b

        return x_p

    def x_mu(self, n_b: np.ndarray) -> np.ndarray:
        """function calculates the muon fraction for any given baryon number density"""

        n_e = self.relation_ne_nb(n_b)
        vect_func = np.vectorize(relation_nmu_ne)
        x_mu = vect_func(n_e)/n_b

        return x_mu
