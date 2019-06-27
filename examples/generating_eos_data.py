"""Generating particle fractions and effective masses for several sets of equation of state parameters
"""

import numpy as np
import pandas as pd
import ns_eos.equilibrium_comp as ec


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


# baryon number density and mass energy density in units of 1/fm**3 and g/cm**3, respectively
n_b = np.arange(0.06, 0.6, 0.01)
rho_b = m_u_cgs * n_b / fm ** 3


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

n_n_NRAPR = eos_NRAPR.n_n(n_b)
n_p_NRAPR = eos_NRAPR.n_p(n_b)
lambda_NRAPR = eos_NRAPR.lambda_L(n_b)
xi_n_NRAPR = eos_NRAPR.xi_n(n_b)
xi_p_NRAPR = eos_NRAPR.xi_p(n_b)
H_i_NRAPR = eos_NRAPR.H_parameters()

print("H_i for NRAPR: ", H_i_NRAPR)

df_NRAPR = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_NRAPR,
        "n_p": n_p_NRAPR,
        "lambda_eff": lambda_NRAPR,
        "xi_n": xi_n_NRAPR,
        "xi_p": xi_p_NRAPR,
        "kappa": lambda_NRAPR / xi_p_NRAPR,
        "R": xi_p_NRAPR / xi_n_NRAPR,
        "epsilon": n_p_NRAPR / n_n_NRAPR,
    }
)

df_NRAPR.columns = pd.MultiIndex.from_tuples(
    zip(
        df_NRAPR.columns,
        [
            "[1/fm**3]",
            "[g/cm**3]",
            "[1/fm**3]",
            "[1/fm**3]",
            "[cm]",
            "[cm]",
            "[cm]",
            "[]",
            "[]",
            "[]",
        ],
    )
)
df_NRAPR.to_csv("./examples/data/parameters_NRAPR.txt", index=False, header=True)


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

n_n_LNS = eos_LNS.n_n(n_b)
n_p_LNS = eos_LNS.n_p(n_b)
lambda_LNS = eos_LNS.lambda_L(n_b)
xi_n_LNS = eos_LNS.xi_n(n_b)
xi_p_LNS = eos_LNS.xi_p(n_b)
H_i_LNS = eos_LNS.H_parameters()

print("H_i for LNS: ", H_i_LNS)

df_LNS = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_LNS,
        "n_p": n_p_LNS,
        "lambda_eff": lambda_LNS,
        "xi_n": xi_n_LNS,
        "xi_p": xi_p_LNS,
        "kappa": lambda_LNS / xi_p_LNS,
        "R": xi_p_LNS / xi_n_LNS,
        "epsilon": n_p_LNS / n_n_LNS,
    }
)

df_LNS.columns = pd.MultiIndex.from_tuples(
    zip(
        df_LNS.columns,
        [
            "[1/fm**3]",
            "[g/cm**3]",
            "[1/fm**3]",
            "[1/fm**3]",
            "[cm]",
            "[cm]",
            "[cm]",
            "[]",
            "[]",
            "[]",
        ],
    )
)
df_LNS.to_csv("./examples/data/parameters_LNS.txt", index=None, header=True)


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

n_n_SLy4 = eos_SLy4.n_n(n_b)
n_p_SLy4 = eos_SLy4.n_p(n_b)
lambda_SLy4 = eos_SLy4.lambda_L(n_b)
xi_n_SLy4 = eos_SLy4.xi_n(n_b)
xi_p_SLy4 = eos_SLy4.xi_p(n_b)
H_i_SLy4 = eos_SLy4.H_parameters()

print("H_i for SLy4: ", H_i_SLy4)

df_SLy4 = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_SLy4,
        "n_p": n_p_SLy4,
        "lambda": lambda_SLy4,
        "xi_n": xi_n_SLy4,
        "xi_p": xi_p_SLy4,
        "kappa": lambda_SLy4 / xi_p_SLy4,
        "R": xi_p_SLy4 / xi_n_SLy4,
        "epsilon": n_p_SLy4 / n_n_SLy4,
    }
)

df_SLy4.columns = pd.MultiIndex.from_tuples(
    zip(
        df_SLy4.columns,
        [
            "[1/fm**3]",
            "[g/cm**3]",
            "[1/fm**3]",
            "[1/fm**3]",
            "[cm]",
            "[cm]",
            "[cm]",
            "[]",
            "[]",
            "[]",
        ],
    )
)
df_SLy4.to_csv("./examples/data/parameters_SLy4.txt", index=None, header=True)
