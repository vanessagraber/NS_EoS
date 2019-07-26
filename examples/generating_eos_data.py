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
    t0=-2719.70,
    t1=417.64,
    t2=-66.69,
    t3=15042.00,
    x0=0.1615,
    x1=-0.0480,
    x2=0.0272,
    x3=0.1361,
    sigma=0.1442,
)

n_n_NRAPR = eos_NRAPR.n_n(n_b)
n_p_NRAPR = eos_NRAPR.n_p(n_b)
lambda_NRAPR = eos_NRAPR.lambda_L(n_b)
xi_n_NRAPR = eos_NRAPR.xi_n(n_b)
xi_p_NRAPR = eos_NRAPR.xi_p(n_b)
H_i_NRAPR = eos_NRAPR.H_parameters()

print("H_i for NRAPR: ", [round(H, 2) for H in H_i_NRAPR])

df_NRAPR = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_NRAPR,
        "n_p": n_p_NRAPR,
        "lambda_eff": lambda_NRAPR,
        "xi_n": xi_n_NRAPR,
        "xi_p": xi_p_NRAPR,
        "kappa": lambda_NRAPR / xi_p_NRAPR,  # Ginzburg-Landau parameter
        "R": xi_p_NRAPR / xi_n_NRAPR,  # coherence length ratio
        "epsilon": n_p_NRAPR / n_n_NRAPR,  # asymmetry parameter
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
    t1=266.74,
    t2=-337.14,
    t3=14588.2,
    x0=0.0623,
    x1=0.6585,
    x2=-0.9538,
    x3=-0.0341,
    sigma=1 / 6,
)

n_n_LNS = eos_LNS.n_n(n_b)
n_p_LNS = eos_LNS.n_p(n_b)
lambda_LNS = eos_LNS.lambda_L(n_b)
xi_n_LNS = eos_LNS.xi_n(n_b)
xi_p_LNS = eos_LNS.xi_p(n_b)
H_i_LNS = eos_LNS.H_parameters()

print("H_i for LNS: ", [round(H, 2) for H in H_i_LNS])

df_LNS = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_LNS,
        "n_p": n_p_LNS,
        "lambda_eff": lambda_LNS,
        "xi_n": xi_n_LNS,
        "xi_p": xi_p_LNS,
        "kappa": lambda_LNS / xi_p_LNS,  # Ginzburg-Landau parameter
        "R": xi_p_LNS / xi_n_LNS,  # coherence length ratio
        "epsilon": n_p_LNS / n_n_LNS,  # asymmetry parameter
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
    sigma=1 / 6,
)

n_n_SLy4 = eos_SLy4.n_n(n_b)
n_p_SLy4 = eos_SLy4.n_p(n_b)
lambda_SLy4 = eos_SLy4.lambda_L(n_b)
xi_n_SLy4 = eos_SLy4.xi_n(n_b)
xi_p_SLy4 = eos_SLy4.xi_p(n_b)
H_i_SLy4 = eos_SLy4.H_parameters()

print("H_i for SLy4: ", [round(H, 2) for H in H_i_SLy4])

df_SLy4 = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_SLy4,
        "n_p": n_p_SLy4,
        "lambda": lambda_SLy4,
        "xi_n": xi_n_SLy4,
        "xi_p": xi_p_SLy4,
        "kappa": lambda_SLy4 / xi_p_SLy4,  # Ginzburg-Landau parameter
        "R": xi_p_SLy4 / xi_n_SLy4,  # coherence length ratio
        "epsilon": n_p_SLy4 / n_n_SLy4,  # asymmetry parameter
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


# ---- Ska35s20 - parameters are taken from Dutra et al. (2012) ---- #

eos_Sk = ec.EquationOfState(
    t0=-1768.8,
    t1=263.9,
    t2=-158.3,
    t3=12904.8,
    x0=0.13,
    x1=-0.8,
    x2=0.0,
    x3=0.01,
    sigma=0.35,
)

n_n_Sk = eos_Sk.n_n(n_b)
n_p_Sk = eos_Sk.n_p(n_b)
lambda_Sk = eos_Sk.lambda_L(n_b)
xi_n_Sk = eos_Sk.xi_n(n_b)
xi_p_Sk = eos_Sk.xi_p(n_b)
H_i_Sk = eos_Sk.H_parameters()

print("H_i for Ska35s20: ", [round(H, 2) for H in H_i_Sk])

df_Sk = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_Sk,
        "n_p": n_p_Sk,
        "lambda": lambda_Sk,
        "xi_n": xi_n_Sk,
        "xi_p": xi_p_Sk,
        "kappa": lambda_Sk / xi_p_Sk,  # Ginzburg-Landau parameter
        "R": xi_p_Sk / xi_n_Sk,  # coherence length ratio
        "epsilon": n_p_Sk / n_n_Sk,  # asymmetry parameter
    }
)

df_Sk.columns = pd.MultiIndex.from_tuples(
    zip(
        df_Sk.columns,
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
df_Sk.to_csv("./examples/data/parameters_Ska35s20.txt", index=None, header=True)


# ---- SQMC700 - parameters are taken from Guichon et al. (2006) ---- #

eos_SQMC = ec.EquationOfState(
    t0=-2429.1,
    t1=370.97,
    t2=-96.69,
    t3=13773.43,
    x0=0.10,
    x1=0.0,
    x2=0.0,
    x3=0.0,
    sigma=1 / 6,
)

n_n_SQMC = eos_SQMC.n_n(n_b)
n_p_SQMC = eos_SQMC.n_p(n_b)
lambda_SQMC = eos_SQMC.lambda_L(n_b)
xi_n_SQMC = eos_SQMC.xi_n(n_b)
xi_p_SQMC = eos_SQMC.xi_p(n_b)
H_i_SQMC = eos_SQMC.H_parameters()

print("H_i for SQMC700: ", [round(H, 2) for H in H_i_SQMC])

df_SQMC = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_SQMC,
        "n_p": n_p_SQMC,
        "lambda": lambda_SQMC,
        "xi_n": xi_n_SQMC,
        "xi_p": xi_p_SQMC,
        "kappa": lambda_SQMC / xi_p_SQMC,  # Ginzburg-Landau parameter
        "R": xi_p_SQMC / xi_n_SQMC,  # coherence length ratio
        "epsilon": n_p_SQMC / n_n_SQMC,  # asymmetry parameter
    }
)

df_SQMC.columns = pd.MultiIndex.from_tuples(
    zip(
        df_SQMC.columns,
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
df_SQMC.to_csv("./examples/data/parameters_SQMC700.txt", index=None, header=True)


# ---- Skchi450 - parameters are taken from Lim and Holt (2017) ---- #

eos_Skchi450 = ec.EquationOfState(
    t0=-1803.29,
    t1=301.82,
    t2=-273.28,
    t3=12783.86,
    t4=564.10,
    x0=0.443,
    x1=-0.3622,
    x2=-0.4105,
    x3=0.6545,
    x4=-11.316,
    sigma=1 / 3,
    sigma_2=1,
)

n_n_Skchi450 = eos_Skchi450.n_n(n_b)
n_p_Skchi450 = eos_Skchi450.n_p(n_b)
lambda_Skchi450 = eos_Skchi450.lambda_L(n_b)
xi_n_Skchi450 = eos_Skchi450.xi_n(n_b)
xi_p_Skchi450 = eos_Skchi450.xi_p(n_b)
H_i_Skchi450 = eos_Skchi450.H_parameters()

print("H_i for Skchi450: ", [round(H, 2) for H in H_i_Skchi450])

df_Skchi450 = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_Skchi450,
        "n_p": n_p_Skchi450,
        "lambda": lambda_Skchi450,
        "xi_n": xi_n_Skchi450,
        "xi_p": xi_p_Skchi450,
        "kappa": lambda_Skchi450 / xi_p_Skchi450,  # Ginzburg-Landau parameter
        "R": xi_p_Skchi450 / xi_n_Skchi450,  # coherence length ratio
        "epsilon": n_p_Skchi450 / n_n_Skchi450,  # asymmetry parameter
    }
)

df_Skchi450.columns = pd.MultiIndex.from_tuples(
    zip(
        df_Skchi450.columns,
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
df_Skchi450.to_csv("./examples/data/parameters_Skchi450.txt", index=None, header=True)