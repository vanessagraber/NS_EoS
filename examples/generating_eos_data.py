"""
Generating number densities, particle fractions, energy gaps, characteristic
length scales and dimensionless parameters for several sets of equation of
state parameters as a function of neutron star density

Authors:

        Vanessa Graber (graber@ice.csic.es)

    Copyright (c) Vanessa Graber
"""

import numpy as np
import pandas as pd
import ns_eos.gap_parametrisation as gp
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

# baryon number density and mass energy density in units of 1/fm**3 and g/cm**3
n_b = np.linspace(0.06, 0.6, 500)
rho_b = m_u_cgs * n_b / fm**3

# Fermi wave number to calculate the
k_F = np.arange(0, 3.5, 0.01)

# ---- Calculating the energy gaps as a function of k_F ---- #

gap_protons = np.vectorize(gp.gap_protons)
gap_neutrons = np.vectorize(gp.gap_neutrons)

df_gaps = pd.DataFrame(
    {"k_F": k_F, "Delta_p": gap_protons(k_F), "Delta_n": gap_neutrons(k_F)}
)

df_gaps.columns = pd.MultiIndex.from_tuples(
    zip(df_gaps.columns, ["[1/fm]", "[MeV]", "[MeV]"])
)
df_gaps.to_csv("./examples/data/energy_gaps.txt", index=False, header=True)


# ---- NRAPR EoS - parameters are taken from Steiner et al. (2005) ---- #

eos_NRAPR = ec.EquationOfState(
    t0=-2719.70,
    t1=417.64,
    t2=-66.687,
    t3=15042.00,
    x0=0.16154,
    x1=-0.047986,
    x2=0.027170,
    x3=0.13611,
    sigma=0.14416,
    W0=41.958,
)

n_n_NRAPR = eos_NRAPR.n_n(n_b)
n_p_NRAPR = eos_NRAPR.n_p(n_b)
k_n_NRAPR = eos_NRAPR.k_F_n(n_b)
k_p_NRAPR = eos_NRAPR.k_F_p(n_b)
Delta_n_NRAPR = gap_neutrons(k_n_NRAPR)
Delta_p_NRAPR = gap_protons(k_p_NRAPR)
lambda_NRAPR = eos_NRAPR.lambda_L(n_b)
xi_n_NRAPR = eos_NRAPR.xi_n(n_b)
xi_p_NRAPR = eos_NRAPR.xi_p(n_b)
meff_relL_n_NRAPR = eos_NRAPR.m_eff_L_n(n_b) / m_u_cgs
meff_relL_p_NRAPR = eos_NRAPR.m_eff_L_p(n_b) / m_u_cgs
A_nn_NRAPR, A_pp_NRAPR, A_np_NRAPR = eos_NRAPR.A_ii(n_b)
H_i_NRAPR = eos_NRAPR.H_parameters()

print("H_i for NRAPR: ", [round(H, 3) for H in H_i_NRAPR])

df_NRAPR = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_NRAPR,
        "n_p": n_p_NRAPR,
        "Delta_n": Delta_n_NRAPR,
        "Delta_p": Delta_p_NRAPR,
        "lambda": lambda_NRAPR,
        "xi_n": xi_n_NRAPR,
        "xi_p": xi_p_NRAPR,
        "kappa": lambda_NRAPR / xi_p_NRAPR,  # Ginzburg-Landau parameter
        "R": xi_p_NRAPR / xi_n_NRAPR,  # coherence length ratio
        "epsilon": n_p_NRAPR / n_n_NRAPR,  # asymmetry parameter
        "m_eff_relL_n": meff_relL_n_NRAPR,
        "m_eff_relL_p": meff_relL_p_NRAPR,
        "A_nn": A_nn_NRAPR,
        "A_pp": A_pp_NRAPR,
        "A_np": A_np_NRAPR,
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
            "[MeV]",
            "[MeV]",
            "[cm]",
            "[cm]",
            "[cm]",
            "[]",
            "[]",
            "[]",
            "[]",
            "[]",
            "[MeV fm**5]",
            "[MeV fm**5]",
            "[MeV fm**5]",
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
    sigma=1 / 6,
    sigma_2=0.0,
    W0=96.00,
)

n_n_LNS = eos_LNS.n_n(n_b)
n_p_LNS = eos_LNS.n_p(n_b)
lambda_LNS = eos_LNS.lambda_L(n_b)
xi_n_LNS = eos_LNS.xi_n(n_b)
xi_p_LNS = eos_LNS.xi_p(n_b)
meff_relL_n_LNS = eos_LNS.m_eff_L_n(n_b) / m_u_cgs
meff_relL_p_LNS = eos_LNS.m_eff_L_p(n_b) / m_u_cgs
A_nn_LNS, A_pp_LNS, A_np_LNS = eos_LNS.A_ii(n_b)
H_i_LNS = eos_LNS.H_parameters()

print("H_i for LNS: ", [round(H, 3) for H in H_i_LNS])

df_LNS = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_LNS,
        "n_p": n_p_LNS,
        "lambda": lambda_LNS,
        "xi_n": xi_n_LNS,
        "xi_p": xi_p_LNS,
        "kappa": lambda_LNS / xi_p_LNS,  # Ginzburg-Landau parameter
        "R": xi_p_LNS / xi_n_LNS,  # coherence length ratio
        "epsilon": n_p_LNS / n_n_LNS,  # asymmetry parameter
        "m_eff_relL_n": meff_relL_n_LNS,
        "m_eff_relL_p": meff_relL_p_LNS,
        "A_nn": A_nn_LNS,
        "A_pp": A_pp_LNS,
        "A_np": A_np_LNS,
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
            "[]",
            "[]",
            "[MeV fm**5]",
            "[MeV fm**5]",
            "[MeV fm**5]",
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
    sigma_2=0.0,
    W0=123.0,
)

n_n_SLy4 = eos_SLy4.n_n(n_b)
n_p_SLy4 = eos_SLy4.n_p(n_b)
k_n_SLy4 = eos_SLy4.k_F_n(n_b)
k_p_SLy4 = eos_SLy4.k_F_p(n_b)
Delta_n_SLy4 = gap_neutrons(k_n_SLy4)
Delta_p_SLy4 = gap_protons(k_p_SLy4)
lambda_SLy4 = eos_SLy4.lambda_L(n_b)
xi_n_SLy4 = eos_SLy4.xi_n(n_b)
xi_p_SLy4 = eos_SLy4.xi_p(n_b)
meff_relL_n_SLy4 = eos_SLy4.m_eff_L_n(n_b) / m_u_cgs
meff_relL_p_SLy4 = eos_SLy4.m_eff_L_p(n_b) / m_u_cgs
A_nn_SLy4, A_pp_SLy4, A_np_SLy4 = eos_SLy4.A_ii(n_b)
H_i_SLy4 = eos_SLy4.H_parameters()

print("H_i for SLy4: ", [round(H, 3) for H in H_i_SLy4])

df_SLy4 = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_SLy4,
        "n_p": n_p_SLy4,
        "Delta_n": Delta_n_SLy4,
        "Delta_p": Delta_p_SLy4,
        "lambda": lambda_SLy4,
        "xi_n": xi_n_SLy4,
        "xi_p": xi_p_SLy4,
        "kappa": lambda_SLy4 / xi_p_SLy4,  # Ginzburg-Landau parameter
        "R": xi_p_SLy4 / xi_n_SLy4,  # coherence length ratio
        "epsilon": n_p_SLy4 / n_n_SLy4,  # asymmetry parameter
        "m_eff_relL_n": meff_relL_n_SLy4,
        "m_eff_relL_p": meff_relL_p_SLy4,
        "A_nn": A_nn_SLy4,
        "A_pp": A_pp_SLy4,
        "A_np": A_np_SLy4,
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
            "[MeV]",
            "[MeV]",
            "[cm]",
            "[cm]",
            "[cm]",
            "[]",
            "[]",
            "[]",
            "[]",
            "[]",
            "[MeV fm**5]",
            "[MeV fm**5]",
            "[MeV fm**5]",
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
    sigma_2=0.0,
    # W0=,
)

n_n_Sk = eos_Sk.n_n(n_b)
n_p_Sk = eos_Sk.n_p(n_b)
lambda_Sk = eos_Sk.lambda_L(n_b)
xi_n_Sk = eos_Sk.xi_n(n_b)
xi_p_Sk = eos_Sk.xi_p(n_b)
# A_nn_Sk, A_pp_Sk, A_np_Sk = eos_Sk.A_ii(n_b)
H_i_Sk = eos_Sk.H_parameters()

print("H_i for Ska35s20: ", [round(H, 3) for H in H_i_Sk])

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
        # "A_nn": A_nn_Sk,
        # "A_pp": A_pp_Sk,
        # "A_np": A_np_Sk,
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
            # "[MeV fm**5]",
            # "[MeV fm**5]",
            # "[MeV fm**5]",
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
    W0=104.58,
)

n_n_SQMC = eos_SQMC.n_n(n_b)
n_p_SQMC = eos_SQMC.n_p(n_b)
lambda_SQMC = eos_SQMC.lambda_L(n_b)
xi_n_SQMC = eos_SQMC.xi_n(n_b)
xi_p_SQMC = eos_SQMC.xi_p(n_b)
A_nn_SQMC, A_pp_SQMC, A_np_SQMC = eos_SQMC.A_ii(n_b)
H_i_SQMC = eos_SQMC.H_parameters()

print("H_i for SQMC700: ", [round(H, 3) for H in H_i_SQMC])

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
        "A_nn": A_nn_SQMC,
        "A_pp": A_pp_SQMC,
        "A_np": A_np_SQMC,
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
            "[MeV fm**5]",
            "[MeV fm**5]",
            "[MeV fm**5]",
        ],
    )
)
df_SQMC.to_csv("./examples/data/parameters_SQMC700.txt", index=None, header=True)


# ---- Skchi450 - parameters are taken from Lim and Holt (2017) ---- #

eos_Skchi450 = ec.EquationOfState(
    t0=-1803.2928,
    t1=301.8208,
    t2=-273.2827,
    t3=12783.8619,
    t4=564.1049,
    x0=0.4430,
    x1=-0.3622,
    x2=-0.4105,
    x3=0.6545,
    x4=-11.3160,
    sigma=1 / 3,
    sigma_2=1,
    W0=106.4288,
)

n_n_Skchi450 = eos_Skchi450.n_n(n_b)
n_p_Skchi450 = eos_Skchi450.n_p(n_b)
k_n_Skchi450 = eos_Skchi450.k_F_n(n_b)
k_p_Skchi450 = eos_Skchi450.k_F_p(n_b)
Delta_n_Skchi450 = gap_neutrons(k_n_Skchi450)
Delta_p_Skchi450 = gap_protons(k_p_Skchi450)
lambda_Skchi450 = eos_Skchi450.lambda_L(n_b)
xi_n_Skchi450 = eos_Skchi450.xi_n(n_b)
xi_p_Skchi450 = eos_Skchi450.xi_p(n_b)
A_nn_Skchi450, A_pp_Skchi450, A_np_Skchi450 = eos_Skchi450.A_ii(n_b)
H_i_Skchi450 = eos_Skchi450.H_parameters()

print("H_i for Skchi450: ", [round(H, 3) for H in H_i_Skchi450])

df_Skchi450 = pd.DataFrame(
    {
        "n_b": n_b,
        "rho_b": rho_b,
        "n_n": n_n_Skchi450,
        "n_p": n_p_Skchi450,
        "Delta_n": Delta_n_Skchi450,
        "Delta_p": Delta_p_Skchi450,
        "lambda": lambda_Skchi450,
        "xi_n": xi_n_Skchi450,
        "xi_p": xi_p_Skchi450,
        "kappa": lambda_Skchi450 / xi_p_Skchi450,  # Ginzburg-Landau parameter
        "R": xi_p_Skchi450 / xi_n_Skchi450,  # coherence length ratio
        "epsilon": n_p_Skchi450 / n_n_Skchi450,  # asymmetry parameter
        "A_nn": A_nn_Skchi450,
        "A_pp": A_pp_Skchi450,
        "A_np": A_np_Skchi450,
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
            "[MeV]",
            "[MeV]",
            "[cm]",
            "[cm]",
            "[cm]",
            "[]",
            "[]",
            "[]",
            "[MeV fm**5]",
            "[MeV fm**5]",
            "[MeV fm**5]",
        ],
    )
)
df_Skchi450.to_csv("./examples/data/parameters_Skchi450.txt", index=None, header=True)
