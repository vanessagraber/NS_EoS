import ns_eos.equilibrium_comp as ec
import numpy as np
import pytest


TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {"muon_eqn_const": 0.029956297154719348, "n_e": 0.001, "n_mu": 0.0}

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "muon_eqn_const": 0.029956297154719348,
        "n_e": 0.01,
        "n_mu": 0.002111682234292741,
        "n_p": 0.012111682234292741,
        "eos": ec.EquationOfState(
            t0=1.0, t1=0.5, t2=1.0, t3=2.0, x0=1.0, x1=0.5, x2=1.0, x3=2.0, alpha=1.0
        ),
        "n_b": 0.2,
        "del_mu": 55.29173887061031,
        "mu_e": 131.52051524885195,
        "min": -76.22877637824163,
        "n_e_min": np.array([0.001266337575998989]),
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "eos": ec.EquationOfState(),
        "n_b": np.array([0.1]),
        "x_e": np.array([0.0382232]),
        "x_p": np.array([0.0382232]),
        "x_mu": np.array([0.0]),
        "n_p": np.array([0.0038223]),
        "n_n": np.array([0.0961777]),
        "m_eff_n": np.array([1.64218807e-24]),
        "m_eff_p": np.array([1.19878858e-24]),
        "m_eff_L_n": np.array([1.445079e-24]),
        "m_eff_L_p": np.array([1.189092e-24]),
        "k_F_n": np.array([1.417420]),
        "k_F_p": np.array([0.483707]),
        "lambda_eff": np.array([9.839113e-12]),
        "xi_n": np.array([2.050619e-10]),
        "xi_p": np.array([1.094111e-12]),
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "eos": ec.EquationOfState(),
        "n_b": np.array([0.2]),
        "x_e": np.array([0.0567969]),
        "x_p": np.array([0.0715545]),
        "x_mu": np.array([0.0147575]),
        "n_p": np.array([0.01431090]),
        "n_n": np.array([0.18568910]),
        "m_eff_n": np.array([1.60724175e-24]),
        "m_eff_p": np.array([9.68987117e-25]),
        "m_eff_L_n": np.array([1.261756e-24]),
        "m_eff_L_p": np.array([9.354640e-25]),
        "k_F_n": np.array([1.764964]),
        "k_F_p": np.array([0.751097]),
        "lambda_eff": np.array([4.526990e-12]),
        "xi_n": np.array([6.831531e-11]),
        "xi_p": np.array([1.747953e-12]),
    }

    return data


@pytest.fixture()
def test_case_5():
    data = {
        "eos": ec.EquationOfState(),
        "H": (85.006655, 322.612908, 218.840437),
    }

    return data


def test_muon_eqn_const(test_case_1):
    """verifying the value of the constant affecting the muon appearance"""
    assert np.abs(ec.muon_eqn_const - test_case_1["muon_eqn_const"]) < TOL


def test_H_parameters(test_case_5):
    """verifying the H_i parameters are correctly calculated"""
    H = test_case_5["eos"].H_parameters()
    assert np.abs((H[0] - test_case_5["H"][0]) / H[0]) < TOL
    assert np.abs((H[1] - test_case_5["H"][1]) / H[1]) < TOL
    assert np.abs((H[2] - test_case_5["H"][2]) / H[1]) < TOL


def test_relation_nmu_ne_01(test_case_1):
    """verifying the relationship between the muon and electron density in absence of muons"""
    n_mu = ec.relation_nmu_ne(test_case_1["n_e"])
    assert n_mu == 0.0


def test_relation_nmu_ne_02(test_case_2):
    """verifying the relationship between the muon and electron density in presence of muons"""
    n_mu = ec.relation_nmu_ne(test_case_2["n_e"])
    assert np.abs((n_mu - test_case_2["n_mu"]) / n_mu) < TOL


def test_relation_np_ne_01(test_case_1):
    """verifying the relationship between the proton and electron density in absence of muons"""
    n_p = ec.relation_np_ne(test_case_1["n_e"])
    assert n_p == test_case_1["n_e"]


def test_relation_np_ne_02(test_case_2):
    """verifying the relationship between the proton and electron density in absence of muons"""
    n_p = ec.relation_np_ne(test_case_2["n_e"])
    assert np.abs((n_p - test_case_2["n_p"]) / n_p) < TOL


def test_relative_chem_pot(test_case_2):
    """verifying the relative chemical potential of the neutron and proton components"""
    rel_chem_pot = test_case_2["eos"].relative_chem_pot(
        test_case_2["n_b"], test_case_2["n_p"]
    )
    assert np.abs((rel_chem_pot - test_case_2["del_mu"]) / rel_chem_pot) < TOL


def test_func_to_minimize(test_case_2):
    """verifying the value of the function that is minimised to relate n_b and n_e"""
    func_min = test_case_2["eos"].func_to_minimize(
        test_case_2["n_e"], test_case_2["n_b"]
    )
    assert np.abs((func_min - test_case_2["min"]) / func_min) < TOL


def test_relation_ne_nb(test_case_2):
    """verifying the relation between the electron and baryon number densities"""
    n_e = test_case_2["eos"].relation_ne_nb(np.array([test_case_2["n_b"]]))
    assert np.abs((n_e - test_case_2["n_e_min"]) / n_e) < TOL


# particle fractions


def test_x_e_01(test_case_3):
    """verifying that the electron fraction is correctly calculated below muon threshold"""
    x_e = test_case_3["eos"].x_e(test_case_3["n_b"])
    assert np.abs((x_e - test_case_3["x_e"]) / x_e) < TOL


def test_x_e_02(test_case_4):
    """verifying that the electron fraction is correctly calculated above muon threshold"""
    x_e = test_case_4["eos"].x_e(test_case_4["n_b"])
    assert np.abs((x_e - test_case_4["x_e"]) / x_e) < TOL


def test_x_p_01(test_case_3):
    """verifying that the proton fraction is correctly calculated below muon threshold"""
    x_p = test_case_3["eos"].x_p(test_case_3["n_b"])
    assert np.abs((x_p - test_case_3["x_p"]) / x_p) < TOL


def test_x_p_02(test_case_4):
    """verifying that the proton fraction is correctly calculated above muon threshold"""
    x_p = test_case_4["eos"].x_p(test_case_4["n_b"])
    assert np.abs((x_p - test_case_4["x_p"]) / x_p) < TOL


def test_x_mu_01(test_case_3):
    """verifying that the muon fraction is correctly calculated below muon threshold"""
    x_mu = test_case_3["eos"].x_mu(test_case_3["n_b"])
    assert x_mu == test_case_3["x_mu"]


def test_x_mu_02(test_case_4):
    """verifying that the muon fraction is correctly calculated above muon threshold"""
    x_mu = test_case_4["eos"].x_mu(test_case_4["n_b"])
    assert np.abs((x_mu - test_case_4["x_mu"]) / x_mu) < TOL


# number densities


def test_n_p_01(test_case_3):
    """verifying that the proton number density is correctly calculated below muon threshold"""
    n_p = test_case_3["eos"].n_p(test_case_3["n_b"])
    assert np.abs((n_p - test_case_3["n_p"]) / n_p) < TOL


def test_n_p_02(test_case_4):
    """verifying that the proton number density is correctly calculated above muon threshold"""
    n_p = test_case_4["eos"].n_p(test_case_4["n_b"])
    assert np.abs((n_p - test_case_4["n_p"]) / n_p) < TOL


def test_n_n_01(test_case_3):
    """verifying that the neutron number density is correctly calculated below muon threshold"""
    n_n = test_case_3["eos"].n_n(test_case_3["n_b"])
    assert np.abs((n_n - test_case_3["n_n"]) / n_n) < TOL


def test_n_n_02(test_case_4):
    """verifying that the neutron number density is correctly calculated above muon threshold"""
    n_n = test_case_4["eos"].n_n(test_case_4["n_b"])
    assert np.abs((n_n - test_case_4["n_n"]) / n_n) < TOL


# dynamic effective masses caused by entrainment


def test_m_eff_n_01(test_case_3):
    """verifying that the neutron effective mass is correctly calculated below muon threshold"""
    m_eff_n = test_case_3["eos"].m_eff_n(test_case_3["n_b"])
    assert np.abs((m_eff_n - test_case_3["m_eff_n"]) / m_eff_n) < TOL


def test_m_eff_n_02(test_case_4):
    """verifying that the neutron effective mass is correctly calculated above muon threshold"""
    m_eff_n = test_case_4["eos"].m_eff_n(test_case_4["n_b"])
    assert np.abs((m_eff_n - test_case_4["m_eff_n"]) / m_eff_n) < TOL


def test_m_eff_p_01(test_case_3):
    """verifying that the proton effective mass is correctly calculated below muon threshold"""
    m_eff_p = test_case_3["eos"].m_eff_p(test_case_3["n_b"])
    assert np.abs((m_eff_p - test_case_3["m_eff_p"]) / m_eff_p) < TOL


def test_m_eff_p_02(test_case_4):
    """verifying that the proton effective mass is correctly calculated above muon threshold"""
    m_eff_p = test_case_4["eos"].m_eff_p(test_case_4["n_b"])
    assert np.abs((m_eff_p - test_case_4["m_eff_p"]) / m_eff_p) < TOL


# Landau effective masses characterising the static ground state


def test_m_eff_L_n_01(test_case_3):
    """verifying that the neutron Landau effective mass is correctly calculated below muon threshold"""
    m_eff_L_n = test_case_3["eos"].m_eff_L_n(test_case_3["n_b"])
    assert np.abs((m_eff_L_n - test_case_3["m_eff_L_n"]) / m_eff_L_n) < TOL


def test_m_eff_L_n_02(test_case_4):
    """verifying that the neutron Landau effective mass is correctly calculated above muon threshold"""
    m_eff_L_n = test_case_4["eos"].m_eff_L_n(test_case_4["n_b"])
    assert np.abs((m_eff_L_n - test_case_4["m_eff_L_n"]) / m_eff_L_n) < TOL


def test_m_eff_L_p_01(test_case_3):
    """verifying that the proton Landau effective mass is correctly calculated below muon threshold"""
    m_eff_L_p = test_case_3["eos"].m_eff_L_p(test_case_3["n_b"])
    assert np.abs((m_eff_L_p - test_case_3["m_eff_L_p"]) / m_eff_L_p) < TOL


def test_m_eff_L_p_02(test_case_4):
    """verifying that the proton Landau effective mass is correctly calculated above muon threshold"""
    m_eff_L_p = test_case_4["eos"].m_eff_L_p(test_case_4["n_b"])
    assert np.abs((m_eff_L_p - test_case_4["m_eff_L_p"]) / m_eff_L_p) < TOL


# Fermi wave numbers


def test_k_F_n_01(test_case_3):
    """verifying that the neutron wave number is correctly calculated below muon threshold"""
    k_F_n = test_case_3["eos"].k_F_n(test_case_3["n_b"])
    assert np.abs((k_F_n - test_case_3["k_F_n"]) / k_F_n) < TOL


def test_k_F_n_02(test_case_4):
    """verifying that the neutron wave number is correctly calculated above muon threshold"""
    k_F_n = test_case_4["eos"].k_F_n(test_case_4["n_b"])
    assert np.abs((k_F_n - test_case_4["k_F_n"]) / k_F_n) < TOL


def test_k_F_p_01(test_case_3):
    """verifying that the proton wave number is correctly calculated below muon threshold"""
    k_F_p = test_case_3["eos"].k_F_p(test_case_3["n_b"])
    assert np.abs((k_F_p - test_case_3["k_F_p"]) / k_F_p) < TOL


def test_k_F_p_02(test_case_4):
    """verifying that the proton wave number is correctly calculated above muon threshold"""
    k_F_p = test_case_4["eos"].k_F_p(test_case_4["n_b"])
    assert np.abs((k_F_p - test_case_4["k_F_p"]) / k_F_p) < TOL


# characteristic length scales


def test_lambda_eff_01(test_case_3):
    """verifying that the London length is correctly calculated below muon threshold"""
    lambda_eff = test_case_3["eos"].lambda_eff(test_case_3["n_b"])
    assert np.abs((lambda_eff - test_case_3["lambda_eff"]) / lambda_eff) < TOL


def test_lambda_eff_02(test_case_4):
    """verifying that the London length is correctly calculated above muon threshold"""
    lambda_eff = test_case_4["eos"].lambda_eff(test_case_4["n_b"])
    assert np.abs((lambda_eff - test_case_4["lambda_eff"]) / lambda_eff) < TOL


def test_xi_n_01(test_case_3):
    """verifying that the neutron coherence length is correctly calculated below muon threshold"""
    xi_n = test_case_3["eos"].xi_n(test_case_3["n_b"])
    assert np.abs((xi_n - test_case_3["xi_n"]) / xi_n) < TOL


def test_xi_n_02(test_case_4):
    """verifying that the neutron coherence length is correctly calculated above muon threshold"""
    xi_n = test_case_4["eos"].xi_n(test_case_4["n_b"])
    assert np.abs((xi_n - test_case_4["xi_n"]) / xi_n) < TOL


def test_xi_p_01(test_case_3):
    """verifying that the proton coherence length is correctly calculated below muon threshold"""
    xi_p = test_case_3["eos"].xi_p(test_case_3["n_b"])
    assert np.abs((xi_p - test_case_3["xi_p"]) / xi_p) < TOL


def test_xi_p_02(test_case_4):
    """verifying that the proton coherence length is correctly calculated above muon threshold"""
    xi_p = test_case_4["eos"].xi_p(test_case_4["n_b"])
    assert np.abs((xi_p - test_case_4["xi_p"]) / xi_p) < TOL
