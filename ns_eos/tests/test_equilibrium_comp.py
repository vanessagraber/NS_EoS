import ns_eos.equilibrium_comp as ec
import numpy as np
import pytest


TOL = 1e-6


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
        "x_e": np.array([0.038223]),
        "x_p": np.array([0.038223]),
        "x_mu": np.array([0.0]),
        "m_eff_n": np.array([1.64218807e-24]),
        "m_eff_p": np.array([[1.19878858e-24]]),
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "eos": ec.EquationOfState(),
        "n_b": np.array([0.2]),
        "x_e": np.array([0.056796]),
        "x_p": np.array([0.071554]),
        "x_mu": np.array([0.014757]),
        "m_eff_n": np.array([[1.60724175e-24]]),
        "m_eff_p": np.array([[9.68987117e-25]]),
    }

    return data


def test_muon_eqn_const(test_case_1):
    """verifying the value of the constant affecting the muon appearance"""
    assert np.abs(ec.muon_eqn_const - test_case_1["muon_eqn_const"]) < TOL


def test_relation_nmu_ne_01(test_case_1):
    """verifying the relationship between the muon and electron density in absence of muons"""
    n_mu = ec.relation_nmu_ne(test_case_1["n_e"])
    assert n_mu == 0.0


def test_relation_nmu_ne_02(test_case_2):
    """verifying the relationship between the muon and electron density in presence of muons"""
    n_mu = ec.relation_nmu_ne(test_case_2["n_e"])
    assert np.abs(n_mu - test_case_2["n_mu"]) < TOL


def test_relation_np_ne_01(test_case_1):
    """verifying the relationship between the proton and electron density in absence of muons"""
    n_p = ec.relation_np_ne(test_case_1["n_e"])
    assert n_p == test_case_1["n_e"]


def test_relation_np_ne_02(test_case_2):
    """verifying the relationship between the proton and electron density in absence of muons"""
    n_p = ec.relation_np_ne(test_case_2["n_e"])
    assert np.abs(n_p - test_case_2["n_p"]) < TOL


def test_relative_chem_pot(test_case_2):
    """verifying the relative chemical potential of the neutron and proton components"""
    rel_chem_pot = test_case_2["eos"].relative_chem_pot(
        test_case_2["n_b"], test_case_2["n_p"]
    )
    assert np.abs(rel_chem_pot - test_case_2["del_mu"]) < TOL


def test_func_to_minimize(test_case_2):
    """verifying the value of the function that is minimised to relate n_b and n_e"""
    func_min = test_case_2["eos"].func_to_minimize(
        test_case_2["n_e"], test_case_2["n_b"]
    )
    assert np.abs(func_min - test_case_2["min"]) < TOL


def test_relation_ne_nb(test_case_2):
    """verifying the relation between the electron and baryon number densities"""
    n_e = test_case_2["eos"].relation_ne_nb(np.array([test_case_2["n_b"]]))
    assert np.abs(n_e - test_case_2["n_e_min"]) < TOL


def test_x_e_01(test_case_3):
    """verifying that the electron fraction is correctly calculated below muon threshold"""
    x_e = test_case_3["eos"].x_e(test_case_3["n_b"])
    assert np.abs(x_e - test_case_3["x_e"]) < TOL


def test_x_e_02(test_case_4):
    """verifying that the electron fraction is correctly calculated above muon threshold"""
    x_e = test_case_4["eos"].x_e(test_case_4["n_b"])
    assert np.abs(x_e - test_case_4["x_e"]) < TOL


def test_x_p_01(test_case_3):
    """verifying that the proton fraction is correctly calculated below muon threshold"""
    x_p = test_case_3["eos"].x_p(test_case_3["n_b"])
    assert np.abs(x_p - test_case_3["x_p"]) < TOL


def test_x_p_02(test_case_4):
    """verifying that the proton fraction is correctly calculated above muon threshold"""
    x_p = test_case_4["eos"].x_p(test_case_4["n_b"])
    assert np.abs(x_p - test_case_4["x_p"]) < TOL


def test_x_mu_01(test_case_3):
    """verifying that the muon fraction is correctly calculated below muon threshold"""
    x_mu = test_case_3["eos"].x_mu(test_case_3["n_b"])
    assert np.abs(x_mu - test_case_3["x_mu"]) < TOL


def test_x_mu_02(test_case_4):
    """verifying that the muon fraction is correctly calculated above muon threshold"""
    x_mu = test_case_4["eos"].x_mu(test_case_4["n_b"])
    assert np.abs(x_mu - test_case_4["x_mu"]) < TOL


def test_m_eff_n_01(test_case_3):
    """verifying that the neutron effective mass is correctly calculated below muon threshold"""
    m_eff_n = test_case_3["eos"].m_eff_n(test_case_3["n_b"])
    assert np.abs(m_eff_n - test_case_3["m_eff_n"]) < TOL


def test_m_eff_n_02(test_case_4):
    """verifying that the neutron effective mass is correctly calculated above muon threshold"""
    m_eff_n = test_case_4["eos"].m_eff_n(test_case_4["n_b"])
    assert np.abs(m_eff_n - test_case_4["m_eff_n"]) < TOL


def test_m_eff_p_01(test_case_3):
    """verifying that the proton effective mass is correctly calculated below muon threshold"""
    m_eff_p = test_case_3["eos"].m_eff_p(test_case_3["n_b"])
    assert np.abs(m_eff_p - test_case_3["m_eff_p"]) < TOL


def test_m_eff_p_02(test_case_4):
    """verifying that the proton effective mass is correctly calculated above muon threshold"""
    m_eff_p = test_case_4["eos"].m_eff_p(test_case_4["n_b"])
    assert np.abs(m_eff_p - test_case_4["m_eff_p"]) < TOL
