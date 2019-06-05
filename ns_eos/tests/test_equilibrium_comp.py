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
        "t0": 1.0,
        "t1": 0.5,
        "t2": 1.0,
        "t3": 2.0,
        "x0": 1.0,
        "x1": 0.5,
        "x2": 1.0,
        "x3": 2.0,
        "alpha": 1.0,
        "n_b": 0.2,
        "del_mu": 55.29173887061031,
        "mu_e": 131.52051524885195,
        "min": -76.22877637824163,
        "n_e_min": 0.001266337575998989,
    }

    return data


def test_const(test_case_1):
    """verifying the value of the constant affecting the muon appearance"""
    assert ec.muon_eqn_const == test_case_1["muon_eqn_const"]


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
    eos = ec.EquationOfState(
        t0=test_case_2["t0"],
        t1=test_case_2["t1"],
        t2=test_case_2["t2"],
        t3=test_case_2["t3"],
        x0=test_case_2["x0"],
        x1=test_case_2["x1"],
        x2=test_case_2["x2"],
        x3=test_case_2["x3"],
        alpha=test_case_2["alpha"],
    )
    rel_chem_pot = eos.relative_chem_pot(test_case_2["n_b"], test_case_2["n_p"])
    assert np.abs(rel_chem_pot - test_case_2["del_mu"]) < TOL


def test_func_to_minimize(test_case_2):
    """verifying the value of the function that is minimised to relate n_b and n_e"""
    eos = ec.EquationOfState(
        t0=test_case_2["t0"],
        t1=test_case_2["t1"],
        t2=test_case_2["t2"],
        t3=test_case_2["t3"],
        x0=test_case_2["x0"],
        x1=test_case_2["x1"],
        x2=test_case_2["x2"],
        x3=test_case_2["x3"],
        alpha=test_case_2["alpha"],
    )
    func_min = eos.func_to_minimize(test_case_2["n_e"], test_case_2["n_b"])
    assert np.abs(func_min - test_case_2["min"]) < TOL


def test_relation_ne_nb(test_case_2):
    """verifying the relation between the electron and baryon number densities"""
    eos = ec.EquationOfState(
        t0=test_case_2["t0"],
        t1=test_case_2["t1"],
        t2=test_case_2["t2"],
        t3=test_case_2["t3"],
        x0=test_case_2["x0"],
        x1=test_case_2["x1"],
        x2=test_case_2["x2"],
        x3=test_case_2["x3"],
        alpha=test_case_2["alpha"],
    )
    n_e = eos.relation_ne_nb(np.array([test_case_2["n_b"]]))
    assert np.abs(n_e - np.array(test_case_2["n_e_min"])) < TOL
