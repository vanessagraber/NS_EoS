import ns_eos.gap_parametrisation as gp
import numpy as np
import pytest


TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {"k_F_n": 0.5, "k_F_p": 0.0, "Delta_n": 0.0, "Delta_p": 0.0}

    return data


@pytest.fixture()
def test_case_2():
    data = {"k_F_n": 1.5, "k_F_p": 0.5, "Delta_n": 0.021607, "Delta_p": 0.850687}

    return data


@pytest.fixture()
def test_case_3():
    data = {"k_F_n": 2.5, "k_F_p": 1.5, "Delta_n": 0.0, "Delta_p": 0.0}

    return data


def test_gap_neutrons_01(test_case_1):
    """verifying that the neutron energy gap vanishes at low k_F_n"""
    Delta_n = gp.gap_neutrons(test_case_1["k_F_n"])
    assert Delta_n == 0.0


def test_gap_protons_01(test_case_1):
    """verifying that the proton energy gap vanishes at low k_F_p"""
    Delta_p = gp.gap_protons(test_case_1["k_F_p"])
    assert Delta_p == 0.0


def test_gap_neutrons_full(test_case_2):
    """verifying that the full neutron energy gap is correctly calculated"""
    Delta_n = gp.gap_neutrons_full(test_case_2["k_F_n"])
    assert np.abs((Delta_n - test_case_2["Delta_n"]) / Delta_n) < TOL


def test_gap_protons_full(test_case_2):
    """verifying that the full proton energy gap is correctly calculated"""
    Delta_p = gp.gap_protons_full(test_case_2["k_F_p"])
    assert np.abs((Delta_p - test_case_2["Delta_p"]) / Delta_p) < TOL


def test_gap_neutrons_02(test_case_2):
    """verifying that the neutron energy gap is correctly calculated"""
    Delta_n = gp.gap_neutrons(test_case_2["k_F_n"])
    assert np.abs((Delta_n - test_case_2["Delta_n"]) / Delta_n) < TOL


def test_gap_protons_02(test_case_2):
    """verifying that the proton energy gap is correctly calculated"""
    Delta_p = gp.gap_protons(test_case_2["k_F_p"])
    assert np.abs((Delta_p - test_case_2["Delta_p"]) / Delta_p) < TOL


def test_gap_neutrons_03(test_case_3):
    """verifying that the neutron energy gap vanishes at high k_F_n"""
    Delta_n = gp.gap_neutrons(test_case_3["k_F_n"])
    assert Delta_n == 0.0


def test_gap_protons_03(test_case_3):
    """verifying that the proton energy gap vanishes at high k_F_p"""
    Delta_p = gp.gap_protons(test_case_3["k_F_p"])
    assert Delta_p == 0.0
