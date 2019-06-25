import ns_eos.gap_parametrisation as gp
import numpy as np
import pytest


TOL = 1e-5


@pytest.fixture()
def test_case_1():
    data = {
        "k_F_n": 1.5,
        "k_F_p": 0.5,
        "Delta_n": 0.021607,
        "Delta_p": 0.850687,
    }

    return data

def test_gap_neutrons(test_case_1):
    """verifying that the neutron energy gap is correctly calculated"""
    Delta_n = gp.gap_neutrons(test_case_1["k_F_n"])
    assert np.abs((Delta_n - test_case_1["Delta_n"]) / Delta_n) < TOL

def test_gap_protons(test_case_1):
    """verifying that the proton energy gap is correctly calculated"""
    Delta_p = gp.gap_protons(test_case_1["k_F_p"])
    assert np.abs((Delta_p - test_case_1["Delta_p"]) / Delta_p) < TOL