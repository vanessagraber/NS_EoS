"""
Tests for the functions in gap_parametrisation.py.

Authors:

        Vanessa Graber (graber@ice.csic.es)

    Copyright (c) Vanessa Graber
"""

import ns_eos.gap_parametrisation as gp
import numpy as np
import pytest


@pytest.fixture()
def test_case_1():
    data = {"k_F": 0.5, "Delta": 0.5924535603715172}

    return data


@pytest.fixture()
def test_case_2():
    data = {
        "k_F_n_left": 0.0,
        "k_F_n_right": 1.5,
        "Delta_n_left": np.nan,
        "Delta_n_right": np.nan,
    }

    return data


@pytest.fixture()
def test_case_3():
    data = {
        "k_F_p_left": 0.0,
        "k_F_p_right": 1.5,
        "Delta_p_left": np.nan,
        "Delta_p_right": np.nan,
    }

    return data


@pytest.fixture()
def test_case_4():
    data = {
        "k_F_n_left": 0.7,
        "k_F_n_right": 3.5,
        "Delta_n_left": np.nan,
        "Delta_n_right": np.nan,
    }

    return data


def test_gap_full(test_case_1):
    """verifying that the energy gap is correctly calculated"""
    Delta = gp.gap_full(test_case_1["k_F"], gp.neutron_triplet_TToa)
    assert np.isclose(Delta, test_case_1["Delta"])


def test_gap_singlet_neutrons_01(test_case_2):
    """verifying that the singlet neutron energy gap vanishes at low k_F_n"""
    Delta_n = gp.gap_singlet_neutrons(
        test_case_2["k_F_n_left"], gp.neutron_singlet_CCDK
    )
    assert Delta_n is test_case_2["Delta_n_left"]


def test_gap_singlet_neutrons_02(test_case_2):
    """verifying that the singlet neutron energy gap vanishes at high k_F_n"""
    Delta_n = gp.gap_singlet_neutrons(
        test_case_2["k_F_n_right"], gp.neutron_singlet_CCDK
    )
    assert Delta_n is test_case_2["Delta_n_right"]


def test_gap_singlet_protons_01(test_case_3):
    """verifying that the singlet proton energy gap vanishes at low k_F_n"""
    Delta_p = gp.gap_singlet_protons(test_case_3["k_F_p_left"], gp.proton_singlet_CCDK)
    assert Delta_p is test_case_3["Delta_p_left"]


def test_gap_singlet_protons_02(test_case_3):
    """verifying that the singlet proton energy gap vanishes at high k_F_n"""
    Delta_p = gp.gap_singlet_protons(test_case_3["k_F_p_right"], gp.proton_singlet_CCDK)
    assert Delta_p is test_case_3["Delta_p_right"]


def test_gap_triplet_neutrons_01(test_case_4):
    """verifying that the triplet neutron energy gap vanishes at low k_F_n"""
    Delta_n = gp.gap_singlet_neutrons(
        test_case_4["k_F_n_left"], gp.neutron_triplet_TToa
    )
    assert Delta_n is test_case_4["Delta_n_left"]


def test_gap_triplet_neutrons_02(test_case_4):
    """verifying that the triplet neutron energy gap vanishes at high k_F_n"""
    Delta_n = gp.gap_singlet_neutrons(
        test_case_4["k_F_n_right"], gp.neutron_triplet_TToa
    )
    assert Delta_n is test_case_4["Delta_n_right"]
