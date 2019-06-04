import ns_eos.equilibrium_comp as ec


def test_const():
    """verifying the value of the constant affecting the muon apprearance"""
    assert ec.const == 0.029956297154719348


def test_relation_nmu_ne_01():
    """verifying the relationship between the muon and electron density in absence of muons"""
    n_e = 0.001
    n_mu = ec.relation_nmu_ne(n_e)
    assert n_mu == 0.0


def test_relation_nmu_ne_02():
    """verifying the relationship between the muon and electron density in presence of muons"""
    n_e = 0.1
    n_mu = ec.relation_nmu_ne(n_e)
    assert n_mu == 0.0798860263555066


def test_relation_np_ne_01():
    """verifying the relationship between the proton and electron density in absence of muons"""
    n_e = 0.001
    n_p = ec.relation_np_ne(n_e)
    assert n_p == n_e


def test_relation_np_ne_02():
    """verifying the relationship between the proton and electron density in absence of muons"""
    n_e = 0.1
    n_p = ec.relation_np_ne(n_e)
    assert n_p == 0.0798860263555066 + n_e
