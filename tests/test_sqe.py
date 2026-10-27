"""
Tests for the split-charge equilibration (SQE) solve, forcefield/m3gnet/sqe.py.

No model weights are needed: a small electrode (2 Mo, 2 S) with one water molecule
bonded to it and one isolated water molecule, and a random symmetric positive-definite
matrix standing in for the Coulomb + hardness matrix A.
Run with pytest, or directly: python tests/test_sqe.py
"""

import torch

from mattersim.forcefield.m3gnet import sqe

torch.set_default_dtype(torch.float64)

# Mo, Mo, S, S | water A (2.5 A above the S plane: bonded to the electrode) | water B (isolated)
Z = torch.tensor([42, 42, 16, 16, 8, 1, 1, 8, 1, 1])
POS = torch.tensor([
    [0.0, 0.0, 2.0], [3.2, 0.0, 2.0], [1.6, 0.9, 3.5], [4.8, 0.9, 3.5],
    [1.6, 0.9, 6.0], [2.2, 0.9, 6.6], [1.0, 0.9, 6.6],
    [8.0, 8.0, 14.0], [8.6, 8.0, 14.6], [7.4, 8.0, 14.6],
])
CELL = torch.eye(3) * 20.0
WATER_A, WATER_B = [4, 5, 6], [7, 8, 9]


def _system(seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(Z)
    m = torch.randn(n, n, generator=g)
    A = m @ m.T / n + torch.diag(torch.rand(n, generator=g) + 5.0)
    rhs = torch.randn(n, generator=g)
    i, j, cls, r, rc, electrode = sqe.enumerate_bonds(Z, POS, CELL)
    kappa = sqe.BondHardness(int(Z.max()))(Z[i], Z[j], cls).double() / sqe.cutoff_switch(r, rc)
    return A, rhs, (i, j, electrode), kappa, cls


def test_bond_classes():
    i, j, cls, r, rc, electrode = sqe.enumerate_bonds(Z, POS, CELL)
    assert electrode.tolist() == [True] * 4 + [False] * 6
    assert int((cls == 0).sum()) == 4                       # two O-H per molecule
    assert not bool((electrode[i] & electrode[j]).any())    # no electrode-electrode bonds
    pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(i, j)}
    assert not any(a in WATER_B or b in WATER_B for a, b in pairs if (a in WATER_A or b in WATER_A))
    assert any(electrode[a] != electrode[b] for a, b in pairs)   # water A is bonded to the electrode


def test_initial_hardness():
    bh = sqe.BondHardness(42)
    intra = bh(torch.tensor([8]), torch.tensor([1]), torch.tensor([0]))
    inter = bh(torch.tensor([8]), torch.tensor([16]), torch.tensor([1]))
    assert abs(float(intra) - sqe.KAPPA_INTRA_INIT) < 1e-5
    assert abs(float(inter) - sqe.KAPPA_INTER_INIT) < 1e-4


def test_isolated_molecule_is_neutral():
    A, rhs, bonds, kappa, _ = _system()
    q, p, r, omega = sqe.sqe_solve(A, rhs, bonds, kappa)
    assert abs(float(q[WATER_B].sum())) < 1e-10
    assert abs(float(q.sum() - r.sum())) < 1e-10           # the total charge lives on the reservoir bonds


def test_solution_is_stationary():
    A, rhs, bonds, kappa, _ = _system(1)
    i, j, electrode = bonds
    q, p, r, omega = sqe.sqe_solve(A, rhs, bonds, kappa)
    p = p.clone().requires_grad_(True)
    r = r.clone().requires_grad_(True)
    res = torch.nonzero(electrode).flatten()
    qq = torch.zeros(len(Z)).index_add(0, i, p).index_add(0, j, -p).index_add(0, res, r)
    om = 0.5 * qq @ (A @ qq) + rhs @ qq + 0.5 * (kappa * p * p).sum()
    gp, gr = torch.autograd.grad(om, (p, r))
    assert float(gp.abs().max()) < 1e-8 and float(gr.abs().max()) < 1e-8
    assert abs(float(om - omega)) < 1e-10


def test_cutoff_switch():
    rc = torch.tensor(2.6)
    r = torch.linspace(0.0, 2.6, 27)
    f = sqe.cutoff_switch(r, rc.expand_as(r))
    assert torch.all(f[r <= 0.75 * 2.6] == 1.0)
    assert torch.all(f[1:] <= f[:-1])
    assert float(f[-1]) == 1e-6                            # clamped, so kappa stays finite


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
