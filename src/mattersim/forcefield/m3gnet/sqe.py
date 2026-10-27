"""Split-charge equilibration (SQE) for the grand-canonical interface.

This is what distinguishes `mattersim_sqe` from `mattersim_neutral_water_qeq` (its parent;
bounded heads and the molecule-aware slab correction are kept).

Idea (Nistor, Polihronov, Mueser, Mosey, J. Chem. Phys. 125, 094108 (2006)): the variables of
the charge equilibration are not atomic charges but BOND charges p_e, one per bond e = (i, j),
transferred from j to i (p_ji = -p_ij). The atomic charge is q_i = sum over bonds of i of p_e, so
charge can only flow along bonds, over a finite distance, and every bond has its own hardness
kappa_e p_e^2 / 2. A molecule without bonds to anything is neutral by construction; neighbouring
molecules can exchange a little charge through hydrogen bonds (kappa large), and the response of
an insulator to a field is dielectric, not metallic.

Grand-canonical electrode: every electrode (non-water) atom i has, in addition, a "reservoir
bond" r_i with the electron reservoir at the electrode potential mu, with no hardness (the atomic
hardness J_i and the Ewald matrix already set the stiffness). Electrode atoms have no bonds among
themselves (those would be redundant with the reservoir bonds). The total charge of the cell is
Q = sum_i r_i, exactly as in GC-QEq.

Bonds (minimum image, re-enumerated every evaluation):
  intra   O-H with r < R_INTRA (1.30 A)                        -> hardness kappa_intra(O,H)
  hbond   O...H with R_INTRA <= r < R_HB (2.60 A)              -> kappa_inter(O,H)
  w-slab  water atom - electrode atom with r < R_WS (3.50 A)   -> kappa_inter(Z_i, Z_j)
Each kappa_e = softplus(theta[Z_i, Z_j, class]) / f_c(r), with f_c a cosine switch that goes to 0
at the class cutoff, so a bond switches off continuously (p_e -> 0 as r -> cutoff).
theta is a trainable table (initialised to kappa_intra = 0.5 eV/e^2, kappa_inter = 20 eV/e^2:
the model starts close to the molecule-neutral QEq it is fine-tuned from).

Energy functional minimised (x = [p; r], q = M x, M = [B S]):
  Omega(x) = (chi + mu)^T q + 1/2 q^T A q + 1/2 sum_e kappa_e p_e^2
  -> (M^T A M + diag(kappa, 0)) x = -M^T (chi + mu)      (strictly convex: kappa_e > 0, A_slab > 0)
Omega at the solution is returned as the electrostatic energy, as in GC-QEq.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

R_INTRA, R_HB, R_WS = 1.30, 2.60, 3.50
KAPPA_INTRA_INIT, KAPPA_INTER_INIT = 0.5, 20.0            # eV / e^2
SQE_ENABLED = os.environ.get("GCQEQ_SQE", "1") == "1"
WATER_Z = (1, 8)


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class BondHardness(nn.Module):
    """Trainable per-pair, per-class bond hardness table: theta[Z_i, Z_j, class], symmetric in i, j."""

    def __init__(self, max_z: int):
        super().__init__()
        t = torch.full((max_z + 1, max_z + 1, 2), _inv_softplus(KAPPA_INTER_INIT))
        t[..., 0] = _inv_softplus(KAPPA_INTRA_INIT)
        self.theta = nn.Parameter(t)

    def forward(self, zi, zj, cls):
        th = 0.5 * (self.theta[zi, zj, cls] + self.theta[zj, zi, cls])
        return torch.nn.functional.softplus(th)


def enumerate_bonds(atomic_numbers: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor):
    """Bond list (i, j), class (0 intra, 1 inter), distance, and the electrode mask."""
    n = atomic_numbers.shape[0]
    water = torch.zeros(n, dtype=torch.bool, device=pos.device)
    for z in WATER_Z:
        water |= atomic_numbers == z
    is_o, is_h = atomic_numbers == 8, atomic_numbers == 1
    inv = torch.linalg.inv(cell)
    d = pos[:, None, :] - pos[None, :, :]
    f = d @ inv
    f = f - torch.round(f)
    dist = torch.linalg.norm(f @ cell, dim=-1)
    iu = torch.triu_indices(n, n, offset=1, device=pos.device)
    i, j = iu[0], iu[1]
    r = dist[i, j]
    oh = (is_o[i] & is_h[j]) | (is_h[i] & is_o[j])
    intra = oh & (r < R_INTRA)
    hbond = oh & (r >= R_INTRA) & (r < R_HB)
    ws = (water[i] != water[j]) & (r < R_WS)
    keep = intra | hbond | ws
    cls = torch.where(intra, torch.zeros_like(r, dtype=torch.long), torch.ones_like(r, dtype=torch.long))
    rcut = torch.where(intra, torch.full_like(r, R_INTRA), torch.where(hbond, torch.full_like(r, R_HB), torch.full_like(r, R_WS)))
    return i[keep], j[keep], cls[keep], r[keep], rcut[keep], ~water


def cutoff_switch(r: torch.Tensor, rcut: torch.Tensor, r0_frac: float = 0.75) -> torch.Tensor:
    """1 for r < r0_frac*rcut, cosine to 0 at rcut (clamped away from 0 so kappa stays finite)."""
    r0 = r0_frac * rcut
    x = ((r - r0) / (rcut - r0)).clamp(0.0, 1.0)
    return (0.5 * (1.0 + torch.cos(math.pi * x))).clamp(min=1e-6)


def sqe_solve(A: torch.Tensor, rhs: torch.Tensor, bonds, kappa: torch.Tensor):
    """Minimise Omega(x); returns atomic charges q, bond charges p, reservoir charges r, Omega."""
    i, j, electrode = bonds
    n, E = A.shape[0], i.shape[0]
    res_idx = torch.nonzero(electrode).flatten()
    nr = res_idx.shape[0]
    M = torch.zeros(n, E + nr, dtype=A.dtype, device=A.device)
    ar = torch.arange(E, device=A.device)
    M[i, ar] = 1.0
    M[j, ar] = -1.0
    M[res_idx, E + torch.arange(nr, device=A.device)] = 1.0
    H = M.T @ A @ M
    H = H + torch.diag(torch.cat([kappa, torch.zeros(nr, dtype=A.dtype, device=A.device)]))
    x = torch.linalg.solve(H, -(M.T @ rhs))
    q = M @ x
    p, r = x[:E], x[E:]
    omega = 0.5 * q @ (A @ q) + rhs @ q + 0.5 * (kappa * p * p).sum()
    return q, p, r, omega
