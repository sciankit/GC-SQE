"""Molecule-neutral GC-QEq and a molecule-aware slab dipole correction.

This module is what distinguishes `mattersim_neutral_water_qeq` from
`mattersim_bounded_J_sigma` (which it is copied from, bounded heads included).
Two changes, both switchable here:

1. NEUTRAL_WATER: every water molecule (an O with the H atoms whose nearest O it is)
   is constrained to zero net charge inside the charge-equilibration solve. Charge
   flows within a molecule and between the electrode and the reservoir at the
   electrode potential, but not from the electrode into the water. In the plain
   GC-QEq of experiments 19-20 the water carried +2.8 to +3.0 e at +-1 V and single
   oxygens reached -2 e; this is the smallest change that removes that channel. It is
   the limiting case of split-charge equilibration (SQE) with bonds only inside
   molecules.

   Solve: minimise 0.5 q^T A q + (chi + mu)^T q subject to C q = 0 (one row per
   molecule) -> KKT system [[A, C^T], [C, 0]] [q; lambda] = [-(chi + mu); 0].
   Omega = 0.5 q^T A q + (chi + mu)^T q at the solution (the constraint terms vanish).

2. DIPOLE_Z_MODE: how the z coordinate that enters the slab dipole / net-charge
   correction (GCQeqSolver._calc_dipole_correction_matrix) is defined.
     'atom'      - the wrapped absolute z of every atom (the original; experiment 21
                   showed it jumps whenever an atom crosses z = 0 / c, which lies
                   inside the water film);
     'molecule'  - each H takes the z of its own O plus the minimum-image O->H offset,
                   so a molecule is never cut (removes the H-crossing jumps, 97 % of
                   the crossings at +1.38 V; molecule crossings still jump);
     'slab'      - as 'molecule', and z is measured from the Mo plane with every
                   molecule's O at its minimum image relative to the slab. The
                   correction then does not depend on the cell origin and nothing is
                   cut at the arbitrary boundary; a molecule crossing the plane
                   opposite the slab (mid-gap) still changes the second-moment term
                   when the cell is charged. Its size is measured in
                   analysis/checks and reported in experiment 22.
   The correction itself is kept (Ankit's premise, 2026-10-06).

Elements: water = O (8) and H (1). Anything else is 'slab' (unconstrained).
"""
from __future__ import annotations

import os

import torch

NEUTRAL_WATER = os.environ.get("GCQEQ_NEUTRAL_WATER", "1") == "1"
DIPOLE_Z_MODE = os.environ.get("GCQEQ_DIPOLE_Z_MODE", "slab")       # 'atom' | 'molecule' | 'slab'
SLAB_REFERENCE_Z = 42                                                 # Mo: the plane the slab z is measured from


def water_groups(atomic_numbers: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor):
    """Group id per atom (-1 = not in a molecule) and, for each H, the index of its O.

    Each H joins the nearest O by minimum image through the full cell. Each O is one
    group; groups are consecutive integers.
    """
    n = atomic_numbers.shape[0]
    groups = torch.full((n,), -1, dtype=torch.long, device=pos.device)
    is_o = atomic_numbers == 8
    is_h = atomic_numbers == 1
    o_idx = torch.nonzero(is_o).flatten()
    h_idx = torch.nonzero(is_h).flatten()
    h_owner = torch.full((n,), -1, dtype=torch.long, device=pos.device)
    if o_idx.numel() == 0:
        return groups, h_owner
    groups[o_idx] = torch.arange(o_idx.numel(), device=pos.device)
    if h_idx.numel():
        inv = torch.linalg.inv(cell)
        d = pos[h_idx][:, None, :] - pos[o_idx][None, :, :]           # (nH, nO, 3)
        f = d @ inv
        f = f - torch.round(f)
        dist = torch.linalg.norm(f @ cell, dim=-1)
        nearest = torch.argmin(dist, dim=1)
        h_owner[h_idx] = o_idx[nearest]
        groups[h_idx] = groups[o_idx[nearest]]
    return groups, h_owner


def correction_z(atomic_numbers: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor,
                 h_owner: torch.Tensor, mode: str = DIPOLE_Z_MODE) -> torch.Tensor:
    """The z coordinate per atom that the slab correction should use (see module doc)."""
    z = pos[:, 2]
    if mode == "atom":
        return z
    Lz = torch.linalg.norm(cell[2])
    has_owner = h_owner >= 0
    zo = torch.where(has_owner, z[h_owner.clamp(min=0)], z)
    dz = z - zo
    dz = dz - Lz * torch.round(dz / Lz)
    z_mol = torch.where(has_owner, zo + dz, z)                         # H attached to its O
    if mode == "molecule":
        return z_mol
    if mode == "slab":
        ref = atomic_numbers == SLAB_REFERENCE_Z
        if ref.any():
            zc = z[ref].mean()                                        # Mo plane (slab centre)
        else:
            zc = torch.zeros((), device=z.device, dtype=z.dtype)
        # O (and any non-water atom) at its minimum image relative to the slab centre;
        # H follows its O
        base = torch.where(has_owner, zo, z) - zc
        base = base - Lz * torch.round(base / Lz)
        return torch.where(has_owner, base + dz, base)
    raise ValueError(f"unknown DIPOLE_Z_MODE {mode!r}")
