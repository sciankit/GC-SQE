"""Per-element windows for the charge-equilibration heads and for the charges.

This module is the only thing that distinguishes `mattersim_bounded_J_sigma`
from the original `mattersim` model code, apart from the few lines that call it
(m3gnet.py: the soft clamp on J and on the Gaussian width factor;
potential.py: the charge-window penalty in the loss).

Why. In the original model the hardness J and the width factor are free
outputs of a softplus, and on the model's own MD trajectories single oxygens
reach charges of -2 to -7 e because J collapses (experiments 16 and 18 in
../paper/07_experiments). On reference frames the heads already span factors
of 4-36 within one element, so there is no physical range to clamp to at
inference; the window has to be part of the model and trained into it.

How. J and the width factor are mapped through a smooth saturating function
into a window around the per-element median of the model this one is fine-tuned
from (results_finetune_optimal_merged_reduced_test_5_active_4/best_model.pth),
measured on 208 solvated + 124 bare-slab training frames
(data/head_centres_and_ddec6_ranges.json):

    J            window = [median / 2, 2 * median]      (hardness; its collapse is the runaway)
    width factor window = [median / 3, 3 * median]      (self-energy term; weaker lever)

    soft_clamp(x) = lo + (hi - lo) * sigmoid( 4 (x - mid) / (hi - lo) ),  mid = (lo + hi)/2

which has unit slope at the centre of the window (so the pretrained behaviour
is kept where it was sensible) and saturates at the edges. With these windows
the head-dependent part of the solver diagonal, D = J + k_e/(sqrt(pi) sigma),
cannot fall below about 14 eV/e^2 for oxygen; the runaway oxygens had D ~ 6.

Charges. A quadratic penalty is added to the training loss for any atomic
charge outside the range of the DDEC6 reference charges of its element (349
labelled training frames), widened by 0.15 e on each side. It costs nothing
while charges are physical and needs no labels, so it acts on every frame.

Elements not listed are left unbounded.
"""
from __future__ import annotations

import torch

# measured per-element medians of the starting model on training frames
J_MEDIAN = {1: 60.21, 8: 25.78, 16: 27.08, 42: 8.52}
WIDTH_MEDIAN = {1: 5.43, 8: 4.12, 16: 4.45, 42: 0.77}
J_FACTOR, WIDTH_FACTOR = 2.0, 3.0

# DDEC6 charge range on the labelled training frames, min and max, in e
DDEC6_RANGE = {1: (0.279, 0.698), 8: (-1.048, -0.495), 16: (-0.554, 0.036), 42: (0.171, 0.937)}
CHARGE_MARGIN = 0.15


def head_windows(max_z: int, device=None):
    """lo/hi tensors indexed by atomic number for J and for the width factor."""
    big = 1.0e6
    J_lo = torch.zeros(max_z + 1, device=device)
    J_hi = torch.full((max_z + 1,), big, device=device)
    w_lo = torch.zeros(max_z + 1, device=device)
    w_hi = torch.full((max_z + 1,), big, device=device)
    for z, m in J_MEDIAN.items():
        J_lo[z], J_hi[z] = m / J_FACTOR, m * J_FACTOR
    for z, m in WIDTH_MEDIAN.items():
        w_lo[z], w_hi[z] = m / WIDTH_FACTOR, m * WIDTH_FACTOR
    return J_lo, J_hi, w_lo, w_hi


def charge_windows(max_z: int, device=None):
    """lo/hi tensors indexed by atomic number for the charge penalty."""
    q_lo = torch.full((max_z + 1,), -10.0, device=device)
    q_hi = torch.full((max_z + 1,), 10.0, device=device)
    for z, (lo, hi) in DDEC6_RANGE.items():
        q_lo[z], q_hi[z] = lo - CHARGE_MARGIN, hi + CHARGE_MARGIN
    return q_lo, q_hi


def soft_clamp(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Smoothly confine x to (lo, hi); identity slope at the window centre.

    Where the window is effectively infinite (unbounded elements) x is returned
    unchanged.
    """
    width = hi - lo
    mid = 0.5 * (lo + hi)
    bounded = width < 1.0e5
    y = lo + width * torch.sigmoid(4.0 * (x - mid) / width)
    return torch.where(bounded, y, x)


def charge_window_penalty(q: torch.Tensor, atomic_numbers: torch.Tensor, q_lo, q_hi):
    """Mean squared distance of each charge outside its element's window."""
    lo, hi = q_lo[atomic_numbers], q_hi[atomic_numbers]
    viol = torch.relu(lo - q) + torch.relu(q - hi)
    return (viol ** 2).mean(), viol.mean()
