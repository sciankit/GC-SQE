# The `sqe` branch: split-charge equilibration for water at an electrode

This branch changes three things in GC-QEq:
- it replaces the atomic-charge solve with **split-charge equilibration (SQE)** for the water;
- it confines the hardness and Gaussian-width heads to per-element windows;
- it changes the coordinate that the slab dipole correction uses.

Everything else is as on `main`: the MatterSim backbone, the potential-conditioned χ / J / width heads, the
Gaussian Ewald sum, and the energy (short-range part + the charge grand potential Ω).

## Why

In atomic-charge GC-QEq any atom can take charge from the electrode. At a charged electrode the water film picks up
net charge, and single oxygens can run to unphysical charges when their learned hardness J collapses.

SQE (Nistor, Polihronov, Müser and Mosey, *J. Chem. Phys.* **125**, 094108 (2006)) lets charge move only along
bonds:
- a water molecule changes its net charge only through hydrogen bonds or bonds to the electrode;
- an isolated molecule stays exactly neutral;
- the total charge of the cell sits on the electrode, which is still in contact with the electron reservoir at the
  applied potential.

## The solve (`src/mattersim/forcefield/m3gnet/sqe.py`)

**Variables.** Bond charges p_e, one per bond e = (i, j), and a reservoir charge r_i on every electrode atom. The
atomic charges are q = M [p; r].

**Minimisation.** With A the Coulomb + hardness matrix of `main`:

    Ω(p, r) = (χ + μ)ᵀ q + ½ qᵀ A q + ½ Σ_e κ_e p_e²
    →  (Mᵀ A M + diag(κ, 0)) [p; r] = −Mᵀ (χ + μ)

Ω at the solution is the electrostatic energy, as on `main`.

**Bonds.** Built at every evaluation, with the minimum-image convention. "Electrode" means every atom that is not H
or O.

| class | pair | distance | initial κ (eV/e²) |
|---|---|---|---|
| intramolecular | O–H | < 1.30 Å | 0.5 |
| hydrogen bond | O···H | 1.30–2.60 Å | 20 |
| water–electrode | water atom – electrode atom | < 3.50 Å | 20 |

- Electrode atoms have no bonds among themselves. Each has a reservoir bond with no bond hardness, so the electrode
  responds as in GC-QEq.
- κ_e = softplus(θ[Z_i, Z_j, class]) / f_c(r). Here θ is a trainable table (`BondHardness`, saved in the checkpoint),
  and f_c is a cosine switch from 1 at 75 % of the class cutoff to 0 at the cutoff. A bond's charge therefore goes
  to zero continuously as the bond breaks.

## All changes relative to `main`

| file | change |
|---|---|
| `src/mattersim/forcefield/m3gnet/sqe.py` | **new**: bond enumeration, cutoff switch, `BondHardness`, `sqe_solve` |
| `src/mattersim/forcefield/m3gnet/bounded_heads.py` | **new**: per-element windows for J and the width factor, the smooth clamp, and the charge-window penalty |
| `src/mattersim/forcefield/m3gnet/neutral_water.py` | **new**: water-molecule grouping, the molecule-neutral constraint, and the z used by the slab correction |
| `src/mattersim/forcefield/m3gnet/m3gnet.py` | `bond_hardness` parameter table; clamp on J and the width factor; per structure, builds molecules, bonds and κ and passes them to the solver |
| `src/mattersim/forcefield/m3gnet/modules/GCQEqSolver.py` | new optional arguments `sqe_bonds`, `sqe_kappa`, `groups`, `correction_z`; `forward` runs the SQE solve when bonds are given, otherwise the molecule-neutral solve when groups are given, otherwise the solve of `main`; the slab correction uses `correction_z` when given |
| `src/mattersim/forcefield/potential.py` | `train_model(charge_window_ratio=1.0)`: the charge-window penalty in the loss; `window_viol(q)` in the training log |
| `run_finetune.py` | `--charge_window_ratio` (default 1.0; 0 switches the penalty off) |
| `tests/test_sqe.py` | **new**: unit tests of the SQE solve (no model weights needed) |

Code comments marked `mattersim_bounded_J_sigma`, `mattersim_neutral_water_qeq` and `mattersim_sqe` name the
development copy each change came from.

**Bounded heads.** J and the width factor pass through
lo + (hi − lo)·sigmoid(4(x − mid)/(hi − lo)), which has slope 1 at the centre of the window. The windows are per
element, set around the medians of the model that was fine-tuned:

| | J window (eV/e²) | width-factor window |
|---|---|---|
| H | 30.1–120.4 | 1.81–16.3 |
| O | 12.9–51.6 | 1.37–12.4 |
| S | 13.5–54.2 | 1.48–13.4 |
| Mo | 4.26–17.0 | 0.26–2.31 |

- Other elements are not bounded, and χ is never bounded.
- The windows are non-persistent buffers, so checkpoints from `main` load unchanged.

**Charge-window penalty.** The loss gains `charge_window_ratio × mean(relu(distance of q_i outside its element's
window)²)` over every atom of every structure; no charge labels are needed. The windows are the DDEC6 charge range of
the training data widened by 0.15 e:

| | H | O | S | Mo |
|---|---|---|---|---|
| window (e) | +0.13…+0.85 | −1.20…−0.35 | −0.70…+0.19 | +0.02…+1.09 |

## Switches

Environment variables, read when the model module is imported:

| variable | default | effect |
|---|---|---|
| `GCQEQ_SQE` | `1` | SQE solve whenever the structure contains electrode atoms |
| `GCQEQ_NEUTRAL_WATER` | `1` | only when SQE is off: every water molecule is held at zero net charge |
| `GCQEQ_DIPOLE_Z_MODE` | `slab` | z used by the slab dipole correction (see below) |

The modes of `GCQEQ_DIPOLE_Z_MODE`:
- `atom`: the wrapped z of every atom, as on `main`;
- `molecule`: each H takes its O's z plus the minimum-image O–H offset;
- `slab`: as `molecule`, but measured from the Mo plane, with every molecule at its minimum image relative to the
  slab.

`GCQEQ_SQE=0 GCQEQ_NEUTRAL_WATER=0 GCQEQ_DIPOLE_Z_MODE=atom` gives the solver of `main`. The bounded heads stay on.

## Training

`run_finetune.py` as on `main`, plus `--charge_window_ratio`. The SQE model of our study was fine-tuned from a
bounded-head GC-QEq model with these settings:
- loss weights: energy 1, `--force_loss_ratio 0.2`, `--charge_loss_ratio 0.2`, `--charge_i_loss_ratio 0.2`,
  `--charge_window_ratio 1.0`;
- `--lr 5e-5`, `--batch_size 2`.

The trained weights are not in the repository.

## Tests

`python tests/test_sqe.py` (or `pytest tests/test_sqe.py`) checks four things:
- the bond classes;
- the initial bond hardnesses;
- that an isolated molecule is exactly neutral and the cell's charge equals the sum of the reservoir charges;
- that the solution is a stationary point of Ω, and the shape of the cutoff switch.

**Check against the code that trained the model.** The same checkpoint on the same structures (three AIMD frames and
two MD frames of a MoS₂/water cell) gives the same energies, forces, charges and heads to within 1.1 × 10⁻⁶. This holds
with SQE on, with the molecule-neutral solve, and with the plain solve.

## Known limitations

- **κ is effectively not learned.** The bond-hardness table is trainable, but in our fine-tune it stayed within 1 % of
  its initial values, so charge transfer is set by the initial κ.
- **χ is not bounded.** In very crowded water a free χ can still drive a single oxygen to a large charge.
- **The slab correction still has a cut.** Its formula is unchanged. In `slab` mode the cut sits mid-gap, opposite
  the slab: when the cell is charged, a molecule crossing that plane changes the energy discontinuously. In `atom`
  mode the cut is at the cell boundary instead.
