<h1>GC-QEq</h1>

> **Branch `sqe`:** this branch adds split-charge equilibration (SQE) for water at an electrode, bounded
> hardness/width heads, and a slab-referenced dipole correction. What changed and how to switch it: [`SQE.md`](SQE.md).

GC-QEq extends a message-passing interatomic potential with a **grand-canonical charge
equilibration (GC-QEq) layer**. The model is conditioned on an applied
electrode potential, which makes it suitable for electrochemical interfaces, charged slabs, and other systems where the number of electrons is set by a reservoir instead of being fixed.

This project contains code derived from [MatterSim](https://github.com/microsoft/mattersim), which is licensed under the MIT License. This project is not affiliated with or endorsed by Microsoft. Copyright for the original MatterSim code remains with Microsoft Corporation. See [`LICENSE.txt`](LICENSE.txt) for details.

## Installation

```bash
conda create -n "your_env" python=3.12
conda activate "your_env"

# Clone the repository
git clone https://github.com/Hu-group-at-GT/GC-QEq.git
cd GC-QEq
pip install -e .
```

## Data format

Training and validation data are read with ASE (e.g. `extxyz`). Each structure must carry the GC-QEq labels:

| Key | Location | Type | Description |
|-----|----------|------|-------------|
| `g_energy` | `atoms.info` | float | grand-canonical energy (eV) |
| forces | `atoms.get_forces()` | `(n_atoms, 3)` | grand-canonical forces (eV/Å) |
| `fermi` | `atoms.info` | float | applied electrode potential (eV) |
| `charge` | `atoms.info` | float | total charge of the cell (e) |

Optionally for masked supervision with per-atom charges: 

| Key | Location | Type | Description |
|-----|----------|------|-------------|
| `charge_mask` | `atoms.info` | float | `1` if per-atom charges are provided for this structure, `0.0` otherwise |
| `charge_i` | `atoms.arrays` | `(n_atoms,)` | reference per-atom charges, e |


## Fine-tuning

[`run_finetune.py`](run_finetune.py) fine-tunes a checkpoint on a GC-QEq dataset.

```bash
# Example
python run_finetune.py \
    --run_name test_run \
    --train_data_path ./train_data_path \
    --valid_data_path ./valid_data_path \
    --load_model_path ./pretrained_models/mattersim-v1.0.0-1M.pth \
    --save_path ./results \
    --epochs 500 \
    --batch_size 4 \
    --lr 0.5e-4 \
    -- scheduler ReduceLROnPlateau
    --include_forces --include_charges --no-include_stresses \
    --force_loss_ratio 10.0 \
    --charge_loss_ratio 5.0 \
    --charge_i_loss_ratio 1.0
    --save-checkpoint
```

## Usage

`MatterSimCalculator` takes an additional `fermi_value` argument.

```python
import torch
from ase.io import read
from mattersim.forcefield import MatterSimCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"

atoms = read("your_slab.extxyz")
atoms.calc = MatterSimCalculator.from_checkpoint(
    load_path="results/best_model.pth",
    fermi_value=-4.5,       # applied Fermi level in eV
    device=device,
)

print("Energy (eV)      =", atoms.get_potential_energy())
print("Forces (eV/A)    =", atoms.get_forces())
print("Atomic charges   =", atoms.get_charges())
```


## Modifications made from `mattersim`

| Path | Contents |
|------|----------|
| [src/mattersim/forcefield/m3gnet/modules/GCQEqSolver.py](src/mattersim/forcefield/m3gnet/modules/GCQEqSolver.py) | Ewald summation and grand-canonical charge equilibration solver |
| [src/mattersim/forcefield/m3gnet/modules/init_elemental_properties.py](src/mattersim/forcefield/m3gnet/modules/init_elemental_properties.py) | elemental `chi`, `J` and covalent radius `sigma` initializers |
| [src/mattersim/forcefield/m3gnet/m3gnet.py](src/mattersim/forcefield/m3gnet/m3gnet.py) | M3GNet backbone with the four GC-QEq heads |
| [src/mattersim/forcefield/potential.py](src/mattersim/forcefield/potential.py) | training loop, charge losses, ASE calculator |
| [src/mattersim/datasets/utils/](src/mattersim/datasets/utils/) | graph conversion and dataloaders carrying charge / Fermi labels |
| [run_finetune.py](run_finetune.py) | Python script for fine-tuning |