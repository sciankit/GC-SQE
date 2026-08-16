# -*- coding: utf-8 -*-
import argparse
import os
import pickle as pkl
import random

import numpy as np
import torch
import torch.distributed
import wandb
from ase.units import GPa

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.scaling import AtomScaling
from mattersim.forcefield.potential import Potential
from mattersim.utils.atoms_utils import AtomsAdaptor
from mattersim.utils.logger_utils import get_logger

logger = get_logger()

is_distributed = all(k in os.environ for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
local_rank = int(os.environ.get("LOCAL_RANK", 0))


def main(args):
    if is_distributed:
        if args.device == "cuda":
            torch.distributed.init_process_group(backend="nccl")
        else:
            torch.distributed.init_process_group(backend="gloo")
    else:
        logger.info(
            "LOCAL_RANK not set, running in single-process mode. "
            "Use `torchrun --nproc_per_node=N finetune.py ...` for distributed training."  # noqa: E501
        )
    args_dict = vars(args)
    if args.wandb and local_rank == 0:
        wandb_api_key = (
            args.wandb_api_key
            if args.wandb_api_key is not None
            else os.getenv("WANDB_API_KEY")
        )
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=args,
            # id=args.run_name,
            # resume="allow",
        )

    if args.wandb:
        args_dict["wandb"] = wandb

    if is_distributed:
        torch.distributed.barrier()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.set_device(local_rank)

    if args.train_data_path.endswith(".pkl"):
        with open(args.train_data_path, "rb") as f:
            atoms_train = pkl.load(f)
    else:
        atoms_train = AtomsAdaptor.from_file(filename=args.train_data_path)
    g_energies = []
    fermi_values = []
    forces = [] if args.include_forces else None
    stresses = [] if args.include_stresses else None
    charges = [] if args.include_charges else None
    charge_i = [] if args.include_charges else None
    charge_mask = [] if args.include_charges else None
    logger.info("Processing training datasets...")
    for atoms in atoms_train:
        g_energies.append(atoms.info["g_energy"])
        fermi_values.append(atoms.info["fermi"])
        if args.include_forces:
            forces.append(atoms.get_forces())
        if args.include_stresses:
            stresses.append(atoms.get_stress(voigt=False) / GPa)  # convert to GPa
        if args.include_charges:
            charges.append(atoms.info["charge"])
            charge_i.append(atoms.arrays["charge_i"])
            charge_mask.append(atoms.info["charge_mask"])

    dataloader = build_dataloader(
        atoms_train,
        g_energies,
        charges,
        charge_i,
        charge_mask,
        fermi_values,
        forces,
        stresses,
        shuffle=True,
        pin_memory=(args.device == "cuda"),
        is_distributed=is_distributed,
        **args_dict,
    )

    device = args.device
    # build energy normalization module
    if args.re_normalize:
        scale = AtomScaling(
            atoms=atoms_train,
            total_energy=g_energies,
            forces=forces,
            verbose=True,
            **args_dict,
        ).to(device)

    if args.valid_data_path is not None:
        if args.valid_data_path.endswith(".pkl"):
            with open(args.valid_data_path, "rb") as f:
                atoms_val = pkl.load(f)
        else:
            atoms_val = AtomsAdaptor.from_file(filename=args.valid_data_path)
        g_energies = []
        fermi_values = []
        forces = [] if args.include_forces else None
        stresses = [] if args.include_stresses else None
        charges = [] if args.include_charges else None
        charge_i = [] if args.include_charges else None
        charge_mask = [] if args.include_charges else None
        logger.info("Processing validation datasets...")
        for atoms in atoms_val:
            g_energies.append(atoms.info["g_energy"])
            fermi_values.append(atoms.info["fermi"])
            if args.include_forces:
                forces.append(atoms.get_forces())
            if args.include_stresses:
                stresses.append(atoms.get_stress(voigt=False) / GPa)  # convert to GPa
            if args.include_charges:
                charges.append(atoms.info["charge"])
                charge_i.append(atoms.arrays["charge_i"])
                charge_mask.append(atoms.info["charge_mask"])
        val_dataloader = build_dataloader(
            atoms_val,
            g_energies,
            charges,
            charge_i,
            charge_mask,
            fermi_values,
            forces,
            stresses,
            pin_memory=(args.device == "cuda"),
            is_distributed=is_distributed,
            **args_dict,
        )
    else:
        val_dataloader = None

    potential = Potential.from_checkpoint(
        load_path=args.load_model_path,
        load_training_state=False,
        **args_dict,
    )

    if args.re_normalize:
        potential.model.set_normalizer(scale)

    if is_distributed:
        if args.device == "cuda":
            potential.model = torch.nn.parallel.DistributedDataParallel(potential.model)
        torch.distributed.barrier()

    potential.train_model(
        dataloader,
        val_dataloader,
        loss=torch.nn.HuberLoss(delta=0.01),
        is_distributed=is_distributed,
        **args_dict,
    )

    if local_rank == 0 and args.save_checkpoint and args.wandb:
        wandb.save(os.path.join(args.save_path, "best_model.pth"))

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path parameters
    parser.add_argument(
        "--run_name", type=str, default="example", help="name of the run"
    )
    parser.add_argument(
        "--train_data_path", type=str, required=True, help="train data path"
    )
    parser.add_argument(
        "--valid_data_path", type=str, default=None, help="valid data path"
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default="./pretrained_models/mattersim-v1.0.0-1M.pth",
        help="path to load the model",
    )
    parser.add_argument(
        "--save_path", type=str, default="./results", help="path to save the model"
    )
    parser.add_argument(
        "--save_checkpoint",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=10,
        help="save checkpoint every ckpt_interval epochs",
    )
    parser.add_argument("--device", type=str, default="cuda")

    # model parameters
    parser.add_argument("--cutoff", type=float, default=5.0, help="cutoff radius")
    parser.add_argument(
        "--threebody_cutoff",
        type=float,
        default=4.0,
        help="cutoff radius for three-body term, which should be smaller than cutoff (two-body)",  # noqa: E501
    )

    # training parameters
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.5e-4)
    parser.add_argument(
        "--scheduler", 
        type=str, 
        default="ReduceLROnPlateau",
        help="ReduceLROnPlateau, StepLR, or CosineAnnealingLR")
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="step epoch for learning rate scheduler",
    )
    parser.add_argument(
        "--include_forces",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--include_stresses",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--include_charges",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--force_loss_ratio", type=float, default=10.0)
    parser.add_argument("--stress_loss_ratio", type=float, default=0.1)
    parser.add_argument("--charge_loss_ratio", type=float, default=5.0)
    parser.add_argument("--charge_i_loss_ratio", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # scaling parameters
    parser.add_argument(
        "--re_normalize",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="re-normalize the energy and forces according to the new data",
    )
    parser.add_argument("--scale_key", type=str, default="per_species_forces_rms")
    parser.add_argument(
        "--shift_key", type=str, default="per_species_energy_mean_linear_reg"
    )
    parser.add_argument("--init_scale", type=float, default=None)
    parser.add_argument("--init_shift", type=float, default=None)
    parser.add_argument(
        "--trainable_scale",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--trainable_shift",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    # wandb parameters
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="wandb_test")
    args = parser.parse_args()
    main(args)

## Minimal Example
# python finetune.py --run_name test \
# --train_data_path ./train_data_path \
# --valid_data_path ./valid_data_path \
# --device cuda \
# --epochs 10 \
# --batch_size 4 \
# --save_checkpoint