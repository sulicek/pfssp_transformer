# Ported from https://github.com/xbresson/TSP_Transformer

import argparse
import datetime
import os
import pprint
import shutil
import time
import warnings
from pathlib import Path
from random import randint
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from tools import expand_machines, get_project_root_dir
from torch.distributions.categorical import Categorical
from torch.optim.optimizer import ParamsT
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

torch.autograd.set_detect_anomaly(True)

###################
# Hardware : CPU / GPU(s)
###################

device = torch.device("cpu")
gpu_id = "0"  # select a single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU name: {:s}, gpu_id: {:s}".format(torch.cuda.get_device_name(0), gpu_id))

print(f"Device: {device}")

# parsing input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Name of the training instance", default="test")
parser.add_argument("--bs", help="Batch size", default=None)

cmd_args = parser.parse_args()
name = cmd_args.name


# Since the training data is synthetic, we have a sampling function instead of a dataset
def get_data_batch():
    chosen_n_machines = args.n_machines
    chosen_nb_nodes = args.nb_nodes

    problem = torch.rand(args.bsz, chosen_nb_nodes, chosen_n_machines, device=device)

    return problem, args.nb_nodes, args.n_machines


args = SimpleNamespace()

# problem description
args.n_machines = 20
args.nb_nodes = 50

# training parameters
args.lr = 1e-5
args.tol = 1e-3

# number of epochs
args.nb_epochs = 40

# dataset setting
args.bsz = 256
args.nb_batch_per_epoch = 2000
args.nb_batch_eval = 300


# make much smaller training if we're just testing the model
if name == "test":
    args.bsz = 12
    args.nb_epochs = 2
    args.nb_batch_per_epoch = 2
    args.nb_batch_eval = 2

if cmd_args.bs:
    args.bsz = int(cmd_args.bs)

args.gpu_id = gpu_id


print(f"Instance name: {name}")
print("___Hyper Parameters___")
for k, v in args.__dict__.items():
    print(f"{k}: {v}")


# import the model
from pfs_net import compute_tour_length, generate_random_valid_solution, get_model

model_train = get_model()

model_baseline = get_model()

optimizer = torch.optim.Adam(model_train.parameters(), lr=args.lr)

model_train = model_train.to(device)
model_baseline = model_baseline.to(device)
model_baseline.eval()

# Create log file, assuming the following structure
# my_project/
# ├── logs/                 - Auto-generated: stores all experiment outputs
# │   └── experiment_01/    - Sub-folder per run
# │       ├── pylog.log     - Python logs
# │       ├── screenlog.log - Screen logs (created through .sh scripts)
# │       └── model.pt      - PyTorch checkpoints
# └── src/                  - Source code
#     └── train.py          - this script

root_dir = get_project_root_dir()
logs_dir = root_dir.parent / "logs" / name
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "pylog.log"
file = open(log_file, "w", 1)
file.write(name + "\n\n")

for arg in vars(args):
    file.write(arg)
    hyper_param_val = "={}".format(getattr(args, arg))
    file.write(hyper_param_val)
    file.write("\n")
file.write("\n\n")

start_training_time = time.time()
for epoch in tqdm(range(0, args.nb_epochs), desc="Epochs:", position=0, leave=True):
    start = time.time()
    model_train.train()

    makespan_calculation_time = 0
    for step in range(1, args.nb_batch_per_epoch + 1):
        x, n_jobs, n_machines = get_data_batch()

        makespan_train, sumLogProbOfActions = model_train(x, deterministic=False)

        with torch.no_grad():
            makespan_baseline, _ = model_baseline(x, deterministic=True)

        makespan_calculation_start = time.perf_counter()
        # get the lengths of the tours
        L_train = compute_tour_length(x, None, makespan_train)  # size(L_train)=(bsz)
        L_baseline = compute_tour_length(
            x, None, makespan_baseline
        )
        makespan_calculation_time += time.perf_counter() - makespan_calculation_start

        loss = torch.mean((L_train - L_baseline) * sumLogProbOfActions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    time_one_epoch = time.time() - start
    time_tot = time.time() - start_training_time

    model_train.eval()
    mean_makespan_train = 0
    mean_makespan_baseline = 0
    mean_makespan_random = 0
    for step in range(0, args.nb_batch_eval):
        # generate a batch of random tsp instances
        x, n_jobs, n_machines = get_data_batch()

        # compute tour for model and baseline
        with torch.no_grad():
            makespan_train, _ = model_train(x, deterministic=True)
            makespan_baseline, _ = model_baseline(x, deterministic=True)
            tour_random = generate_random_valid_solution(x)

        # get the lengths of the tours
        makespan_calculation_start = time.perf_counter()
        L_train = compute_tour_length(x, None, makespan_train)
        L_baseline = compute_tour_length(x, None, makespan_baseline)
        L_random = compute_tour_length(x, None, tour_random)
        makespan_calculation_time += time.perf_counter() - makespan_calculation_start

        mean_makespan_train += L_train.mean().item()
        mean_makespan_baseline += L_baseline.mean().item()
        mean_makespan_random += L_random.mean().item()

    mean_makespan_train = mean_makespan_train / args.nb_batch_eval
    mean_makespan_baseline = mean_makespan_baseline / args.nb_batch_eval
    mean_makespan_random = mean_makespan_random / args.nb_batch_eval

    # evaluate train model and baseline and update if train model is better
    update_baseline = mean_makespan_train + args.tol < mean_makespan_baseline
    if update_baseline:
        model_baseline.load_state_dict(model_train.state_dict())

    mystring_min = (
        f"Epoch: {epoch}, "
        f"tot time: {time_tot / 86400:.3f}day, "
        f"epoch time: {time_one_epoch / 60:.3f}min, "
        f"cptl time: {makespan_calculation_time / 60:.3f}min, "
        f"L_train: {mean_makespan_train:.3f}, "
        f"L_base: {mean_makespan_baseline:.3f}, "
        f"L_random: {mean_makespan_random:.3f}, "
        f"update: {update_baseline}"
    )
    tqdm.write(mystring_min)
    file.write(mystring_min + "\n")

    # Saving checkpoint
    torch.save(
        {
            "epoch": epoch,
            "time": time_one_epoch,
            "tot_time": time_tot,
            "loss": loss.item(),
            "L_average": [torch.mean(L_train).item(), torch.mean(L_baseline).item()],
            "model_baseline": model_baseline.state_dict(),
            "model_train": model_train.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        logs_dir / "model.pkl",
    )

    # Save the model python file as well for easier reproducibility
    shutil.copy(root_dir / "pfs_net.py", logs_dir / "model_impl.py")
