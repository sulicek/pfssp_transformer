from pfs_net import generate_random_valid_solution as pfs_rand

from tools import  load_test_cases, expand_machines, save_test_case_results

from types import SimpleNamespace as sn

import time

import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

cpu_device = torch.device("cpu")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


from pfs_net import compute_tour_length as compute_tour_length_pfs
from cplex import CplexFS

cplex = CplexFS(log=False)

torch.manual_seed(0)
info = sn()
results = []

# load the test case
test_cases = "taillard"
info.name = test_cases
test_cases = load_test_cases(test_cases, torch.device("cpu"))
cplex_time_override = 60*10 # in seconds, zero if we use the one provided in the test case

def announce(name):
    res = sn()
    # result data for current test case
    res.l_mean = l.mean().clone()
    res.l_ratio_mean = ((l.mean()-l_opt.mean()) / l_opt.mean()).clone()
    res.l = l.clone()
    res.l_ratio = ((l-l_opt) / l_opt).clone()
    res.times = times

    setattr(result, name, res)

    print(f"{name}: {res.l_mean:.2f}, {res.l_ratio_mean * 100:.2f}%")
    print(f" l:{res.l}")
    print(f" %:{res.l_ratio*100}")
    print(f" times: {res.times}")

for x in test_cases:
    cplex.time_limit = cplex_time_override
    print(f"Cplex time limit: {cplex.time_limit}")
    result = sn()
    print(x.shape)
    try:
        t, times = cplex.run_cplex_pfs_batched(x,None)
        l = compute_tour_length_pfs(x, None, t)
        l_opt = l
        announce("cplex")
    except AttributeError:
        setattr(result, "cplex", None)
        print("cplex was unable to find a solution")

    results.append(result)
    save_test_case_results(test_cases, results, info,f"cbase{cplex.time_limit}_{info.name}")
