import time
import warnings
from types import SimpleNamespace as sn

import torch
from cplex import CplexFS
from docplex.cp.solver.cpo_callback import CpoCallback
from pfs_net import compute_tour_length as compute_tour_length_pfs
from pfs_net import generate_random_valid_solution as pfs_rand
from tools import (
    expand_machines,
    load_eval_test_case_results,
    load_test_cases,
    save_test_case_results,
)
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

cpu_device = torch.device("cpu")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# Custom callback
class StopAtGoalCallback(CpoCallback):
    def __init__(self, target):
        self.target = target
        self.report = True

    def invoke(self, solver, event, sres):
        if event == "Solution":
            obj = sres.get_objective_value()
            if self.report:
                tqdm.write(f"Found solution with objective {obj}, ?= {self.target}")
            if obj <= self.target:
                if self.report:
                    tqdm.write(f"Target {self.target} reached, stopping solver.")
                solver.abort_search()  # stops the search immediately


cplex = CplexFS(log=False)

torch.manual_seed(0)
info = sn()
results = []

# load the test case
cplex_results_name = "cbase600_taillard"
cplex_results_name = "cbase600_VRF_large"
_, cplex_results, _ = load_eval_test_case_results(cplex_results_name)

evalg_results_name = "evalg64"  # configured manually
test_cases, evalg_results, _ = load_eval_test_case_results(
    f"{evalg_results_name}_{cplex_results_name}"
)

info.name = f"{evalg_results_name}_{cplex_results_name}"
cplex_time_override = (
    600  # in seconds, zero if we use the one provided in the test case
)


def announce(name):
    res = sn()
    # result data for current test case
    res.l_mean = l.mean().clone()
    res.l_ratio_mean = ((l.mean() - l_opt.mean()) / l_opt.mean()).clone()
    res.l = l.clone()
    res.l_ratio = ((l - l_opt) / l_opt).clone()
    res.times = times
    res.times_save = og_cplex_time - res.times

    setattr(result, name, res)

    print(f"{name}: {res.l_mean:.2f}, {res.l_ratio_mean * 100:.2f}%")
    print(f" l:{res.l}")
    print(f" %:{res.l_ratio * 100}")
    print(f" times: {res.times}")
    print(f" times save: {res.times_save}")


for i, x in enumerate(test_cases):
    cplex.time_limit = cplex_time_override
    print(f"Cplex time limit: {cplex.time_limit}")
    result = sn()

    # find the  transformer tour
    model_name = "transformer embed model no eval"
    ws = evalg_results[i].__dict__[model_name].tour
    makespan_target = cplex_results[i].cplex.l[0]

    # pull original cplex times
    og_cplex_time = cplex_results[i].__dict__["cplex"].times

    # set callback
    cplex.solution_callback = StopAtGoalCallback(makespan_target)
    print(x.shape)

    try:
        t, times = cplex.run_cplex_pfs_batched(x, None, warm_start_sequence=ws)
        l = compute_tour_length_pfs(x, None, t)
        l_opt = l
        announce("cplex ws")
    except AttributeError:
        setattr(result, "cplex ws", None)
        print("cplex was unable to find a solution")

    try:
        t, times = cplex.run_cplex_pfs_batched(x, None)
        l = compute_tour_length_pfs(x, None, t)
        l_opt = l
        announce("cplex")
    except AttributeError:
        setattr(result, "cplex", None)
        print("cplex was unable to find a solution")

    results.append(result)
    save_test_case_results(
        test_cases, results, info, f"ws{cplex.time_limit}_{info.name}"
    )
