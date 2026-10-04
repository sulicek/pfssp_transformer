from pathlib import Path

import torch
import time

# x: (bs, nb_nodes, n_machines)
def flowshop_lb_askin_standridge(x):
    bsz = x.shape[0]
    n_jobs = x.shape[1]
    n_machines = x.shape[2]
    lbf = torch.zeros([bsz], device=x.device)
    for b in range(bsz):
        lb = torch.zeros([bsz, n_machines], device=x.device)
        lb[b] = x[b].sum(0)
        for m in range(n_machines):
            # we take the smallest job -
            if m != 0:
                # across all prevous machines
                lb[b, m] += x[b][:, :m].sum(1).min(0)[0]
            if m != n_machines - 1:
                # across all following machines
                lb[b, m] += x[b][:, m + 1 :].sum(1).min(0)[0]
        # now we pick the machine with the highest lb
        lbf[b] = lb[b].max(0)[0]

    return lb


# x shape bs nb_nodes, n_machines
def expand_machines(x, expand_to):
    m = x.shape[2]
    if expand_to == m:
        return x
    assert expand_to > m
    diff = expand_to - m
    exp = torch.zeros((x.shape[0], x.shape[1], diff), device=x.device)
    res = torch.cat((x, exp), dim=-1)
    return res

# Load saved data for evaluation
def load_test_cases(path, device):
    data = torch.load(f"evaluations/data/{path}.pkl", weights_only=False)
    tcs = data["test_cases"]
    return tcs

# Save evaluated test case data with results
def save_test_case_results(tcs, results, info, name):
    path = get_project_root_dir() / (f"evaluations/results/{name}.pkl")
    torch.save({"test_cases": tcs, "results": results, "info": info}, path)
    print(f'Result saved as "{path}"')


def load_eval_test_case_results(name):
    root_dir = Path(__file__).resolve().parent
    path = Path(f"evaluations/results/{name}.pkl")
    data = torch.load(
        root_dir / path, weights_only=False, map_location=torch.device("cpu")
    )
    tcs = data["test_cases"]
    results = data["results"]
    info = data["info"]
    return tcs, results, info

def build_warm_start_pfs(sequence, operations, op_durations, buildover_times, model):
    # Create warm start
    warm_start = model.create_empty_solution()
    NB_MACHINES = len(operations[0])
    if len(sequence) > 0:
        for m in range(NB_MACHINES):
            crnt_time = 0
            last_job = -1
            for j in sequence:  # go through all elements of the ws permutation
                # offset the current time (wait for):
                crnt_time = max(
                    crnt_time + buildover_times[m][last_job][j]
                    if last_job != -1 and buildover_times is not None
                    else 0,  # buildover from last job
                    warm_start[operations[j][m - 1]].end
                    if m != 0
                    else 0,  # this job on last machine to finish
                    crnt_time,
                )  # if none applicable, we wait for nothing

                # time when the current job ends
                end_time = crnt_time + int(op_durations[j][m])

                # set the warm start interval variable
                warm_start[operations[j][m]] = (
                    crnt_time,
                    end_time,
                    end_time - crnt_time,
                    end_time - crnt_time,
                )  # ( start_time end_time size length)

                # set helper variables
                crnt_time = end_time
                last_job = j
    return warm_start

def get_project_root_dir() -> Path:
    root_dir = Path(__file__).resolve().parent
    return root_dir
