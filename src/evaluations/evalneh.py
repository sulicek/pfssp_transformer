import time
import warnings
from types import SimpleNamespace as sn

import torch
from neh import run_neh_pfs_batched
from pfs_net import compute_tour_length as compute_tour_length_pfs
from tools import load_test_cases, save_test_case_results

warnings.filterwarnings("ignore", category=UserWarning)

cpu_device = torch.device("cpu")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

torch.manual_seed(0)
info = sn()
results = []

# load the test case
test_cases = "taillard"
test_cases = "VRF_large"
info.name = test_cases
test_cases = load_test_cases(test_cases, torch.device("cpu"))

# Only compute NEH on the (n_jobs, n_machines) sizes reported for this dataset
# in Table 2 of arXiv:2210.17178 (the paper's Taillard/VRF benchmark comparison).
target_sizes = {
    "taillard": {(50, 5), (100, 5), (100, 20), (200, 20), (500, 20)},
    "VRF_large": {(600, 20), (700, 20), (800, 20)},
}[info.name]


def announce(name):
    res = sn()
    res.l_mean = l.mean().clone()
    res.l = l.clone()
    res.tour = t
    res.times = times

    setattr(result, name, res)

    print(f"{name}: {res.l_mean:.2f}")
    print(f" l:{res.l}")
    print(f" times: {res.times}")


# placeholder for instances outside target_sizes, so results stays index-aligned
# with test_cases (downstream code matches eval files by raw order/index)
def skip(name):
    res = sn()
    res.l_mean = torch.tensor(-1.0)
    res.l = torch.tensor([-1.0])
    res.tour = None
    res.times = -1.0
    setattr(result, name, res)


for i, x in enumerate(test_cases):
    result = sn()
    size = (x.shape[1], x.shape[2])

    if size not in target_sizes:
        skip("neh")
    else:
        print(x.shape)
        start = time.perf_counter()
        t = run_neh_pfs_batched(x)
        times = time.perf_counter() - start

        l = compute_tour_length_pfs(x, None, t)
        announce("neh")

    results.append(result)
    save_test_case_results(test_cases, results, info, f"evalneh_{info.name}")
