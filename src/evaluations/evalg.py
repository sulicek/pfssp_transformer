import time
import warnings
from types import SimpleNamespace as sn

import torch
from pfs_net import generate_random_valid_solution as pfs_rand
from tools import (
    expand_machines,
    flowshop_lb_askin_standridge,
    get_project_root_dir,
    load_eval_test_case_results,
    load_test_cases,
    save_test_case_results,
)

warnings.filterwarnings("ignore", category=UserWarning)

cpu_device = torch.device("cpu")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Gpu device: {device}")


from models.repository import (
    linear_model,
    lje_model,
    pfsnet20_ln_model,
    pfsnetgen_ln_model,
    improve_by_candidates,
)
from pfs_net import compute_tour_length as compute_tour_length_pfs
from pfs_net import generate_random_valid_solution

linear = linear_model(device)
lje = lje_model(device)

pfsnet20_ln = pfsnet20_ln_model(device)
pfsnetgen_ln = pfsnetgen_ln_model(device)

torch.manual_seed(0)
info = sn()
results = []

info.nn_samples = 64

# load the test case
test_cases = "taillard"
test_cases = "VRF_large"
info.name = test_cases
cplex_prefix = "cbase600"
test_cases, cplex_results, info_cplex_baseline = load_eval_test_case_results(
    f"{cplex_prefix}_{test_cases}"
)

manual_offset = 0

print(f"nn_samples: {info.nn_samples}")


def benchmark(msg=""):
    global stamp
    if msg == "":
        stamp = time.perf_counter()
    else:
        passed = time.perf_counter() - stamp
        print(f"{msg}:{passed:.2f}")
        return passed


def announce(name: str, data: torch.Tensor):
    res = sn()
    # result data for current test case
    res.times = times
    res.tour = t
    res.l_mean = l.mean().clone()
    res.l = l.clone()
    if l_opt is not None:
        res.l_ratio_mean = ((l.mean() - l_opt.mean()) / l_opt.mean()).clone()
        res.l_ratio = ((l - l_opt) / l_opt).clone()
    else:
        res.l_ratio_mean = float("nan")
        res.l_ratio = None
    # res.tc = data

    setattr(result, name, res)

    print(f"{name}: {res.l_mean:.2f}, {res.l_ratio_mean * 100:.2f}%")
    print(f" l:{res.l}")
    if res.l_ratio is not None:
        print(f" %:{res.l_ratio * 100}")
    print(f" times: {res.times}")


for i, x in enumerate(test_cases[manual_offset:]):
    i += manual_offset
    result = sn()
    print(x.shape)
    l_opt = cplex_results[i].cplex.l if cplex_results[i].cplex is not None else None

    # pfsnetgen_ln
    with torch.no_grad():
        # move to gpu
        x_e = x.to(device)
        benchmark()
        t, _ = improve_by_candidates(
            pfsnetgen_ln, x_e / 100, info.nn_samples, compute_tour_length_pfs
        )
        times = benchmark("model run")

        benchmark()
        l = compute_tour_length_pfs(x_e, None, t)
        benchmark("tour length")

        t = t.to(cpu_device)
        l = l.to(cpu_device)
        announce("pfsnetgen_ln", x.clone())

    # pfsnetgen_ln no eval
    with torch.no_grad():
        # move to gpu
        x_e = x.to(device)
        benchmark()
        t, _ = improve_by_candidates(
            pfsnetgen_ln,
            x_e / 100,
            info.nn_samples,
            compute_tour_length_pfs,
            do_eval=False,
        )
        times = benchmark("model run")

        benchmark()
        l = compute_tour_length_pfs(x_e, None, t)
        benchmark("tour length")

        t = t.to(cpu_device)
        l = l.to(cpu_device)
        announce("pfsnetgen_ln no eval", x.clone())

    # pfsnetgen_bn
    with torch.no_grad():
        # move to gpu
        x_e = x.to(device)
        benchmark()
        t, _ = improve_by_candidates(lje, x_e / 100, info.nn_samples, compute_tour_length_pfs)
        times = benchmark("model run")

        benchmark()
        l = compute_tour_length_pfs(x_e, None, t)
        benchmark("tour length")

        t = t.to(cpu_device)
        l = l.to(cpu_device)
        announce("transformer embed model", x.clone())

    # pfsnetgen_at
    with torch.no_grad():
        # move to gpu
        x_e = x.to(device)
        benchmark()
        t, _ = improve_by_candidates(
            lje, x_e / 100, info.nn_samples, compute_tour_length_pfs, do_eval=False
        )
        times = benchmark("model run")

        benchmark()
        l = compute_tour_length_pfs(x_e, None, t)
        benchmark("tour length")

        t = t.to(cpu_device)
        l = l.to(cpu_device)
        announce("transformer embed model no eval", x.clone())
    lje = lje_model(device)

    if x.shape[2] <= 20:
        # pfsnet20_bn
        with torch.no_grad():
            # move to gpu
            x_e = x.to(device)
            x_e = expand_machines(x_e, 20)

            benchmark()
            t, _ = improve_by_candidates(
                linear, x_e / 100, info.nn_samples, compute_tour_length_pfs
            )
            times = benchmark("model run")

            benchmark()
            l = compute_tour_length_pfs(x_e, None, t)
            benchmark("tour length")

            t = t.to(cpu_device)
            l = l.to(cpu_device)
            announce("transformer linear model", x.clone())

        # pfsnet20_at
        with torch.no_grad():
            # move to gpu
            x_e = x.to(device)
            x_e = expand_machines(x_e, 20)

            benchmark()
            t, _ = improve_by_candidates(
                linear,
                x_e / 100,
                info.nn_samples,
                compute_tour_length_pfs,
                do_eval=False,
            )
            times = benchmark("model run")

            benchmark()
            l = compute_tour_length_pfs(x_e, None, t)
            benchmark("tour length")

            t = t.to(cpu_device)
            l = l.to(cpu_device)
            announce("transformer linear model no eval", x.clone())
        linear = linear_model(device)

        # pfsnet20_ln
        with torch.no_grad():
            # move to gpu
            x_e = x.to(device)
            x_e = expand_machines(x_e, 20)

            benchmark()
            t, _ = improve_by_candidates(
                pfsnet20_ln,
                x_e / 100,
                info.nn_samples,
                compute_tour_length_pfs,
            )
            times = benchmark("model run")

            benchmark()
            l = compute_tour_length_pfs(x_e, None, t)
            benchmark("tour length")

            t = t.to(cpu_device)
            l = l.to(cpu_device)
            announce("pfsnet20_ln", x.clone())

        # pfsnet20_ln no eval
        with torch.no_grad():
            # move to gpu
            x_e = x.to(device)
            x_e = expand_machines(x_e, 20)

            benchmark()
            t, _ = improve_by_candidates(
                pfsnet20_ln,
                x_e / 100,
                info.nn_samples,
                compute_tour_length_pfs,
                do_eval=False,
            )
            times = benchmark("model run")

            benchmark()
            l = compute_tour_length_pfs(x_e, None, t)
            benchmark("tour length")

            t = t.to(cpu_device)
            l = l.to(cpu_device)
            announce("pfsnet20_ln no eval", x.clone())

    # better lowerbound model
    benchmark()
    t = generate_random_valid_solution(x, 1)
    l = flowshop_lb_askin_standridge(x)
    times = benchmark("better lowerbound run")
    benchmark()
    # perform mean across the random samples for consistency
    # NOTE: this assumes the batch size is 1!
    l = l.mean()
    benchmark("tour length")
    announce("better lowerbound model", x.clone())

    # random permutation model
    random_count = 100
    benchmark()
    t = generate_random_valid_solution(x, random_count)
    times = benchmark("random model run")
    benchmark()
    l = compute_tour_length_pfs(x.repeat([random_count, 1, 1]), None, t)
    # perform mean across the random samples for consistency
    # NOTE: this assumes the batch size is 1!
    l = l.mean()
    benchmark("tour length")
    announce("random model", x.clone())

    results.append(result)
    save_test_case_results(
        test_cases,
        results,
        info,
        f"evalg{info.nn_samples}_{cplex_prefix}_{info.name}",
    )
