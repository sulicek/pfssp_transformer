import torch
from tqdm import tqdm

from baselines.neh import neh as neh_pfssil


# x: (nb_nodes, n_machines), not batched
def run_neh_pfs(x):
    assert len(x.shape) == 2, "Shape of x must not be batched"

    pt = x.cpu().numpy()
    _, seq = neh_pfssil(pt)

    return torch.as_tensor(seq, device=x.device, dtype=torch.long)


# x: (bs, nb_nodes, n_machines)
def run_neh_pfs_batched(x, leave=True):
    bs = x.shape[0]
    tours = -torch.ones((bs, x.shape[1]), device=x.device, dtype=torch.long)
    for b in tqdm(range(bs), leave=leave, desc="NEH"):  # no option other than lazy batching, NEH is sequential per instance
        tours[b] = run_neh_pfs(x[b])
    return tours
