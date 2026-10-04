from pathlib import Path

import torch
#bn models
from models.linear.model_impl import get_model as _get_linear
from models.lje.model_impl import get_model as _get_lje

# ln models
from models.pfsnet20_ln.model_impl import get_model as _get_pfsnet20_ln
from models.pfsnetgen_ln.model_impl import get_model as _get_pfsnetgen_ln

# load models from their repository


def _instantiate_model(get_model_fn, weights_src, device):
    model = get_model_fn()
    model = model.to(device)
    current_dir = Path(__file__).resolve().parent.parent
    weights_src = Path(weights_src)
    weights = torch.load(
        (current_dir / weights_src).resolve(), map_location=device, weights_only=False
    )
    model.load_state_dict(weights["model_baseline"])
    del weights
    return model


def pfsnetgen_ln_model(device):
    nobo = _instantiate_model(
        _get_pfsnetgen_ln, "models/pfsnetgen_ln/model.pkl", device
    )
    return nobo


def pfsnet20_ln_model(device):
    nobo = _instantiate_model(_get_pfsnet20_ln, "models/pfsnet20_ln/model.pkl", device)
    return nobo


def linear_model(device):
    model = _instantiate_model(_get_linear, "models/linear/model.pkl", device)
    return model


def lje_model(device):
    model = _instantiate_model(_get_lje, "models/lje/model.pkl", device)
    return model

def improve_by_candidates(model, x_in: torch.Tensor, n_candidates, loss_fn, do_eval=True):
    bs = x_in.shape[0]

    # duplicate bs to samples and flatten to bs
    x = x_in.unsqueeze(0).repeat(n_candidates, 1, 1, 1)
    x = torch.flatten(x, 0, 1)

    # run the model on samples x bs
    if do_eval:
        model.eval()
    res, _ = model.forward(x, deterministic=False)

    # evaluate sample results
    res_l = loss_fn(x, None, res)

    # reshape back to samples, samples now on "1th" dimension
    res = res.reshape(n_candidates, res.shape[0] // n_candidates, *res.shape[1:]).permute(
        1, 0, 2
    )
    res_l = res_l.reshape(
        n_candidates, res_l.shape[0] // n_candidates, *res_l.shape[1:]
    ).permute(1, 0)

    # pick the best sample for each batch
    best_sample = torch.argmin(res_l, dim=1)

    # compute the average loss for each sample
    avg_sample = torch.mean(res_l, dim=1)

    # select the best samples
    # btp_l = res_l[torch.arange(bs), best_sample] # for comparison with btp_l
    bt = res[torch.arange(bs), best_sample]

    # ensure the best sample has the matching losses,
    # since we are selecting from a different tensor
    bt_l = loss_fn(x_in, None, bt)

    # print("improvement", (avg_sample - bt_l) / bt_l)
    if do_eval:  # unset eval mode
        model.train()
    return bt, None  # return None to keep consistent with model signature
