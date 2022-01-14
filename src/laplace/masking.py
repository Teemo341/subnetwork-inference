import copy
import torch

from src.utils import get_loss_func, get_n_params, unflatten_nn
from src.baselines.swag import fit_swag


def get_mask_from_weight_score_vec(model, weight_score_vec, n_weights_subnet):
    """ compute mask based on the given (descending) weight score vector """
    assert torch.all(weight_score_vec >= 0)

    idx = torch.argsort(weight_score_vec, descending=True)[:n_weights_subnet]
    idx = idx.sort()[0]
    mask_vec = torch.zeros_like(weight_score_vec)
    mask_vec[idx] = 1.

    # define layer-wise masks based on this threshold, to then prune weights with it
    mask = unflatten_nn(model, mask_vec)
    return mask, idx, weight_score_vec


def random_mask(model, n_weights_subnet, device='cpu'):
    """ Compute a subnetwork mask uniformly at random.

    Args:
        model: the model to mask.
        n_weights_subnet: number of model weights to keep for the subnetwork.
    Returns:
        corresponding mask
    """

    weight_score_vec = torch.rand(get_n_params(model), device=device)
    return get_mask_from_weight_score_vec(model, weight_score_vec, n_weights_subnet)


def wasserstein_mask(model, n_weights_subnet, train_loader, device, loss="cross_entropy", n_snapshots=256, swag_lr=1e-2, swag_c_epochs=1, swag_c_batches=None, parallel=False):
    # compute weight score vector required for Wasserstein pruning strategy using SWAG for variance estimation
    loss_func = get_loss_func(loss).to(device)
    swag_model = fit_swag(copy.deepcopy(model), device, train_loader, loss_func, diag_only=True, max_num_models=n_snapshots, swa_lr=swag_lr, swa_c_epochs=swag_c_epochs, swa_c_batches=swag_c_batches, parallel=parallel)
    weight_score_vec = swag_model.get_variance_vector()
    return get_mask_from_weight_score_vec(model, weight_score_vec, n_weights_subnet)
