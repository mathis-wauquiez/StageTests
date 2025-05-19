import torch
from torch.utils.data._utils.collate import default_collate
from functools import partial

def reflow_wrapper(flow_model, base_collate=default_collate, **sampling_params):
    """
    Returns a collate_fn that returns (x0_batch, z1_batch) or (x0_batch, z1_batch, y_batch)
    with z1 sampled from the flow model. (see Reflow paper)
    """
    device = next(flow_model.parameters()).device

    def collate(batch):
        # 1) produce the “base” batch
        collated = base_collate(batch)

        if len(collated) == 2:
            x0, _ = collated
            y = None
        elif len(collated) == 3:
            x0, _, y = collated
        else:
            raise ValueError(f"Expected 2 or 3 elements in collated batch, got {len(collated)}")

        # move to device
        x0 = x0.to(device)
        if isinstance(y, torch.Tensor):
            y = y.to(device)

        # sample z1 = psi(x0, y=y?)
        with torch.no_grad():
            if y is not None:
                z1 = flow_model(x0, y=y, **sampling_params)
            else:
                z1 = flow_model(x0, **sampling_params)

        return (x0, z1) if y is None else (x0, z1, y)

    return collate


def get_reflow_collate_fn(flow_model, **sampling_params):
    return reflow_wrapper(flow_model, base_collate=default_collate, **sampling_params)

def get_reflow_stack_collate_fn(flow_models, **sampling_params):
    """
    Collate fn to get (Z^k_0, Z^k_1) or (Z^k_0, Z^k_1, y) for k flows.
    """
    base_collate = default_collate
    for flow_model in flow_models:
        base_collate = partial(reflow_wrapper, flow_model, base_collate=base_collate, **sampling_params)
    return base_collate

