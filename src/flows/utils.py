import torch
import numpy as np

def _to_tensor_scalar(t, device=None, dtype=torch.float32):
    if isinstance(t, torch.Tensor):
        return t.to(device=device, dtype=dtype) if t.ndim == 0 else t
    if isinstance(t, (int, float, np.number)):
        return torch.tensor(t, device=device, dtype=dtype)
    raise TypeError(f"Unsupported scalar type: {type(t)}")
