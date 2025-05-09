import torch
import torch.nn.functional as F
"""
Loss functions used for flow matching.
In the frameework of flow matching, the loss functions need to Bregmann divergence


"""


class MSEFlowMatchingLoss:
    """
    Mean Squared Error (MSE) loss for flow matching.
    """

    def __init__(self):
        pass

    def __call__(self, v, v_theta):
        """
        Compute the MSE loss.

        Args:
            v: Tensor (bs, ...), the estimated velocity
            v_theta: Tensor (bs, ...), the target velocity
        Returns:
            loss: Tensor (bs,), the MSE loss
        """
        return F.mse_loss(v, v_theta, reduction="mean")