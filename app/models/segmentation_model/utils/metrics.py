# |----------------------------------|
# | MODEL PREDICTIONS SCORING (DICE) |
# |----------------------------------|

# First, import the necessary libraries:
import torch
from segmentation_models_pytorch.losses import DiceLoss, BINARY_MODE, MULTICLASS_MODE

# Define loss_fn class for easy access in the future:
#   1. Multiclass DiceLoss.
#       - MULTICLASS_MODE.
#       - log_loss=False - values in range [0, 1].
#       - from_logits=False - for easy access.
_dice_loss_m = DiceLoss(
    MULTICLASS_MODE, log_loss=False, from_logits=True, smooth=1e-06, eps=1e-06
)

#   2. Binary DiceLoss.
#       - BINARY_MODE.
#       - log_loss=False - values in range [0, 1].
#       - from_logits=False - for easy access.
_dice_loss_b = DiceLoss(
    BINARY_MODE, log_loss=False, from_logits=True, smooth=1e-06, eps=1e-06
)


def _calculate_dice_loss(
    logits: torch.Tensor, y_true: torch.Tensor, case: str
) -> torch.Tensor:
    if case == "overall":
        return _dice_loss_m(logits, y_true)
    elif case == "kidney":
        return _dice_loss_b(logits[:, 1, :, :].unsqueeze(1), (y_true == 1).long())
    elif case == "tumor":
        return _dice_loss_b(logits[:, 2, :, :].unsqueeze(1), (y_true == 2).long())
    elif case == "cyst":
        return _dice_loss_b(logits[:, 3, :, :].unsqueeze(1), (y_true == 3).long())
    else:
        raise AssertionError(f"{case} is not supported.")


def dice_score(
    logits: torch.Tensor, y_true: torch.Tensor, case: str = "overall"
) -> torch.Tensor:
    """
    Calculates the balanced accuracy score (Dice score) for predicted values and true labels.

    Accepts logits (torch.Tensor) of shape (B, 4, 512, 512) and y_true (torch.Tensor) of shape (B, 512, 512) or (B, 1, 512, 512).

    Args:
        logits (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True labels.
        case (str): Type of organ: {"overall", "kidney", "tumor", "cyst"}. Default: "overall".

    Returns:
        torch.Tensor: Balanced accuracy score (Dice score).

    Raises:
        AssertionError: If the task type is not supported.
    """
    assert logits.ndim == 4, f"Invalid logits.shape: {logits.shape}."
    assert case in {"overall", "kidney", "tumor", "cyst"}, f"{case} is not supported."
    return 1.0 - _calculate_dice_loss(logits, y_true, case)
