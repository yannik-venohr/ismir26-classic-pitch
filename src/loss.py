import torch


def onset_loss_with_tolerance(
    y_true: torch.Tensor,
    y_pred_logits: torch.Tensor,
    positive_weight: float = 1.0,
    label_smoothing: float = 0.0,
    tolerance_frames: int = 0,
) -> torch.Tensor:
    """
    Shift-tolerant BCE loss with temporal max-pooling.
    Args:
        y_true: (B, F, T) binary labels
        y_pred_logits: (B, F, T) logits
        positive_weight: multiplicative weight for positive samples (1.0 = no reweighting)
        label_smoothing: smoothing factor, squeezes labels towards 0.5
        tolerance_frames: number of frames for temporal tolerance in max-pooling
    Returns:
        scalar loss
    """
    y_true_smoothed = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    # directly use binary cross-entropy if no tolerance is specified
    if tolerance_frames == 0:
        pos_weight_tensor = torch.tensor(positive_weight, device=y_true.device, dtype=y_pred_logits.dtype)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred_logits,
            y_true_smoothed,
            pos_weight=pos_weight_tensor,
        )

    # else apply max-pooling for shift tolerance
    B, F, T = y_true.shape

    # --- max-pool predictions over (2 * tolerance_frames + 1) frames ---
    y_pred = torch.sigmoid(y_pred_logits)
    y_pred_max_pool = torch.nn.functional.max_pool1d(
        y_pred.reshape(B * F, 1, T),
        kernel_size=2 * tolerance_frames + 1,
        stride=1,
        padding=tolerance_frames,
    ).reshape(B, F, T)

    # --- max-pool labels over (4 * tolerance_frames + 1) frames ---
    # negatives close to an onset are ignored
    y_true_max_pool = torch.nn.functional.max_pool1d(
        y_true.reshape(B * F, 1, T),
        kernel_size=4 * tolerance_frames + 1,
        stride=1,
        padding=2 * tolerance_frames,
    ).reshape(B, F, T)

    pos_term = positive_weight * y_true_smoothed * torch.log(y_pred_max_pool.clamp(min=1e-8))

    neg_mask = 1.0 - y_true_max_pool
    neg_term = neg_mask * (1.0 - y_true_smoothed) * torch.log((1.0 - y_pred_max_pool).clamp(min=1e-8))

    loss = -(pos_term + neg_term)
    return loss.mean()


def frame_loss(
    y_true: torch.Tensor,
    y_pred_logits: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Binary cross-entropy on frame-wise note activity, with label smoothing.

    Args:
        y_true (B, F, T): Ground truth binary labels (0s and 1s).
        y_pred_logits (B, F, T): Model outputs before sigmoid activation (logits).
        label_smoothing (float): Smoothing factor. Squeezes labels towards 0.5.
    Returns:
        scalar loss
    """
    y_true_smoothed = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    return torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred_logits,
        y_true_smoothed,
    )
