"""
Normalized Wasserstein Distance (NWD) Loss
==========================================
Replacement for IoU-based losses that is more robust for tiny objects.

Paper: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
       Wang et al., 2021
       https://arxiv.org/abs/2110.13389

The key insight: IoU is very sensitive to small positional errors for tiny objects.
A 2-pixel shift on a 10×10 box causes much larger IoU drop than on a 100×100 box.
NWD treats boxes as 2D Gaussian distributions and measures their Wasserstein distance,
which is more forgiving for small objects.
"""

import torch
import torch.nn as nn
import math


def box_to_gaussian(boxes):
    """
    Convert bounding boxes to 2D Gaussian parameters.

    Each box [x1, y1, x2, y2] is modeled as a Gaussian with:
    - Mean: center of the box (cx, cy)
    - Std: proportional to box dimensions (w/2, h/2)

    Args:
        boxes: Tensor of shape [N, 4] in xyxy format

    Returns:
        centers: [N, 2] - Gaussian means (cx, cy)
        stds: [N, 2] - Gaussian stds (w/2, h/2)
    """
    # Get center and dimensions
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    centers = torch.stack([cx, cy], dim=1)
    # Std is proportional to size (using w/2, h/2 as in the paper)
    stds = torch.stack([w / 2, h / 2], dim=1)

    return centers, stds


def wasserstein_distance_2d(centers1, stds1, centers2, stds2):
    """
    Compute 2D Wasserstein distance between two Gaussians.

    For 2D Gaussians with diagonal covariance, the squared Wasserstein distance is:
    W^2 = ||μ1 - μ2||^2 + ||σ1 - σ2||^2

    Where:
    - μ = center (mean)
    - σ = std (sqrt of variance)

    Args:
        centers1, centers2: [N, 2] - Gaussian means
        stds1, stds2: [N, 2] - Gaussian stds

    Returns:
        distances: [N] - Wasserstein distances
    """
    # Squared distance between centers
    center_dist_sq = torch.sum((centers1 - centers2) ** 2, dim=1)

    # Squared distance between stds
    std_dist_sq = torch.sum((stds1 - stds2) ** 2, dim=1)

    # Total squared Wasserstein distance
    w2 = center_dist_sq + std_dist_sq

    # Wasserstein distance (L2)
    return torch.sqrt(w2 + 1e-7)


def nwd_loss(pred_boxes, target_boxes, constant=12.0):
    """
    Compute Normalized Wasserstein Distance loss.

    NWD = exp(-sqrt(W^2) / C)

    Where:
    - W^2 is squared Wasserstein distance
    - C is a normalizing constant (typically related to image size)

    Loss = 1 - NWD (so lower is better, like IoU loss)

    Args:
        pred_boxes: [N, 4] predicted boxes in xyxy format
        target_boxes: [N, 4] target boxes in xyxy format
        constant: Normalizing constant C (default 12.0, tunable)

    Returns:
        loss: [N] NWD loss values (1 - NWD)
    """
    # Convert boxes to Gaussian parameters
    pred_centers, pred_stds = box_to_gaussian(pred_boxes)
    target_centers, target_stds = box_to_gaussian(target_boxes)

    # Compute Wasserstein distance
    w_dist = wasserstein_distance_2d(pred_centers, pred_stds, target_centers, target_stds)

    # Normalize and convert to similarity (0 to 1)
    nwd = torch.exp(-w_dist / constant)

    # Return loss (1 - similarity)
    return 1 - nwd


def nwd_iou_hybrid_loss(pred_boxes, target_boxes, iou_weight=0.5, nwd_weight=0.5, constant=12.0):
    """
    Hybrid loss combining IoU and NWD.

    This can give the best of both worlds:
    - IoU: Good for larger objects, well-established
    - NWD: Better gradients for tiny objects

    Args:
        pred_boxes: [N, 4] predicted boxes in xyxy format
        target_boxes: [N, 4] target boxes in xyxy format
        iou_weight: Weight for IoU loss
        nwd_weight: Weight for NWD loss
        constant: NWD normalizing constant

    Returns:
        loss: [N] combined loss values
    """
    from ultralytics.utils.metrics import bbox_iou

    # Compute IoU
    iou = bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=True)
    iou_loss = 1 - iou

    # Compute NWD
    nwd = nwd_loss(pred_boxes, target_boxes, constant)

    # Combine
    return iou_weight * iou_loss + nwd_weight * nwd


# =============================================================================
# Ultralytics Integration
# =============================================================================
def patch_ultralytics_bbox_loss():
    """
    Monkey-patch Ultralytics to use NWD loss.

    Call this BEFORE creating the YOLO model.

    WARNING: This modifies Ultralytics internals and may break with version updates.

    Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py
    """
    try:
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import bbox2dist
        import torch.nn.functional as F

        # Store original forward method
        original_forward = BboxLoss.forward

        def nwd_forward(
            self,
            pred_dist: torch.Tensor,
            pred_bboxes: torch.Tensor,
            anchor_points: torch.Tensor,
            target_bboxes: torch.Tensor,
            target_scores: torch.Tensor,
            target_scores_sum: torch.Tensor,
            fg_mask: torch.Tensor,
            imgsz: torch.Tensor,
            stride: torch.Tensor,
        ):
            """Modified forward that uses NWD instead of CIoU for box loss."""
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            # NWD loss instead of IoU loss
            pred_fg = pred_bboxes[fg_mask]
            target_fg = target_bboxes[fg_mask]

            if pred_fg.numel() == 0 or target_fg.numel() == 0:
                loss_iou = torch.tensor(0.0, device=pred_dist.device)
            else:
                nwd = nwd_loss(pred_fg, target_fg, constant=50.0)  # Tuned for tiny objects (16-32px)
                loss_iou = (nwd * weight).sum() / target_scores_sum

            # DFL loss (unchanged from original)
            if self.dfl_loss:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
                loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                target_ltrb = bbox2dist(anchor_points, target_bboxes)
                # normalize ltrb by image size
                target_ltrb = target_ltrb * stride
                target_ltrb[..., 0::2] /= imgsz[1]
                target_ltrb[..., 1::2] /= imgsz[0]
                pred_dist = pred_dist * stride
                pred_dist[..., 0::2] /= imgsz[1]
                pred_dist[..., 1::2] /= imgsz[0]
                loss_dfl = (
                    F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
                )
                loss_dfl = loss_dfl.sum() / target_scores_sum

            return loss_iou, loss_dfl

        # Apply patch
        BboxLoss.forward = nwd_forward
        print("✓ Patched Ultralytics BboxLoss to use NWD")
        return True

    except AttributeError as e:
        print(f"✗ Failed to patch: BboxLoss structure changed in this Ultralytics version")
        print(f"  Error: {e}")
        return False
    except ImportError as e:
        print(f"✗ Failed to import Ultralytics loss module: {e}")
        return False
    except Exception as e:
        print(f"✗ Failed to patch Ultralytics: {e}")
        return False


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    # Test NWD computation
    print("Testing NWD Loss\n" + "=" * 40)

    # Create sample boxes
    pred = torch.tensor([
        [10, 10, 20, 20],   # 10x10 box
        [100, 100, 200, 200],  # 100x100 box
    ], dtype=torch.float32)

    # Shift both boxes by 2 pixels
    target = torch.tensor([
        [12, 12, 22, 22],   # 10x10 box shifted by 2
        [102, 102, 202, 202],  # 100x100 box shifted by 2
    ], dtype=torch.float32)

    # Compute NWD loss
    nwd = nwd_loss(pred, target)
    print(f"NWD loss for 10x10 box (2px shift):  {nwd[0]:.4f}")
    print(f"NWD loss for 100x100 box (2px shift): {nwd[1]:.4f}")

    # Compare with IoU
    from ultralytics.utils.metrics import bbox_iou
    iou = bbox_iou(pred, target, xywh=False)
    print(f"\nIoU for 10x10 box (2px shift):  {iou[0].item():.4f}  (loss: {1-iou[0].item():.4f})")
    print(f"IoU for 100x100 box (2px shift): {iou[1].item():.4f}  (loss: {1-iou[1].item():.4f})")

    print("\n" + "=" * 40)
    print("Notice: IoU drops much more for the small box!")
    print("NWD is more consistent across object sizes.")
