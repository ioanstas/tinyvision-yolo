"""
Normalized Wasserstein Distance (NWD) Loss — C=12.8 (Official Default)
======================================================================
Same as nwd_loss.py but using the official paper's default constant C=12.8.

Paper: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
       Wang et al., 2021
       https://arxiv.org/abs/2110.13389

Official code: https://github.com/jwwangchn/NWD
"""

import torch
import torch.nn.functional as F


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
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    centers = torch.stack([cx, cy], dim=1)
    stds = torch.stack([w / 2, h / 2], dim=1)

    return centers, stds


def wasserstein_distance_2d(centers1, stds1, centers2, stds2):
    """
    Compute 2D Wasserstein distance between two Gaussians.

    For 2D Gaussians with diagonal covariance, the squared Wasserstein distance is:
    W^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2

    Args:
        centers1, centers2: [N, 2] - Gaussian means
        stds1, stds2: [N, 2] - Gaussian stds

    Returns:
        distances: [N] - Wasserstein distances
    """
    center_dist_sq = torch.sum((centers1 - centers2) ** 2, dim=1)
    std_dist_sq = torch.sum((stds1 - stds2) ** 2, dim=1)
    w2 = center_dist_sq + std_dist_sq
    return torch.sqrt(w2 + 1e-7)


def nwd_loss(pred_boxes, target_boxes, constant=12.8):
    """
    Compute Normalized Wasserstein Distance loss.

    NWD = exp(-sqrt(W^2) / C)
    Loss = 1 - NWD

    Args:
        pred_boxes: [N, 4] predicted boxes in xyxy format
        target_boxes: [N, 4] target boxes in xyxy format
        constant: Normalizing constant C (default 12.8, official paper value)

    Returns:
        loss: [N] NWD loss values (1 - NWD)
    """
    pred_centers, pred_stds = box_to_gaussian(pred_boxes)
    target_centers, target_stds = box_to_gaussian(target_boxes)

    w_dist = wasserstein_distance_2d(pred_centers, pred_stds, target_centers, target_stds)

    nwd = torch.exp(-w_dist / constant)

    return 1 - nwd


# =============================================================================
# Ultralytics Integration
# =============================================================================
def patch_ultralytics_bbox_loss():
    """
    Monkey-patch Ultralytics to use NWD loss with C=12.8.

    Call this BEFORE creating the YOLO model.
    """
    try:
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import bbox2dist

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
            """Modified forward that uses NWD (C=12.8) instead of CIoU for box loss."""
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            # NWD loss instead of IoU loss
            pred_fg = pred_bboxes[fg_mask]
            target_fg = target_bboxes[fg_mask]

            if pred_fg.numel() == 0 or target_fg.numel() == 0:
                loss_iou = torch.tensor(0.0, device=pred_dist.device)
            else:
                nwd = nwd_loss(pred_fg, target_fg, constant=12.8)
                loss_iou = (nwd * weight).sum() / target_scores_sum

            # DFL loss (unchanged from original)
            if self.dfl_loss:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
                loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                target_ltrb = bbox2dist(anchor_points, target_bboxes)
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

        BboxLoss.forward = nwd_forward
        print("Patched Ultralytics BboxLoss to use NWD (C=12.8)")
        return True

    except AttributeError as e:
        print(f"Failed to patch: BboxLoss structure changed in this Ultralytics version")
        print(f"  Error: {e}")
        return False
    except ImportError as e:
        print(f"Failed to import Ultralytics loss module: {e}")
        return False
    except Exception as e:
        print(f"Failed to patch Ultralytics: {e}")
        return False


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    print("Testing NWD Loss (C=12.8)\n" + "=" * 40)

    pred = torch.tensor([
        [10, 10, 20, 20],
        [100, 100, 200, 200],
    ], dtype=torch.float32)

    target = torch.tensor([
        [12, 12, 22, 22],
        [102, 102, 202, 202],
    ], dtype=torch.float32)

    nwd = nwd_loss(pred, target, constant=12.8)
    print(f"NWD loss for 10x10 box (2px shift):  {nwd[0].item():.4f}")
    print(f"NWD loss for 100x100 box (2px shift): {nwd[1].item():.4f}")

    from ultralytics.utils.metrics import bbox_iou
    iou = bbox_iou(pred, target, xywh=False)
    print(f"\nIoU for 10x10 box (2px shift):  {iou[0].item():.4f}  (loss: {1-iou[0].item():.4f})")
    print(f"IoU for 100x100 box (2px shift): {iou[1].item():.4f}  (loss: {1-iou[1].item():.4f})")

    print("\n" + "=" * 40)
    print("C=12.8 gives steeper gradients than C=50.")
