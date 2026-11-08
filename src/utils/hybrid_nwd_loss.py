"""
Hybrid NWD + CIoU Loss
======================
Combines CIoU (good for medium/large objects) with NWD (better for tiny objects).

CIoU handles overall localization while NWD provides fairer gradients for tiny objects.

Paper: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
       Wang et al., 2021 - https://arxiv.org/abs/2110.13389
"""

import torch
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


# =============================================================================
# NWD Core Functions
# =============================================================================
def box_to_gaussian(boxes):
    """
    Convert bounding boxes to 2D Gaussian parameters.
    Boxes in xyxy format -> Gaussian (center, std).
    """
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    centers = torch.stack([cx, cy], dim=1)
    stds = torch.stack([w / 2, h / 2], dim=1)

    return centers, stds


def wasserstein_distance_2d(centers1, stds1, centers2, stds2):
    """Compute 2D Wasserstein distance between two Gaussians."""
    center_dist_sq = torch.sum((centers1 - centers2) ** 2, dim=1)
    std_dist_sq = torch.sum((stds1 - stds2) ** 2, dim=1)
    return torch.sqrt(center_dist_sq + std_dist_sq + 1e-7)


def nwd_similarity(pred_boxes, target_boxes, constant=32.0):
    """
    Compute NWD similarity (0 to 1, higher is better, like IoU).

    Returns:
        similarity: [N] values in range (0, 1)
    """
    pred_centers, pred_stds = box_to_gaussian(pred_boxes)
    target_centers, target_stds = box_to_gaussian(target_boxes)

    w_dist = wasserstein_distance_2d(pred_centers, pred_stds, target_centers, target_stds)

    # NWD similarity (like IoU: 1 = perfect, 0 = bad)
    return torch.exp(-w_dist / constant)


# =============================================================================
# Hybrid Loss: CIoU + NWD
# =============================================================================
def hybrid_ciou_nwd_loss(pred_boxes, target_boxes, ciou_weight=0.5, nwd_weight=0.5, constant=32.0):
    """
    Compute hybrid loss combining CIoU and NWD.

    Both CIoU and NWD produce similarity values in [0, 1].
    We blend them and return (1 - blended_similarity) as the loss.

    Args:
        pred_boxes: [N, 4] in xyxy format
        target_boxes: [N, 4] in xyxy format
        ciou_weight: Weight for CIoU component (default 0.5)
        nwd_weight: Weight for NWD component (default 0.5)
        constant: NWD normalizing constant (default 32.0)

    Returns:
        loss: [N] loss values in [0, 1]
    """
    # CIoU similarity (0 to 1)
    ciou = bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=True).squeeze(-1)

    # NWD similarity (0 to 1)
    nwd = nwd_similarity(pred_boxes, target_boxes, constant=constant)

    # Blend similarities
    blended = ciou_weight * ciou + nwd_weight * nwd

    # Return loss (1 - similarity)
    return 1.0 - blended


# =============================================================================
# Ultralytics Patch
# =============================================================================
def patch_ultralytics_hybrid_loss(ciou_weight=0.5, nwd_weight=0.5, constant=32.0):
    """
    Monkey-patch Ultralytics BboxLoss to use hybrid CIoU + NWD loss.

    Call BEFORE creating the YOLO model.

    Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py
    """
    try:
        from ultralytics.utils.loss import BboxLoss

        original_forward = BboxLoss.forward

        def hybrid_forward(
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
            """Forward with hybrid CIoU + NWD loss."""
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            # Hybrid loss
            pred_fg = pred_bboxes[fg_mask]
            target_fg = target_bboxes[fg_mask]

            if pred_fg.numel() == 0 or target_fg.numel() == 0:
                loss_iou = torch.tensor(0.0, device=pred_dist.device)
            else:
                loss = hybrid_ciou_nwd_loss(
                    pred_fg, target_fg,
                    ciou_weight=ciou_weight,
                    nwd_weight=nwd_weight,
                    constant=constant
                )
                loss_iou = (loss * weight).sum() / target_scores_sum

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

        BboxLoss.forward = hybrid_forward
        print(f"Patched BboxLoss with Hybrid CIoU+NWD (ciou={ciou_weight}, nwd={nwd_weight}, C={constant})")
        return True

    except Exception as e:
        print(f"Failed to patch: {e}")
        return False


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    print("Testing Hybrid CIoU + NWD Loss\n" + "=" * 50)

    # Same-size boxes shifted by 2 pixels
    pred = torch.tensor([
        [10, 10, 20, 20],       # 10x10 (tiny)
        [100, 100, 200, 200],   # 100x100 (large)
    ], dtype=torch.float32)

    target = torch.tensor([
        [12, 12, 22, 22],       # shifted by 2
        [102, 102, 202, 202],   # shifted by 2
    ], dtype=torch.float32)

    # Pure CIoU
    ciou = bbox_iou(pred, target, xywh=False, CIoU=True)
    print(f"CIoU loss  10x10:  {1-ciou[0].item():.4f}")
    print(f"CIoU loss 100x100: {1-ciou[1].item():.4f}")

    # Pure NWD
    nwd = nwd_similarity(pred, target, constant=32.0)
    print(f"\nNWD loss  10x10:  {1-nwd[0].item():.4f}")
    print(f"NWD loss 100x100: {1-nwd[1].item():.4f}")

    # Hybrid
    hybrid = hybrid_ciou_nwd_loss(pred, target, ciou_weight=0.5, nwd_weight=0.5, constant=32.0)
    print(f"\nHybrid loss  10x10:  {hybrid[0].item():.4f}")
    print(f"Hybrid loss 100x100: {hybrid[1].item():.4f}")

    print("\n" + "=" * 50)
    print("Hybrid should be between CIoU and NWD values.")
    print("Tiny box penalty should be lower than pure CIoU.")
