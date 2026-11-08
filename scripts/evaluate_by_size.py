"""
Evaluate Model by Object Size
=============================
Computes AP metrics broken down by object size (small/medium/large).
Use this script to measure improvement on tiny objects across experiments.

Usage:
    python scripts/evaluate_by_size.py --model experiments/naive_baseline/weights/best.pt
    python scripts/evaluate_by_size.py --model experiments/p2_baseline/weights/best.pt
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model" / "dataset.yaml"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluations"

# Size thresholds (in pixels, area = width * height)
# COCO standard: small < 32², medium < 96², large >= 96²
# Custom for tiny objects: add extra-small category
SIZE_THRESHOLDS = {
    "tiny": 16 ** 2,      # < 16x16 = 256 px²
    "small": 32 ** 2,     # < 32x32 = 1024 px²
    "medium": 96 ** 2,    # < 96x96 = 9216 px²
    # large: >= 96x96
}

CLASSES = {
    0: "ship",
    1: "car",
    2: "cyclist",
    3: "pedestrian",
    4: "bus",
    5: "drone",
    6: "plane"
}


# =============================================================================
# Helper Functions
# =============================================================================
def load_yolo_labels(labels_dir, img_width=640, img_height=512):
    """
    Load YOLO format labels and convert to absolute pixel coordinates.
    Returns: dict[image_name] = list of (class_id, x1, y1, x2, y2, area)
    """
    ground_truths = {}

    for label_file in Path(labels_dir).glob("*.txt"):
        img_name = label_file.stem
        boxes = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    w = float(parts[3]) * img_width
                    h = float(parts[4]) * img_height

                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                    area = w * h

                    boxes.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'area': area
                    })

        ground_truths[img_name] = boxes

    return ground_truths


def get_size_category(area):
    """Categorize object by area."""
    if area < SIZE_THRESHOLDS["tiny"]:
        return "tiny"
    elif area < SIZE_THRESHOLDS["small"]:
        return "small"
    elif area < SIZE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "large"


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def compute_ap(precisions, recalls):
    """Compute Average Precision using 101-point interpolation."""
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # Add boundary points
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max() / 101

    return ap


def evaluate_size_category(predictions, ground_truths, size_category, iou_threshold=0.5):
    """
    Compute precision, recall, and AP for a specific size category.
    """
    all_detections = []  # (confidence, is_tp, class_id)
    total_gt = 0

    for img_name, gt_boxes in ground_truths.items():
        # Filter GT by size
        gt_filtered = [gt for gt in gt_boxes if get_size_category(gt['area']) == size_category]
        total_gt += len(gt_filtered)

        # Get predictions for this image
        pred_boxes = predictions.get(img_name, [])

        # Track which GT boxes have been matched
        gt_matched = [False] * len(gt_filtered)

        # Sort predictions by confidence
        pred_boxes = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)

        for pred in pred_boxes:
            pred_area = (pred['bbox'][2] - pred['bbox'][0]) * (pred['bbox'][3] - pred['bbox'][1])

            # Only consider predictions in this size category
            if get_size_category(pred_area) != size_category:
                continue

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_filtered):
                if gt_matched[gt_idx]:
                    continue
                if gt['class_id'] != pred['class_id']:
                    continue

                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                all_detections.append((pred['confidence'], True, pred['class_id']))
                gt_matched[best_gt_idx] = True
            else:
                all_detections.append((pred['confidence'], False, pred['class_id']))

    if total_gt == 0:
        return {'ap': 0, 'precision': 0, 'recall': 0, 'total_gt': 0, 'total_pred': len(all_detections)}

    # Sort by confidence
    all_detections = sorted(all_detections, key=lambda x: x[0], reverse=True)

    # Compute precision-recall curve
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for conf, is_tp, _ in all_detections:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt

        precisions.append(precision)
        recalls.append(recall)

    ap = compute_ap(precisions, recalls) if precisions else 0
    final_precision = precisions[-1] if precisions else 0
    final_recall = recalls[-1] if recalls else 0

    return {
        'ap': ap,
        'precision': final_precision,
        'recall': final_recall,
        'total_gt': total_gt,
        'total_pred': len(all_detections)
    }


# =============================================================================
# Main Evaluation
# =============================================================================
def run_evaluation(model_path, split='val', conf_threshold=0.001):
    """
    Run full evaluation with size breakdown.
    """
    print("=" * 60)
    print("EVALUATION BY OBJECT SIZE")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Split: {split}")
    print(f"Confidence threshold: {conf_threshold}")

    # Load model
    print("\nLoading model...")
    model = YOLO(model_path)

    # Determine paths
    data_dir = PROJECT_ROOT / "data" / split
    images_dir = data_dir / "images" / "rgb"
    labels_dir = data_dir / "labels" / "rgb"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return None

    # Load ground truths
    print("Loading ground truth labels...")
    ground_truths = load_yolo_labels(labels_dir)
    print(f"  Loaded {len(ground_truths)} images with labels")

    # Count GT by size
    gt_by_size = defaultdict(int)
    for img_name, boxes in ground_truths.items():
        for box in boxes:
            gt_by_size[get_size_category(box['area'])] += 1

    print("\n  Ground truth distribution:")
    for size in ["tiny", "small", "medium", "large"]:
        print(f"    {size:<8}: {gt_by_size[size]:>6} objects")

    # Run inference
    print("\nRunning inference...")
    results = model.predict(
        source=str(images_dir),
        conf=conf_threshold,
        iou=0.5,
        verbose=False,
        stream=True
    )

    # Collect predictions
    predictions = {}
    total_preds = 0

    for result in results:
        img_name = Path(result.path).stem
        boxes = []

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append({
                    'class_id': int(box.cls[0].cpu().numpy()),
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'bbox': xyxy.tolist()
                })
                total_preds += 1

        predictions[img_name] = boxes

    print(f"  Total predictions: {total_preds}")

    # Evaluate by size category
    print("\n" + "=" * 60)
    print("RESULTS BY SIZE CATEGORY")
    print("=" * 60)

    results_table = {}

    for size in ["tiny", "small", "medium", "large"]:
        metrics = evaluate_size_category(predictions, ground_truths, size)
        results_table[size] = metrics

        print(f"\n{size.upper()} (area < {SIZE_THRESHOLDS.get(size, 'inf')} px²):")
        print(f"  Ground truths: {metrics['total_gt']:>6}")
        print(f"  AP@50:         {metrics['ap']:>6.4f}")
        print(f"  Precision:     {metrics['precision']:>6.4f}")
        print(f"  Recall:        {metrics['recall']:>6.4f}")

    # Compute overall AP (weighted by GT count)
    total_gt = sum(r['total_gt'] for r in results_table.values())
    weighted_ap = sum(r['ap'] * r['total_gt'] for r in results_table.values()) / total_gt if total_gt > 0 else 0

    print("\n" + "-" * 40)
    print(f"WEIGHTED mAP@50: {weighted_ap:.4f}")
    print("-" * 40)

    # Also run standard YOLO validation for comparison
    print("\n" + "=" * 60)
    print("STANDARD YOLO METRICS (for comparison)")
    print("=" * 60)

    val_results = model.val(data=str(CONFIG_PATH), split=split, verbose=False)
    print(f"\n  mAP@50:     {val_results.box.map50:.4f}")
    print(f"  mAP@50-95:  {val_results.box.map:.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).parent.parent.name
    output_file = RESULTS_DIR / f"{model_name}_{split}_size_metrics.json"

    save_data = {
        'model': str(model_path),
        'split': split,
        'size_metrics': {k: {**v, 'ap': float(v['ap'])} for k, v in results_table.items()},
        'weighted_map50': float(weighted_ap),
        'standard_map50': float(val_results.box.map50),
        'standard_map50_95': float(val_results.box.map),
        'thresholds': SIZE_THRESHOLDS
    }

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results_table


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate model by object size")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")

    args = parser.parse_args()

    run_evaluation(args.model, args.split, args.conf)


if __name__ == "__main__":
    main()
