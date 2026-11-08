"""
SAHI Evaluation Script
======================
Evaluate models using Slicing Aided Hyper Inference (SAHI) for better
small object detection.

SAHI slices images into overlapping patches, runs detection on each patch,
and merges results. This preserves small object resolution.

Install: pip install sahi

Usage:
    python scripts/evaluate_sahi.py --model experiments/naive_baseline/weights/best.pt
    python scripts/evaluate_sahi.py --model experiments/p2_baseline/weights/best.pt --slice_size 512
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("WARNING: SAHI not installed. Run: pip install sahi")

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model" / "dataset.yaml"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluations"

SIZE_THRESHOLDS = {
    "tiny": 16 ** 2,
    "small": 32 ** 2,
    "medium": 96 ** 2,
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
# Helper Functions (same as evaluate_by_size.py)
# =============================================================================
def load_yolo_labels(labels_dir, img_width=640, img_height=512):
    """Load YOLO format labels and convert to absolute pixel coordinates."""
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
    if not precisions or not recalls:
        return 0.0

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max() / 101

    return ap


def evaluate_size_category(predictions, ground_truths, size_category, iou_threshold=0.5):
    """Compute precision, recall, and AP for a specific size category."""
    all_detections = []
    total_gt = 0

    for img_name, gt_boxes in ground_truths.items():
        gt_filtered = [gt for gt in gt_boxes if get_size_category(gt['area']) == size_category]
        total_gt += len(gt_filtered)

        pred_boxes = predictions.get(img_name, [])
        gt_matched = [False] * len(gt_filtered)
        pred_boxes = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)

        for pred in pred_boxes:
            pred_area = (pred['bbox'][2] - pred['bbox'][0]) * (pred['bbox'][3] - pred['bbox'][1])

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

    all_detections = sorted(all_detections, key=lambda x: x[0], reverse=True)

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

    ap = compute_ap(precisions, recalls)
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
# SAHI Inference
# =============================================================================
def run_sahi_inference(model_path, images_dir, slice_size=640, overlap_ratio=0.2, conf_threshold=0.25):
    """
    Run SAHI sliced inference on all images.

    Args:
        model_path: Path to YOLO weights
        images_dir: Directory with images
        slice_size: Size of each slice (default 640)
        overlap_ratio: Overlap between slices (default 0.2 = 20%)
        conf_threshold: Confidence threshold

    Returns:
        dict[image_name] = list of predictions
    """
    if not SAHI_AVAILABLE:
        raise ImportError("SAHI not installed. Run: pip install sahi")

    print(f"\nInitializing SAHI model...")
    print(f"  Slice size: {slice_size}x{slice_size}")
    print(f"  Overlap ratio: {overlap_ratio}")
    print(f"  Confidence threshold: {conf_threshold}")

    # Load model with SAHI wrapper
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(model_path),
        confidence_threshold=conf_threshold,
        device="cuda:0"  # Change to "cpu" if no GPU
    )

    # Get all images
    image_files = list(Path(images_dir).glob("*.jpg"))
    print(f"  Found {len(image_files)} images")

    predictions = {}
    total_preds = 0

    print("\nRunning SAHI inference...")
    for img_path in tqdm(image_files, desc="SAHI Inference"):
        # Run sliced prediction
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5,
            verbose=0
        )

        # Extract predictions
        img_name = img_path.stem
        boxes = []

        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()  # [x1, y1, x2, y2]
            boxes.append({
                'class_id': pred.category.id,
                'confidence': pred.score.value,
                'bbox': bbox
            })
            total_preds += 1

        predictions[img_name] = boxes

    print(f"  Total predictions: {total_preds}")
    return predictions


# =============================================================================
# Standard Inference (for comparison)
# =============================================================================
def run_standard_inference(model_path, images_dir, conf_threshold=0.25):
    """Run standard YOLO inference without SAHI."""
    from ultralytics import YOLO

    print(f"\nRunning standard inference...")
    model = YOLO(model_path)

    image_files = list(Path(images_dir).glob("*.jpg"))
    predictions = {}
    total_preds = 0

    results = model.predict(
        source=str(images_dir),
        conf=conf_threshold,
        verbose=False,
        stream=True
    )

    for result in tqdm(results, total=len(image_files), desc="Standard Inference"):
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
    return predictions


# =============================================================================
# Main Evaluation
# =============================================================================
def run_evaluation(model_path, split='val', slice_size=640, overlap_ratio=0.2, conf_threshold=0.25, compare=True):
    """Run SAHI evaluation with optional comparison to standard inference."""

    print("=" * 60)
    print("SAHI EVALUATION")
    print("=" * 60)
    print(f"\nModel: {model_path}")
    print(f"Split: {split}")
    print(f"Slice size: {slice_size}")
    print(f"Overlap ratio: {overlap_ratio}")

    # Paths
    data_dir = PROJECT_ROOT / "data" / split
    images_dir = data_dir / "images" / "rgb"
    labels_dir = data_dir / "labels" / "rgb"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return None

    # Load ground truths
    print("\nLoading ground truth labels...")
    ground_truths = load_yolo_labels(labels_dir)

    # Count GT by size
    gt_by_size = defaultdict(int)
    for img_name, boxes in ground_truths.items():
        for box in boxes:
            gt_by_size[get_size_category(box['area'])] += 1

    print(f"  Ground truth distribution:")
    for size in ["tiny", "small", "medium", "large"]:
        print(f"    {size:<8}: {gt_by_size[size]:>6} objects")

    # Run SAHI inference
    sahi_predictions = run_sahi_inference(
        model_path, images_dir, slice_size, overlap_ratio, conf_threshold
    )

    # Optionally run standard inference for comparison
    standard_predictions = None
    if compare:
        standard_predictions = run_standard_inference(model_path, images_dir, conf_threshold)

    # Evaluate SAHI results
    print("\n" + "=" * 60)
    print("SAHI RESULTS BY SIZE CATEGORY")
    print("=" * 60)

    sahi_results = {}
    for size in ["tiny", "small", "medium", "large"]:
        metrics = evaluate_size_category(sahi_predictions, ground_truths, size)
        sahi_results[size] = metrics

        print(f"\n{size.upper()}:")
        print(f"  AP@50: {metrics['ap']:.4f}  |  Precision: {metrics['precision']:.4f}  |  Recall: {metrics['recall']:.4f}")

    # Evaluate standard results for comparison
    if compare and standard_predictions:
        print("\n" + "=" * 60)
        print("STANDARD RESULTS (for comparison)")
        print("=" * 60)

        standard_results = {}
        for size in ["tiny", "small", "medium", "large"]:
            metrics = evaluate_size_category(standard_predictions, ground_truths, size)
            standard_results[size] = metrics

            print(f"\n{size.upper()}:")
            print(f"  AP@50: {metrics['ap']:.4f}  |  Precision: {metrics['precision']:.4f}  |  Recall: {metrics['recall']:.4f}")

        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON: SAHI vs STANDARD")
        print("=" * 60)
        print(f"\n{'Size':<10} {'Standard AP':<15} {'SAHI AP':<15} {'Improvement':<15}")
        print("-" * 55)

        for size in ["tiny", "small", "medium", "large"]:
            std_ap = standard_results[size]['ap']
            sahi_ap = sahi_results[size]['ap']
            improvement = sahi_ap - std_ap
            pct = (improvement / std_ap * 100) if std_ap > 0 else 0

            print(f"{size:<10} {std_ap:<15.4f} {sahi_ap:<15.4f} {improvement:+.4f} ({pct:+.1f}%)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).parent.parent.name
    output_file = RESULTS_DIR / f"{model_name}_{split}_sahi_metrics.json"

    save_data = {
        'model': str(model_path),
        'split': split,
        'slice_size': slice_size,
        'overlap_ratio': overlap_ratio,
        'sahi_metrics': {k: {**v, 'ap': float(v['ap'])} for k, v in sahi_results.items()},
    }

    if compare and standard_predictions:
        save_data['standard_metrics'] = {k: {**v, 'ap': float(v['ap'])} for k, v in standard_results.items()}

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return sahi_results


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate model with SAHI")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--slice_size", type=int, default=640, help="Slice size for SAHI")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio (0-1)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no_compare", action="store_true", help="Skip standard inference comparison")

    args = parser.parse_args()

    if not SAHI_AVAILABLE:
        print("ERROR: SAHI not installed.")
        print("Install with: pip install sahi")
        return

    run_evaluation(
        args.model,
        args.split,
        args.slice_size,
        args.overlap,
        args.conf,
        compare=not args.no_compare
    )


if __name__ == "__main__":
    main()
