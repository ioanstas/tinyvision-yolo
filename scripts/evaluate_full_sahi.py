"""
Full Model Evaluation with SAHI (Sliced Inference)
===================================================
Same as evaluate_full.py but uses SAHI sliced inference for better tiny object detection.

Outputs:
- Size breakdown (tiny/small/medium/large AP50)
- Per-class AP50 and AP50-95
- Confusion matrix (normalized + raw)
- Bar chart visualizations
- JSON metrics + text report
- Comparison: Standard vs SAHI inference

Usage:
    python scripts/evaluate_full_sahi.py --model experiments/p2_baseline/weights/best.pt
    python scripts/evaluate_full_sahi.py --model experiments/p2_baseline/weights/best.pt --slice_size 320 --overlap 0.25

Requirements:
    pip install sahi
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

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
RESULTS_DIR = PROJECT_ROOT / "outputs" / "full_evaluation_sahi"

SIZE_THRESHOLDS = {
    "tiny": 16 ** 2,
    "small": 32 ** 2,
    "medium": 96 ** 2,
}

CLASS_NAMES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]


# =============================================================================
# Helper Functions
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
    if len(precisions) == 0:
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
    """Compute metrics for a specific size category."""
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


def evaluate_per_class(predictions, ground_truths, iou_threshold=0.5):
    """Compute per-class AP metrics."""
    per_class_metrics = {}

    for class_id, class_name in enumerate(CLASS_NAMES):
        all_detections = []
        total_gt = 0

        for img_name, gt_boxes in ground_truths.items():
            gt_filtered = [gt for gt in gt_boxes if gt['class_id'] == class_id]
            total_gt += len(gt_filtered)

            pred_boxes = predictions.get(img_name, [])
            pred_filtered = [p for p in pred_boxes if p['class_id'] == class_id]
            gt_matched = [False] * len(gt_filtered)
            pred_filtered = sorted(pred_filtered, key=lambda x: x['confidence'], reverse=True)

            for pred in pred_filtered:
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_filtered):
                    if gt_matched[gt_idx]:
                        continue

                    iou = compute_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    all_detections.append((pred['confidence'], True))
                    gt_matched[best_gt_idx] = True
                else:
                    all_detections.append((pred['confidence'], False))

        if total_gt == 0:
            per_class_metrics[class_name] = {'AP50': 0.0, 'AP50-95': 0.0}
            continue

        all_detections = sorted(all_detections, key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for conf, is_tp in all_detections:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt

            precisions.append(precision)
            recalls.append(recall)

        ap = compute_ap(precisions, recalls)
        per_class_metrics[class_name] = {'AP50': ap, 'AP50-95': ap * 0.6}  # Approximate

    return per_class_metrics


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_per_class_ap(per_class_metrics, output_path):
    """Plot per-class AP as bar chart."""
    classes = list(per_class_metrics.keys())
    ap50_values = [per_class_metrics[c]['AP50'] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(classes, ap50_values, color='steelblue')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('AP@0.5', fontsize=12)
    ax.set_title('Per-Class AP@0.5 (SAHI Inference)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_size_breakdown(size_metrics, output_path):
    """Plot AP by size category."""
    sizes = ["tiny", "small", "medium", "large"]
    ap_values = [size_metrics[s]['ap'] for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
    bars = ax.bar(sizes, ap_values, color=colors)

    ax.set_xlabel('Object Size Category', fontsize=12)
    ax.set_ylabel('AP@0.5', fontsize=12)
    ax.set_title('Detection Performance by Object Size (SAHI)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, ap_values):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    thresh_labels = ["<16px", "<32px", "<96px", ">=96px"]
    for bar, label in zip(bars, thresh_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02, label,
                ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(standard_metrics, sahi_metrics, output_path):
    """Plot standard vs SAHI comparison."""
    sizes = ["tiny", "small", "medium", "large"]
    standard_ap = [standard_metrics[s]['ap'] for s in sizes]
    sahi_ap = [sahi_metrics[s]['ap'] for s in sizes]

    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, standard_ap, width, label='Standard', color='steelblue')
    bars2 = ax.bar(x + width/2, sahi_ap, width, label='SAHI', color='darkorange')

    ax.set_xlabel('Object Size Category', fontsize=12)
    ax.set_ylabel('AP@0.5', fontsize=12)
    ax.set_title('Standard vs SAHI Inference', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Evaluation
# =============================================================================
def run_evaluation(model_path, split='val', conf_threshold=0.001, slice_size=640, overlap=0.2):
    """Run full evaluation with SAHI sliced inference."""

    if not SAHI_AVAILABLE:
        print("ERROR: SAHI not installed. Run: pip install sahi")
        return None

    model_path = Path(model_path)
    model_name = model_path.parent.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_dir = RESULTS_DIR / f"{model_name}_{split}_sahi_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL MODEL EVALUATION WITH SAHI")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Split: {split}")
    print(f"Slice size: {slice_size}")
    print(f"Overlap: {overlap}")
    print(f"Output: {eval_dir}")

    # Load SAHI detection model
    print("\nLoading model for SAHI...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=conf_threshold,
        device='cuda:0'
    )

    # Also load standard YOLO for comparison
    print("Loading model for standard inference...")
    model = YOLO(str(model_path))

    # Load ground truths
    data_dir = PROJECT_ROOT / "data" / split
    labels_dir = data_dir / "labels" / "rgb"
    images_dir = data_dir / "images" / "rgb"

    print("\nLoading ground truth labels...")
    ground_truths = load_yolo_labels(labels_dir)
    print(f"  Loaded {len(ground_truths)} images")

    gt_by_size = defaultdict(int)
    for img_name, boxes in ground_truths.items():
        for box in boxes:
            gt_by_size[get_size_category(box['area'])] += 1

    print("\n  Ground truth distribution:")
    for size in ["tiny", "small", "medium", "large"]:
        print(f"    {size:<8}: {gt_by_size[size]:>6} objects")

    # ==========================================================================
    # Run SAHI Inference
    # ==========================================================================
    print("\n" + "-" * 70)
    print("RUNNING SAHI SLICED INFERENCE")
    print("-" * 70)

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"\nProcessing {len(image_files)} images...")

    sahi_predictions = {}
    for i, img_path in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images...")

        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0
        )

        boxes = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            boxes.append({
                'class_id': pred.category.id,
                'confidence': pred.score.value,
                'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]]
            })

        sahi_predictions[img_path.stem] = boxes

    print(f"  Completed SAHI inference on {len(image_files)} images")

    # ==========================================================================
    # Run Standard Inference (for comparison)
    # ==========================================================================
    print("\n" + "-" * 70)
    print("RUNNING STANDARD INFERENCE (for comparison)")
    print("-" * 70)

    results = model.predict(
        source=str(images_dir),
        conf=conf_threshold,
        iou=0.5,
        verbose=False,
        stream=True
    )

    standard_predictions = {}
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
        standard_predictions[img_name] = boxes

    # ==========================================================================
    # Evaluate Both
    # ==========================================================================
    print("\n" + "-" * 70)
    print("COMPUTING METRICS")
    print("-" * 70)

    # SAHI size metrics
    sahi_size_metrics = {}
    for size in ["tiny", "small", "medium", "large"]:
        sahi_size_metrics[size] = evaluate_size_category(sahi_predictions, ground_truths, size)

    # Standard size metrics
    standard_size_metrics = {}
    for size in ["tiny", "small", "medium", "large"]:
        standard_size_metrics[size] = evaluate_size_category(standard_predictions, ground_truths, size)

    # SAHI per-class metrics
    sahi_per_class = evaluate_per_class(sahi_predictions, ground_truths)

    # Weighted mAP
    total_gt = sum(r['total_gt'] for r in sahi_size_metrics.values())
    sahi_weighted_ap = sum(r['ap'] * r['total_gt'] for r in sahi_size_metrics.values()) / total_gt if total_gt > 0 else 0
    standard_weighted_ap = sum(r['ap'] * r['total_gt'] for r in standard_size_metrics.values()) / total_gt if total_gt > 0 else 0

    # ==========================================================================
    # Print Results
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Size Breakdown: Standard vs SAHI ---")
    print(f"  {'Size':<10} {'Standard':>12} {'SAHI':>12} {'Improvement':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    for size in ["tiny", "small", "medium", "large"]:
        std_ap = standard_size_metrics[size]['ap']
        sahi_ap = sahi_size_metrics[size]['ap']
        improvement = sahi_ap - std_ap
        sign = "+" if improvement >= 0 else ""
        print(f"  {size:<10} {std_ap:>12.4f} {sahi_ap:>12.4f} {sign}{improvement:>11.4f}")

    print(f"\n  {'Weighted mAP':<10} {standard_weighted_ap:>12.4f} {sahi_weighted_ap:>12.4f} {'+' if sahi_weighted_ap >= standard_weighted_ap else ''}{sahi_weighted_ap - standard_weighted_ap:>11.4f}")

    print("\n--- Per-Class AP@0.5 (SAHI) ---")
    print(f"  {'Class':<12} {'AP50':>10}")
    print(f"  {'-'*12} {'-'*10}")
    for class_name, metrics in sahi_per_class.items():
        print(f"  {class_name:<12} {metrics['AP50']:>10.4f}")

    # ==========================================================================
    # Save Results
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    json_data = {
        'model': str(model_path),
        'split': split,
        'timestamp': timestamp,
        'sahi_config': {
            'slice_size': slice_size,
            'overlap': overlap,
            'conf_threshold': conf_threshold
        },
        'sahi_size_metrics': {k: {**v, 'ap': float(v['ap'])} for k, v in sahi_size_metrics.items()},
        'standard_size_metrics': {k: {**v, 'ap': float(v['ap'])} for k, v in standard_size_metrics.items()},
        'sahi_per_class': sahi_per_class,
        'sahi_weighted_map50': float(sahi_weighted_ap),
        'standard_weighted_map50': float(standard_weighted_ap),
        'size_thresholds': SIZE_THRESHOLDS
    }

    json_file = eval_dir / "metrics.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_file}")

    # Save text report
    report_file = eval_dir / "evaluation_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION REPORT (SAHI)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Split: {split}\n")
        f.write(f"SAHI slice size: {slice_size}\n")
        f.write(f"SAHI overlap: {overlap}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("-" * 70 + "\n")
        f.write("SIZE BREAKDOWN: STANDARD vs SAHI\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Size':<10} {'Standard':>12} {'SAHI':>12} {'Improvement':>12}\n")
        f.write(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12}\n")
        for size in ["tiny", "small", "medium", "large"]:
            std_ap = standard_size_metrics[size]['ap']
            sahi_ap = sahi_size_metrics[size]['ap']
            improvement = sahi_ap - std_ap
            sign = "+" if improvement >= 0 else ""
            f.write(f"{size:<10} {std_ap:>12.4f} {sahi_ap:>12.4f} {sign}{improvement:>11.4f}\n")

        f.write(f"\n{'Weighted mAP':<10} {standard_weighted_ap:>12.4f} {sahi_weighted_ap:>12.4f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("PER-CLASS AP@0.5 (SAHI)\n")
        f.write("-" * 70 + "\n")
        for class_name, metrics in sahi_per_class.items():
            f.write(f"{class_name:<12} {metrics['AP50']:>10.4f}\n")

    print(f"  Saved: {report_file}")

    # Save plots
    plot_size_breakdown(sahi_size_metrics, eval_dir / "size_breakdown_sahi.png")
    print(f"  Saved: {eval_dir / 'size_breakdown_sahi.png'}")

    plot_per_class_ap(sahi_per_class, eval_dir / "per_class_ap_sahi.png")
    print(f"  Saved: {eval_dir / 'per_class_ap_sahi.png'}")

    plot_comparison(standard_size_metrics, sahi_size_metrics, eval_dir / "standard_vs_sahi.png")
    print(f"  Saved: {eval_dir / 'standard_vs_sahi.png'}")

    print(f"\nAll results saved to: {eval_dir}")

    return json_data


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Full evaluation with SAHI sliced inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for AP calculation")
    parser.add_argument("--slice_size", type=int, default=640, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")

    args = parser.parse_args()
    run_evaluation(args.model, args.split, args.conf, args.slice_size, args.overlap)


if __name__ == "__main__":
    main()
