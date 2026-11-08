"""
Full Model Evaluation Script
============================
Combines size-based evaluation with per-class metrics, confusion matrix, and plots.

Outputs:
- Size breakdown (tiny/small/medium/large AP50)
- Per-class AP50 and AP50-95
- Confusion matrix (normalized + raw)
- Bar chart visualizations
- JSON metrics + text report

Usage:
    python scripts/evaluate_full.py --model experiments/naive_baseline/weights/best.pt
    python scripts/evaluate_full.py --model experiments/p2_baseline/weights/best.pt
    python scripts/evaluate_full.py --model experiments/p2_nwd_12_8/weights/best.pt
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model" / "dataset.yaml"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "full_evaluation"

# Size thresholds (area in pixels)
SIZE_THRESHOLDS = {
    "tiny": 16 ** 2,      # < 256 px²
    "small": 32 ** 2,     # < 1024 px²
    "medium": 96 ** 2,    # < 9216 px²
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


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_confusion_matrix(confusion_matrix, class_names, output_path, normalize=True):
    """Plot and save confusion matrix."""
    if confusion_matrix is None:
        return

    labels = class_names + ["background"]
    fig, ax = plt.subplots(figsize=(12, 10))

    if normalize:
        cm = confusion_matrix.astype('float')
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
        fmt = '.2f'
        title = "Normalized Confusion Matrix"
    else:
        cm = confusion_matrix.astype(int)
        fmt = 'd'
        title = "Confusion Matrix (Counts)"

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=labels,
           yticklabels=labels,
           xlabel='Predicted',
           ylabel='Actual',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            text = f'{val:.2f}' if normalize else f'{int(val)}'
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_ap(per_class_metrics, output_path):
    """Plot per-class AP as bar chart."""
    classes = list(per_class_metrics.keys())
    ap50_values = [per_class_metrics[c]['AP50'] for c in classes]
    ap_values = [per_class_metrics[c]['AP50-95'] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, ap50_values, width, label='AP@0.5', color='steelblue')
    bars2 = ax.bar(x + width/2, ap_values, width, label='AP@0.5:0.95', color='darkorange')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Average Precision', fontsize=12)
    ax.set_title('Per-Class Average Precision', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
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


def plot_size_breakdown(size_metrics, output_path):
    """Plot AP by size category."""
    sizes = ["tiny", "small", "medium", "large"]
    ap_values = [size_metrics[s]['ap'] for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
    bars = ax.bar(sizes, ap_values, color=colors)

    ax.set_xlabel('Object Size Category', fontsize=12)
    ax.set_ylabel('AP@0.5', fontsize=12)
    ax.set_title('Detection Performance by Object Size', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, ap_values):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # Add threshold labels
    thresh_labels = ["<16px", "<32px", "<96px", ">=96px"]
    for i, (bar, label) in enumerate(zip(bars, thresh_labels)):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02, label,
                ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_overall_metrics(overall_metrics, output_path):
    """Plot overall metrics as horizontal bar chart."""
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    values = [
        overall_metrics['mAP50'],
        overall_metrics['mAP50-95'],
        overall_metrics['precision'],
        overall_metrics['recall']
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']
    bars = ax.barh(metrics, values, color=colors)

    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Evaluation
# =============================================================================
def run_evaluation(model_path, split='val', conf_threshold=0.001):
    """Run full evaluation with size breakdown and per-class metrics."""

    model_path = Path(model_path)
    model_name = model_path.parent.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    eval_dir = RESULTS_DIR / f"{model_name}_{split}_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL MODEL EVALUATION")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Split: {split}")
    print(f"Output: {eval_dir}")

    # Load model
    print("\nLoading model...")
    model = YOLO(str(model_path))

    # ==========================================================================
    # Part 1: Standard YOLO Validation (per-class metrics)
    # ==========================================================================
    print("\n" + "-" * 70)
    print("STANDARD YOLO VALIDATION")
    print("-" * 70)

    val_metrics = model.val(
        data=str(CONFIG_PATH),
        split=split,
        conf=0.25,
        iou=0.5,
        plots=True,
        verbose=True
    )

    # Extract overall metrics
    overall = {
        'mAP50': float(val_metrics.box.map50),
        'mAP50-95': float(val_metrics.box.map),
        'precision': float(val_metrics.box.mp),
        'recall': float(val_metrics.box.mr)
    }

    # Extract per-class metrics
    per_class = {}
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(val_metrics.box.ap50):
            per_class[class_name] = {
                'AP50': float(val_metrics.box.ap50[i]),
                'AP50-95': float(val_metrics.box.ap[i]) if i < len(val_metrics.box.ap) else 0.0
            }

    # Get confusion matrix
    confusion_matrix = None
    if hasattr(val_metrics, 'confusion_matrix') and val_metrics.confusion_matrix is not None:
        confusion_matrix = val_metrics.confusion_matrix.matrix

    # ==========================================================================
    # Part 2: Size-Based Evaluation
    # ==========================================================================
    print("\n" + "-" * 70)
    print("SIZE-BASED EVALUATION")
    print("-" * 70)

    # Load ground truths
    data_dir = PROJECT_ROOT / "data" / split
    labels_dir = data_dir / "labels" / "rgb"
    images_dir = data_dir / "images" / "rgb"

    print("\nLoading ground truth labels...")
    ground_truths = load_yolo_labels(labels_dir)
    print(f"  Loaded {len(ground_truths)} images")

    # Count GT by size
    gt_by_size = defaultdict(int)
    for img_name, boxes in ground_truths.items():
        for box in boxes:
            gt_by_size[get_size_category(box['area'])] += 1

    print("\n  Ground truth distribution:")
    for size in ["tiny", "small", "medium", "large"]:
        print(f"    {size:<8}: {gt_by_size[size]:>6} objects")

    # Run inference
    print("\nRunning inference for size evaluation...")
    results = model.predict(
        source=str(images_dir),
        conf=conf_threshold,
        iou=0.5,
        verbose=False,
        stream=True
    )

    # Collect predictions
    predictions = {}
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
        predictions[img_name] = boxes

    # Evaluate by size
    size_metrics = {}
    for size in ["tiny", "small", "medium", "large"]:
        size_metrics[size] = evaluate_size_category(predictions, ground_truths, size)

    # Weighted mAP
    total_gt = sum(r['total_gt'] for r in size_metrics.values())
    weighted_ap = sum(r['ap'] * r['total_gt'] for r in size_metrics.values()) / total_gt if total_gt > 0 else 0

    # ==========================================================================
    # Print Results
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Overall Metrics ---")
    print(f"  mAP@0.5:      {overall['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {overall['mAP50-95']:.4f}")
    print(f"  Precision:    {overall['precision']:.4f}")
    print(f"  Recall:       {overall['recall']:.4f}")

    print("\n--- Per-Class AP ---")
    print(f"  {'Class':<12} {'AP50':>10} {'AP50-95':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for class_name, metrics in per_class.items():
        print(f"  {class_name:<12} {metrics['AP50']:>10.4f} {metrics['AP50-95']:>10.4f}")

    print("\n--- Size Breakdown ---")
    print(f"  {'Size':<10} {'GT Count':>10} {'AP50':>10} {'Precision':>10} {'Recall':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for size in ["tiny", "small", "medium", "large"]:
        m = size_metrics[size]
        print(f"  {size:<10} {m['total_gt']:>10} {m['ap']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}")

    print(f"\n  Weighted mAP@50 (by size): {weighted_ap:.4f}")

    # ==========================================================================
    # Save Results
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save JSON
    json_data = {
        'model': str(model_path),
        'split': split,
        'timestamp': timestamp,
        'overall': overall,
        'per_class': per_class,
        'size_metrics': {k: {**v, 'ap': float(v['ap'])} for k, v in size_metrics.items()},
        'weighted_map50': float(weighted_ap),
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
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("-" * 70 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"mAP@0.5:      {overall['mAP50']:.4f}\n")
        f.write(f"mAP@0.5:0.95: {overall['mAP50-95']:.4f}\n")
        f.write(f"Precision:    {overall['precision']:.4f}\n")
        f.write(f"Recall:       {overall['recall']:.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("PER-CLASS AP\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<12} {'AP50':>10} {'AP50-95':>10}\n")
        f.write(f"{'-'*12} {'-'*10} {'-'*10}\n")
        for class_name, metrics in per_class.items():
            f.write(f"{class_name:<12} {metrics['AP50']:>10.4f} {metrics['AP50-95']:>10.4f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("SIZE BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Size':<10} {'GT Count':>10} {'AP50':>10} {'Precision':>10} {'Recall':>10}\n")
        f.write(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")
        for size in ["tiny", "small", "medium", "large"]:
            m = size_metrics[size]
            f.write(f"{size:<10} {m['total_gt']:>10} {m['ap']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}\n")
        f.write(f"\nWeighted mAP@50: {weighted_ap:.4f}\n")

    print(f"  Saved: {report_file}")

    # Save plots
    plot_overall_metrics(overall, eval_dir / "overall_metrics.png")
    print(f"  Saved: {eval_dir / 'overall_metrics.png'}")

    plot_per_class_ap(per_class, eval_dir / "per_class_ap.png")
    print(f"  Saved: {eval_dir / 'per_class_ap.png'}")

    plot_size_breakdown(size_metrics, eval_dir / "size_breakdown.png")
    print(f"  Saved: {eval_dir / 'size_breakdown.png'}")

    if confusion_matrix is not None:
        plot_confusion_matrix(confusion_matrix, CLASS_NAMES, eval_dir / "confusion_matrix_normalized.png", normalize=True)
        print(f"  Saved: {eval_dir / 'confusion_matrix_normalized.png'}")
        plot_confusion_matrix(confusion_matrix, CLASS_NAMES, eval_dir / "confusion_matrix_counts.png", normalize=False)
        print(f"  Saved: {eval_dir / 'confusion_matrix_counts.png'}")

    print(f"\nAll results saved to: {eval_dir}")

    return json_data


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Full model evaluation with size and class breakdown")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for size evaluation")

    args = parser.parse_args()
    run_evaluation(args.model, args.split, args.conf)


if __name__ == "__main__":
    main()
