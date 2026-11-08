"""
SAHI Grid Search - Stage 1: Slice Size x Overlap
=================================================
Searches over slice_size and overlap_ratio while fixing:
  - postprocess_type = GREEDYNMM
  - postprocess_match_metric = IOS
  - postprocess_match_threshold = 0.5
  - perform_standard_pred = True
  - confidence_threshold = 0.001

Grid: 4 slice sizes x 4 overlaps = 16 combinations.
Also runs standard inference once as the baseline.

Usage:
    python scripts/sahi_grid_search_stage1.py
    python scripts/sahi_grid_search_stage1.py --model experiments/p2_baseline/weights/best.pt
    python scripts/sahi_grid_search_stage1.py --split test

Requirements:
    pip install sahi
"""

import argparse
import json
import time
import numpy as np
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
    print("ERROR: SAHI not installed. Run: pip install sahi")

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent

SIZE_THRESHOLDS = {
    "tiny": 16 ** 2,      # < 256 px²
    "small": 32 ** 2,     # < 1024 px²
    "medium": 96 ** 2,    # < 9216 px²
}

CLASS_NAMES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]

# Fixed SAHI parameters for Stage 1
FIXED_PARAMS = {
    "confidence_threshold": 0.001,
    "postprocess_type": "GREEDYNMM",
    "postprocess_match_metric": "IOS",
    "postprocess_match_threshold": 0.5,
    "perform_standard_pred": True,
}

# Stage 1 search grid
SLICE_SIZES = [128, 256, 320, 512]
OVERLAP_RATIOS = [0.1, 0.2, 0.3, 0.4]


# =============================================================================
# Helper Functions (from evaluate_full_obss_sahi.py)
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
    if area < SIZE_THRESHOLDS["tiny"]:
        return "tiny"
    elif area < SIZE_THRESHOLDS["small"]:
        return "small"
    elif area < SIZE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "large"


def compute_iou(box1, box2):
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
                all_detections.append((pred['confidence'], True))
                gt_matched[best_gt_idx] = True
            else:
                all_detections.append((pred['confidence'], False))
    if total_gt == 0:
        return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'total_gt': 0, 'total_pred': len(all_detections)}
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
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / total_gt)
    ap = compute_ap(precisions, recalls)
    return {
        'ap': float(ap),
        'precision': float(precisions[-1]) if precisions else 0.0,
        'recall': float(recalls[-1]) if recalls else 0.0,
        'total_gt': total_gt,
        'total_pred': len(all_detections)
    }


def evaluate_per_class(predictions, ground_truths, iou_threshold=0.5):
    per_class_metrics = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        all_detections = []
        total_gt = 0
        for img_name, gt_boxes in ground_truths.items():
            gt_filtered = [gt for gt in gt_boxes if gt['class_id'] == class_id]
            total_gt += len(gt_filtered)
            pred_boxes = predictions.get(img_name, [])
            pred_filtered = sorted(
                [p for p in pred_boxes if p['class_id'] == class_id],
                key=lambda x: x['confidence'], reverse=True
            )
            gt_matched = [False] * len(gt_filtered)
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
            per_class_metrics[class_name] = {'AP50': 0.0}
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
            precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            recalls.append(tp_cumsum / total_gt)
        ap = compute_ap(precisions, recalls)
        per_class_metrics[class_name] = {'AP50': float(ap)}
    return per_class_metrics


def evaluate_overall(predictions, ground_truths, iou_threshold=0.5):
    all_detections = []
    total_gt = 0
    for img_name, gt_boxes in ground_truths.items():
        total_gt += len(gt_boxes)
        pred_boxes = predictions.get(img_name, [])
        gt_matched = [False] * len(gt_boxes)
        pred_boxes = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                if gt['class_id'] != pred['class_id']:
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
        return {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0}
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
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / total_gt)
    ap = compute_ap(precisions, recalls)
    return {
        'mAP50': float(ap),
        'precision': float(precisions[-1]) if precisions else 0.0,
        'recall': float(recalls[-1]) if recalls else 0.0,
    }


# =============================================================================
# Inference Functions
# =============================================================================
def run_sahi_inference(detection_model, image_files, slice_size, overlap):
    """Run SAHI sliced inference with given parameters."""
    predictions = {}
    for i, img_path in enumerate(image_files):
        if (i + 1) % 200 == 0:
            print(f"      {i + 1}/{len(image_files)} images...")

        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            perform_standard_pred=FIXED_PARAMS["perform_standard_pred"],
            postprocess_type=FIXED_PARAMS["postprocess_type"],
            postprocess_match_metric=FIXED_PARAMS["postprocess_match_metric"],
            postprocess_match_threshold=FIXED_PARAMS["postprocess_match_threshold"],
            postprocess_class_agnostic=False,
            verbose=0,
        )

        boxes = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            boxes.append({
                'class_id': pred.category.id,
                'confidence': pred.score.value,
                'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]]
            })
        predictions[img_path.stem] = boxes

    return predictions


def run_standard_inference(model, images_dir):
    """Run standard YOLO inference (baseline, done once)."""
    results = model.predict(
        source=str(images_dir),
        conf=FIXED_PARAMS["confidence_threshold"],
        iou=0.5,
        verbose=False,
        stream=True,
    )
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
    return predictions


def compute_all_metrics(predictions, ground_truths):
    """Compute all metrics for a set of predictions."""
    # Size-based
    size_metrics = {}
    for size in ["tiny", "small", "medium", "large"]:
        size_metrics[size] = evaluate_size_category(predictions, ground_truths, size)

    # Weighted mAP
    total_gt = sum(m['total_gt'] for m in size_metrics.values())
    weighted_map = (
        sum(m['ap'] * m['total_gt'] for m in size_metrics.values()) / total_gt
        if total_gt > 0 else 0.0
    )

    # Per-class
    per_class = evaluate_per_class(predictions, ground_truths)
    mean_per_class_ap = float(np.mean([v['AP50'] for v in per_class.values()]))

    # Overall
    overall = evaluate_overall(predictions, ground_truths)

    return {
        'size_metrics': size_metrics,
        'per_class': per_class,
        'overall': overall,
        'weighted_map50': float(weighted_map),
        'mean_per_class_ap50': mean_per_class_ap,
        'total_predictions': sum(len(v) for v in predictions.values()),
    }


# =============================================================================
# Main Grid Search
# =============================================================================
def run_grid_search(model_path, split='val'):
    if not SAHI_AVAILABLE:
        print("ERROR: SAHI not installed. Run: pip install sahi")
        return

    model_path = Path(model_path)
    model_name = model_path.parent.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = PROJECT_ROOT / "outputs" / "sahi_grid_search" / f"stage1_{model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAHI GRID SEARCH - STAGE 1")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Split: {split}")
    print(f"Output: {output_dir}")
    print(f"\nFixed parameters:")
    for k, v in FIXED_PARAMS.items():
        print(f"  {k}: {v}")
    print(f"\nSearch grid:")
    print(f"  slice_sizes: {SLICE_SIZES}")
    print(f"  overlap_ratios: {OVERLAP_RATIOS}")
    print(f"  Total combinations: {len(SLICE_SIZES) * len(OVERLAP_RATIOS)}")

    # Load data paths
    data_dir = PROJECT_ROOT / "data" / split
    labels_dir = data_dir / "labels" / "rgb"
    images_dir = data_dir / "images" / "rgb"

    # Load ground truths
    print("\nLoading ground truth labels...")
    ground_truths = load_yolo_labels(labels_dir)

    gt_by_size = defaultdict(int)
    for boxes in ground_truths.values():
        for box in boxes:
            gt_by_size[get_size_category(box['area'])] += 1
    print(f"  {len(ground_truths)} images")
    for size in ["tiny", "small", "medium", "large"]:
        print(f"    {size:<8}: {gt_by_size[size]:>6} objects")

    # Collect image paths
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"  {len(image_files)} image files")

    # Load models (once)
    print("\nLoading SAHI detection model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=FIXED_PARAMS["confidence_threshold"],
        device='cuda:0',
    )

    print("Loading YOLO model for standard inference...")
    yolo_model = YOLO(str(model_path))

    # =========================================================================
    # Run standard inference (baseline, once)
    # =========================================================================
    print("\n" + "-" * 70)
    print("BASELINE: Standard inference")
    print("-" * 70)
    t0 = time.time()
    standard_preds = run_standard_inference(yolo_model, images_dir)
    std_time = time.time() - t0
    print(f"  Done in {std_time:.1f}s")

    standard_metrics = compute_all_metrics(standard_preds, ground_truths)
    print(f"  mAP50: {standard_metrics['overall']['mAP50']:.4f}")
    print(f"  Weighted mAP50: {standard_metrics['weighted_map50']:.4f}")
    print(f"  Tiny AP50: {standard_metrics['size_metrics']['tiny']['ap']:.4f}")

    # Save standard baseline
    baseline_data = {
        'type': 'standard_baseline',
        'model': str(model_path),
        'split': split,
        'inference_time_s': round(std_time, 2),
        **standard_metrics,
    }
    with open(output_dir / "baseline_standard.json", 'w') as f:
        json.dump(baseline_data, f, indent=2)

    # =========================================================================
    # Grid search
    # =========================================================================
    all_results = []
    total_combos = len(SLICE_SIZES) * len(OVERLAP_RATIOS)
    combo_idx = 0

    for slice_size in SLICE_SIZES:
        for overlap in OVERLAP_RATIOS:
            combo_idx += 1
            combo_name = f"slice{slice_size}_overlap{overlap}"

            print(f"\n{'=' * 70}")
            print(f"  [{combo_idx}/{total_combos}] slice_size={slice_size}, overlap={overlap}")
            print(f"{'=' * 70}")

            t0 = time.time()
            sahi_preds = run_sahi_inference(detection_model, image_files, slice_size, overlap)
            elapsed = time.time() - t0

            metrics = compute_all_metrics(sahi_preds, ground_truths)

            # Compute improvements over baseline
            improvements = {
                'mAP50_diff': metrics['overall']['mAP50'] - standard_metrics['overall']['mAP50'],
                'weighted_map50_diff': metrics['weighted_map50'] - standard_metrics['weighted_map50'],
                'tiny_ap_diff': metrics['size_metrics']['tiny']['ap'] - standard_metrics['size_metrics']['tiny']['ap'],
                'small_ap_diff': metrics['size_metrics']['small']['ap'] - standard_metrics['size_metrics']['small']['ap'],
            }

            result = {
                'combo_name': combo_name,
                'slice_size': slice_size,
                'overlap': overlap,
                'inference_time_s': round(elapsed, 2),
                'improvements_vs_standard': improvements,
                **metrics,
            }
            all_results.append(result)

            # Print summary
            print(f"    Time: {elapsed:.1f}s")
            print(f"    mAP50:         {metrics['overall']['mAP50']:.4f}  ({improvements['mAP50_diff']:+.4f})")
            print(f"    Weighted mAP:  {metrics['weighted_map50']:.4f}  ({improvements['weighted_map50_diff']:+.4f})")
            print(f"    Tiny AP50:     {metrics['size_metrics']['tiny']['ap']:.4f}  ({improvements['tiny_ap_diff']:+.4f})")
            print(f"    Small AP50:    {metrics['size_metrics']['small']['ap']:.4f}  ({improvements['small_ap_diff']:+.4f})")
            print(f"    Predictions:   {metrics['total_predictions']}")

            # Save individual result
            with open(output_dir / f"{combo_name}.json", 'w') as f:
                json.dump(result, f, indent=2)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE - STAGE 1 SUMMARY")
    print("=" * 70)

    # Sort by weighted mAP
    sorted_results = sorted(all_results, key=lambda r: r['weighted_map50'], reverse=True)

    print(f"\n{'Rank':<6} {'Config':<25} {'mAP50':>8} {'W-mAP':>8} {'Tiny':>8} {'Small':>8} {'Med':>8} {'Large':>8} {'Time':>7}")
    print("-" * 98)

    # Print baseline
    bm = standard_metrics
    print(f"{'base':<6} {'standard':<25} {bm['overall']['mAP50']:>8.4f} {bm['weighted_map50']:>8.4f} "
          f"{bm['size_metrics']['tiny']['ap']:>8.4f} {bm['size_metrics']['small']['ap']:>8.4f} "
          f"{bm['size_metrics']['medium']['ap']:>8.4f} {bm['size_metrics']['large']['ap']:>8.4f} "
          f"{std_time:>6.1f}s")
    print("-" * 98)

    for rank, r in enumerate(sorted_results, 1):
        sm = r['size_metrics']
        print(f"{'#'+str(rank):<6} {r['combo_name']:<25} {r['overall']['mAP50']:>8.4f} {r['weighted_map50']:>8.4f} "
              f"{sm['tiny']['ap']:>8.4f} {sm['small']['ap']:>8.4f} "
              f"{sm['medium']['ap']:>8.4f} {sm['large']['ap']:>8.4f} "
              f"{r['inference_time_s']:>6.1f}s")

    # Save full summary
    summary = {
        'model': str(model_path),
        'split': split,
        'timestamp': timestamp,
        'fixed_params': FIXED_PARAMS,
        'search_grid': {
            'slice_sizes': SLICE_SIZES,
            'overlap_ratios': OVERLAP_RATIOS,
        },
        'baseline': baseline_data,
        'results': sorted_results,
        'best_by_weighted_map50': sorted_results[0]['combo_name'],
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest config (weighted mAP50): {sorted_results[0]['combo_name']}")
    print(f"All results saved to: {output_dir}")

    return summary


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="SAHI Grid Search - Stage 1")
    parser.add_argument("--model", type=str,
                        default="experiments/p2_baseline/weights/best.pt",
                        help="Path to model weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])

    args = parser.parse_args()

    # Resolve relative paths
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    run_grid_search(str(model_path), args.split)


if __name__ == "__main__":
    main()
