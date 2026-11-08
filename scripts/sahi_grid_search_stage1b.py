"""
SAHI Grid Search - Stage 1b: Confidence Threshold Search
=========================================================
Tests whether raising confidence for BOTH standard and SAHI inference
makes SAHI competitive. Stage 1 showed SAHI worse at conf=0.001 due to
false positive explosion from slicing.

Grid:
  confidence:  [0.1, 0.25, 0.5]
  slice_size:  [128, 256, 512]
  overlap:     [0.2]
  = 9 SAHI combos + 3 standard baselines (one per confidence)

Fixed:
  postprocess_type = GREEDYNMM
  postprocess_match_metric = IOS
  postprocess_match_threshold = 0.5
  perform_standard_pred = True

Usage:
    python scripts/sahi_grid_search_stage1b.py
    python scripts/sahi_grid_search_stage1b.py --model experiments/p2_baseline/weights/best.pt
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

# Fixed postprocess parameters
FIXED_POSTPROCESS = {
    "postprocess_type": "GREEDYNMM",
    "postprocess_match_metric": "IOS",
    "postprocess_match_threshold": 0.5,
    "perform_standard_pred": True,
}

# Search grid
CONFIDENCE_THRESHOLDS = [0.1, 0.25, 0.5]
SLICE_SIZES = [128, 256, 512]
OVERLAP = 0.2  # Fixed from stage 1 best


# =============================================================================
# Helper Functions
# =============================================================================
def load_yolo_labels(labels_dir, img_width=640, img_height=512):
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


def compute_all_metrics(predictions, ground_truths):
    size_metrics = {}
    for size in ["tiny", "small", "medium", "large"]:
        size_metrics[size] = evaluate_size_category(predictions, ground_truths, size)
    total_gt = sum(m['total_gt'] for m in size_metrics.values())
    weighted_map = (
        sum(m['ap'] * m['total_gt'] for m in size_metrics.values()) / total_gt
        if total_gt > 0 else 0.0
    )
    overall = evaluate_overall(predictions, ground_truths)
    return {
        'size_metrics': size_metrics,
        'overall': overall,
        'weighted_map50': float(weighted_map),
        'total_predictions': sum(len(v) for v in predictions.values()),
    }


# =============================================================================
# Inference Functions
# =============================================================================
def run_sahi_inference(detection_model, image_files, slice_size, overlap):
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
            perform_standard_pred=FIXED_POSTPROCESS["perform_standard_pred"],
            postprocess_type=FIXED_POSTPROCESS["postprocess_type"],
            postprocess_match_metric=FIXED_POSTPROCESS["postprocess_match_metric"],
            postprocess_match_threshold=FIXED_POSTPROCESS["postprocess_match_threshold"],
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


def run_standard_inference(model, images_dir, conf_threshold):
    results = model.predict(
        source=str(images_dir),
        conf=conf_threshold,
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


# =============================================================================
# Main
# =============================================================================
def run_grid_search(model_path, split='val'):
    if not SAHI_AVAILABLE:
        print("ERROR: SAHI not installed. Run: pip install sahi")
        return

    model_path = Path(model_path)
    model_name = model_path.parent.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = PROJECT_ROOT / "outputs" / "sahi_grid_search" / f"stage1b_{model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_combos = len(CONFIDENCE_THRESHOLDS) * len(SLICE_SIZES)

    print("=" * 70)
    print("SAHI GRID SEARCH - STAGE 1b (Confidence Threshold)")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Split: {split}")
    print(f"Output: {output_dir}")
    print(f"\nFixed postprocess: {FIXED_POSTPROCESS}")
    print(f"Fixed overlap: {OVERLAP}")
    print(f"\nSearch grid:")
    print(f"  confidence:  {CONFIDENCE_THRESHOLDS}")
    print(f"  slice_sizes: {SLICE_SIZES}")
    print(f"  Total: {len(CONFIDENCE_THRESHOLDS)} baselines + {total_combos} SAHI combos")

    # Load data
    data_dir = PROJECT_ROOT / "data" / split
    labels_dir = data_dir / "labels" / "rgb"
    images_dir = data_dir / "images" / "rgb"

    print("\nLoading ground truth labels...")
    ground_truths = load_yolo_labels(labels_dir)

    gt_by_size = defaultdict(int)
    for boxes in ground_truths.values():
        for box in boxes:
            gt_by_size[get_size_category(box['area'])] += 1
    print(f"  {len(ground_truths)} images")
    for size in ["tiny", "small", "medium", "large"]:
        print(f"    {size:<8}: {gt_by_size[size]:>6} objects")

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"  {len(image_files)} image files")

    yolo_model = YOLO(str(model_path))

    all_results = []
    baselines = {}
    combo_idx = 0

    for conf_thresh in CONFIDENCE_THRESHOLDS:
        print(f"\n{'#' * 70}")
        print(f"  CONFIDENCE THRESHOLD = {conf_thresh}")
        print(f"{'#' * 70}")

        # Standard baseline at this confidence
        print(f"\n  Running standard inference (conf={conf_thresh})...")
        t0 = time.time()
        std_preds = run_standard_inference(yolo_model, images_dir, conf_thresh)
        std_time = time.time() - t0
        std_metrics = compute_all_metrics(std_preds, ground_truths)

        baselines[conf_thresh] = {
            'type': 'standard_baseline',
            'confidence': conf_thresh,
            'inference_time_s': round(std_time, 2),
            **std_metrics,
        }

        print(f"    Done in {std_time:.1f}s")
        print(f"    mAP50:  {std_metrics['overall']['mAP50']:.4f}")
        print(f"    W-mAP:  {std_metrics['weighted_map50']:.4f}")
        print(f"    Tiny:   {std_metrics['size_metrics']['tiny']['ap']:.4f}")
        print(f"    Preds:  {std_metrics['total_predictions']}")

        # Load SAHI model at this confidence
        print(f"\n  Loading SAHI model (conf={conf_thresh})...")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=str(model_path),
            confidence_threshold=conf_thresh,
            device='cuda:0',
        )

        # SAHI combos at this confidence
        for slice_size in SLICE_SIZES:
            combo_idx += 1
            combo_name = f"conf{conf_thresh}_slice{slice_size}"

            print(f"\n  [{combo_idx}/{total_combos}] conf={conf_thresh}, slice={slice_size}, overlap={OVERLAP}")

            t0 = time.time()
            sahi_preds = run_sahi_inference(detection_model, image_files, slice_size, OVERLAP)
            elapsed = time.time() - t0

            metrics = compute_all_metrics(sahi_preds, ground_truths)

            # Improvements vs SAME-confidence standard baseline
            improvements = {
                'mAP50_diff': metrics['overall']['mAP50'] - std_metrics['overall']['mAP50'],
                'weighted_map50_diff': metrics['weighted_map50'] - std_metrics['weighted_map50'],
                'tiny_ap_diff': metrics['size_metrics']['tiny']['ap'] - std_metrics['size_metrics']['tiny']['ap'],
                'small_ap_diff': metrics['size_metrics']['small']['ap'] - std_metrics['size_metrics']['small']['ap'],
            }

            result = {
                'combo_name': combo_name,
                'confidence': conf_thresh,
                'slice_size': slice_size,
                'overlap': OVERLAP,
                'inference_time_s': round(elapsed, 2),
                'improvements_vs_same_conf_standard': improvements,
                **metrics,
            }
            all_results.append(result)

            print(f"    Time: {elapsed:.1f}s")
            print(f"    mAP50:  {metrics['overall']['mAP50']:.4f}  ({improvements['mAP50_diff']:+.4f} vs std)")
            print(f"    W-mAP:  {metrics['weighted_map50']:.4f}  ({improvements['weighted_map50_diff']:+.4f} vs std)")
            print(f"    Tiny:   {metrics['size_metrics']['tiny']['ap']:.4f}  ({improvements['tiny_ap_diff']:+.4f} vs std)")
            print(f"    Small:  {metrics['size_metrics']['small']['ap']:.4f}  ({improvements['small_ap_diff']:+.4f} vs std)")
            print(f"    Preds:  {metrics['total_predictions']}")

            with open(output_dir / f"{combo_name}.json", 'w') as f:
                json.dump(result, f, indent=2)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 100}")
    print("STAGE 1b COMPLETE - FULL COMPARISON")
    print(f"{'=' * 100}")

    # Print baselines
    print(f"\n--- Standard Baselines ---")
    print(f"{'Conf':>6} {'mAP50':>8} {'W-mAP':>8} {'Tiny':>8} {'Small':>8} {'Med':>8} {'Large':>8} {'Preds':>8} {'Time':>7}")
    print("-" * 80)
    for conf_thresh in CONFIDENCE_THRESHOLDS:
        b = baselines[conf_thresh]
        sm = b['size_metrics']
        print(f"{conf_thresh:>6} {b['overall']['mAP50']:>8.4f} {b['weighted_map50']:>8.4f} "
              f"{sm['tiny']['ap']:>8.4f} {sm['small']['ap']:>8.4f} "
              f"{sm['medium']['ap']:>8.4f} {sm['large']['ap']:>8.4f} "
              f"{b['total_predictions']:>8} {b['inference_time_s']:>6.1f}s")

    # Print all SAHI results grouped by confidence
    print(f"\n--- SAHI Results (improvement vs same-conf standard) ---")
    print(f"{'Config':<25} {'mAP50':>8} {'W-mAP':>8} {'Tiny':>8} {'Small':>8} {'Preds':>8} {'mAP50+':>8} {'Tiny+':>8}")
    print("-" * 95)

    for conf_thresh in CONFIDENCE_THRESHOLDS:
        conf_results = [r for r in all_results if r['confidence'] == conf_thresh]
        for r in conf_results:
            sm = r['size_metrics']
            imp = r['improvements_vs_same_conf_standard']
            print(f"{r['combo_name']:<25} {r['overall']['mAP50']:>8.4f} {r['weighted_map50']:>8.4f} "
                  f"{sm['tiny']['ap']:>8.4f} {sm['small']['ap']:>8.4f} "
                  f"{r['total_predictions']:>8} {imp['mAP50_diff']:>+8.4f} {imp['tiny_ap_diff']:>+8.4f}")
        print()

    # Save summary
    summary = {
        'model': str(model_path),
        'split': split,
        'timestamp': timestamp,
        'fixed_postprocess': FIXED_POSTPROCESS,
        'fixed_overlap': OVERLAP,
        'search_grid': {
            'confidence_thresholds': CONFIDENCE_THRESHOLDS,
            'slice_sizes': SLICE_SIZES,
        },
        'baselines': {str(k): v for k, v in baselines.items()},
        'results': all_results,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"All results saved to: {output_dir}")
    return summary


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="SAHI Grid Search - Stage 1b (Confidence)")
    parser.add_argument("--model", type=str,
                        default="experiments/p2_baseline/weights/best.pt",
                        help="Path to model weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    run_grid_search(str(model_path), args.split)


if __name__ == "__main__":
    main()
