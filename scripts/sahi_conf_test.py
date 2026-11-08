"""
SAHI Confidence Test
====================
Quick test: does raising confidence make SAHI competitive?

Runs 6 experiments:
  - Standard @ conf=0.1, 0.25, 0.5
  - SAHI slice=128 overlap=0.2 @ conf=0.1, 0.25, 0.5
  - SAHI slice=512 overlap=0.2 @ conf=0.1, 0.25, 0.5

Usage:
    python scripts/sahi_conf_test.py
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent

SIZE_THRESHOLDS = {"tiny": 16**2, "small": 32**2, "medium": 96**2}
CLASS_NAMES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]

POSTPROCESS = {
    "postprocess_type": "GREEDYNMM",
    "postprocess_match_metric": "IOS",
    "postprocess_match_threshold": 0.5,
    "perform_standard_pred": True,
}

CONF_THRESHOLDS = [0.1, 0.25]
SAHI_CONFIGS = [
    {"slice_size": 128, "overlap": 0.2},
    {"slice_size": 512, "overlap": 0.2},
]


# =============================================================================
# Eval helpers
# =============================================================================
def load_yolo_labels(labels_dir, img_width=640, img_height=512):
    ground_truths = {}
    for label_file in Path(labels_dir).glob("*.txt"):
        boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    xc, yc = float(parts[1]) * img_width, float(parts[2]) * img_height
                    w, h = float(parts[3]) * img_width, float(parts[4]) * img_height
                    boxes.append({
                        'class_id': cls,
                        'bbox': [xc - w/2, yc - h/2, xc + w/2, yc + h/2],
                        'area': w * h
                    })
        ground_truths[label_file.stem] = boxes
    return ground_truths


def size_cat(area):
    if area < SIZE_THRESHOLDS["tiny"]: return "tiny"
    elif area < SIZE_THRESHOLDS["small"]: return "small"
    elif area < SIZE_THRESHOLDS["medium"]: return "medium"
    else: return "large"


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0


def compute_ap(precs, recs):
    if not precs: return 0.0
    r, p = np.array(recs), np.array(precs)
    idx = np.argsort(r); r, p = r[idx], p[idx]
    r = np.concatenate([[0], r, [1]]); p = np.concatenate([[0], p, [0]])
    for i in range(len(p)-2, -1, -1): p[i] = max(p[i], p[i+1])
    return sum(p[r >= t].max() for t in np.linspace(0, 1, 101) if (r >= t).any()) / 101


def eval_size(preds, gts, size):
    dets, total_gt = [], 0
    for img, gt_boxes in gts.items():
        gt_f = [g for g in gt_boxes if size_cat(g['area']) == size]
        total_gt += len(gt_f)
        pbs = sorted(preds.get(img, []), key=lambda x: x['confidence'], reverse=True)
        matched = [False] * len(gt_f)
        for p in pbs:
            pa = (p['bbox'][2]-p['bbox'][0]) * (p['bbox'][3]-p['bbox'][1])
            if size_cat(pa) != size: continue
            best_iou, best_idx = 0, -1
            for gi, g in enumerate(gt_f):
                if matched[gi] or g['class_id'] != p['class_id']: continue
                iou = compute_iou(p['bbox'], g['bbox'])
                if iou > best_iou: best_iou, best_idx = iou, gi
            if best_iou >= 0.5 and best_idx >= 0:
                dets.append((p['confidence'], True)); matched[best_idx] = True
            else:
                dets.append((p['confidence'], False))
    if total_gt == 0: return {'ap': 0.0, 'total_gt': 0, 'total_pred': len(dets)}
    dets.sort(key=lambda x: x[0], reverse=True)
    tp, fp, precs, recs = 0, 0, [], []
    for _, is_tp in dets:
        if is_tp: tp += 1
        else: fp += 1
        precs.append(tp/(tp+fp)); recs.append(tp/total_gt)
    return {'ap': float(compute_ap(precs, recs)), 'total_gt': total_gt, 'total_pred': len(dets)}


def eval_overall(preds, gts):
    dets, total_gt = [], 0
    for img, gt_boxes in gts.items():
        total_gt += len(gt_boxes)
        pbs = sorted(preds.get(img, []), key=lambda x: x['confidence'], reverse=True)
        matched = [False] * len(gt_boxes)
        for p in pbs:
            best_iou, best_idx = 0, -1
            for gi, g in enumerate(gt_boxes):
                if matched[gi] or g['class_id'] != p['class_id']: continue
                iou = compute_iou(p['bbox'], g['bbox'])
                if iou > best_iou: best_iou, best_idx = iou, gi
            if best_iou >= 0.5 and best_idx >= 0:
                dets.append((p['confidence'], True)); matched[best_idx] = True
            else:
                dets.append((p['confidence'], False))
    if total_gt == 0: return {'mAP50': 0.0}
    dets.sort(key=lambda x: x[0], reverse=True)
    tp, fp, precs, recs = 0, 0, [], []
    for _, is_tp in dets:
        if is_tp: tp += 1
        else: fp += 1
        precs.append(tp/(tp+fp)); recs.append(tp/total_gt)
    return {'mAP50': float(compute_ap(precs, recs)), 'precision': precs[-1], 'recall': recs[-1]}


def all_metrics(preds, gts):
    sm = {s: eval_size(preds, gts, s) for s in ["tiny", "small", "medium", "large"]}
    tgt = sum(m['total_gt'] for m in sm.values())
    wmap = sum(m['ap'] * m['total_gt'] for m in sm.values()) / tgt if tgt > 0 else 0
    return {
        'size_metrics': sm, 'overall': eval_overall(preds, gts),
        'weighted_map50': float(wmap),
        'total_predictions': sum(len(v) for v in preds.values()),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    model_path = PROJECT_ROOT / "experiments" / "p2_baseline" / "weights" / "best.pt"
    split = "val"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "sahi_grid_search" / f"conf_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data" / split
    labels_dir = data_dir / "labels" / "rgb"
    images_dir = data_dir / "images" / "rgb"

    print("Loading ground truths...")
    gts = load_yolo_labels(labels_dir)
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"  {len(gts)} images, {len(image_files)} files")

    yolo_model = YOLO(str(model_path))
    results_table = []

    for conf in CONF_THRESHOLDS:
        print(f"\n{'='*70}")
        print(f"  CONFIDENCE = {conf}")
        print(f"{'='*70}")

        # Standard baseline
        print(f"\n  Standard inference (conf={conf})...")
        t0 = time.time()
        std_results = yolo_model.predict(source=str(images_dir), conf=conf, iou=0.5, verbose=False, stream=True)
        std_preds = {}
        for r in std_results:
            boxes = []
            if r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes.append({'class_id': int(box.cls[0].cpu().numpy()),
                                  'confidence': float(box.conf[0].cpu().numpy()),
                                  'bbox': xyxy.tolist()})
            std_preds[Path(r.path).stem] = boxes
        std_time = time.time() - t0
        std_m = all_metrics(std_preds, gts)

        entry = {
            'name': f'standard_conf{conf}', 'type': 'standard', 'confidence': conf,
            'slice_size': None, 'overlap': None, 'time_s': round(std_time, 1), **std_m
        }
        results_table.append(entry)
        sm = std_m['size_metrics']
        print(f"    mAP50={std_m['overall']['mAP50']:.4f}  W-mAP={std_m['weighted_map50']:.4f}  "
              f"Tiny={sm['tiny']['ap']:.4f}  Small={sm['small']['ap']:.4f}  Preds={std_m['total_predictions']}")

        # SAHI configs
        det_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', model_path=str(model_path),
            confidence_threshold=conf, device='cuda:0',
        )

        for cfg in SAHI_CONFIGS:
            ss, ov = cfg['slice_size'], cfg['overlap']
            print(f"\n  SAHI slice={ss} overlap={ov} conf={conf}...")

            t0 = time.time()
            sahi_preds = {}
            for i, img_path in enumerate(image_files):
                if (i+1) % 200 == 0: print(f"      {i+1}/{len(image_files)}...")
                result = get_sliced_prediction(
                    str(img_path), det_model,
                    slice_height=ss, slice_width=ss,
                    overlap_height_ratio=ov, overlap_width_ratio=ov,
                    perform_standard_pred=True,
                    postprocess_type="GREEDYNMM",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.5,
                    postprocess_class_agnostic=False, verbose=0,
                )
                boxes = []
                for pred in result.object_prediction_list:
                    bbox = pred.bbox.to_xyxy()
                    boxes.append({'class_id': pred.category.id, 'confidence': pred.score.value,
                                  'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]]})
                sahi_preds[img_path.stem] = boxes
            elapsed = time.time() - t0

            sahi_m = all_metrics(sahi_preds, gts)
            diff_map = sahi_m['overall']['mAP50'] - std_m['overall']['mAP50']
            diff_tiny = sahi_m['size_metrics']['tiny']['ap'] - sm['tiny']['ap']

            entry = {
                'name': f'sahi_s{ss}_conf{conf}', 'type': 'sahi', 'confidence': conf,
                'slice_size': ss, 'overlap': ov, 'time_s': round(elapsed, 1),
                'diff_mAP50_vs_std': round(diff_map, 4),
                'diff_tiny_vs_std': round(diff_tiny, 4),
                **sahi_m
            }
            results_table.append(entry)
            ssm = sahi_m['size_metrics']
            print(f"    mAP50={sahi_m['overall']['mAP50']:.4f} ({diff_map:+.4f})  "
                  f"W-mAP={sahi_m['weighted_map50']:.4f}  "
                  f"Tiny={ssm['tiny']['ap']:.4f} ({diff_tiny:+.4f})  "
                  f"Preds={sahi_m['total_predictions']}")

    # =========================================================================
    # Final table
    # =========================================================================
    print(f"\n{'='*110}")
    print("FINAL COMPARISON")
    print(f"{'='*110}")
    print(f"{'Name':<25} {'Conf':>5} {'mAP50':>8} {'W-mAP':>8} {'Tiny':>8} {'Small':>8} {'Med':>8} {'Large':>8} {'Preds':>9} {'vs Std':>8}")
    print("-" * 110)

    for r in results_table:
        sm = r['size_metrics']
        diff = r.get('diff_mAP50_vs_std', '')
        diff_str = f"{diff:+.4f}" if diff != '' else "  base"
        print(f"{r['name']:<25} {r['confidence']:>5} {r['overall']['mAP50']:>8.4f} {r['weighted_map50']:>8.4f} "
              f"{sm['tiny']['ap']:>8.4f} {sm['small']['ap']:>8.4f} "
              f"{sm['medium']['ap']:>8.4f} {sm['large']['ap']:>8.4f} "
              f"{r['total_predictions']:>9} {diff_str:>8}")

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_table, f, indent=2)
    print(f"\nSaved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
