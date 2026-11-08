"""
Inference Visualization with Best SAHI Configuration
=====================================================
Runs all 3 models (Naive Baseline, P2 Baseline, P2+SAHI) on the same images
and produces side-by-side comparison panels.

Best SAHI config (from grid search):
    slice=512, overlap=0.2, conf=0.25,
    GREEDYNMM, IOU, threshold=0.7

Usage:
    python scripts/visualize_best_sahi.py
    python scripts/visualize_best_sahi.py --split test
    python scripts/visualize_best_sahi.py --split val --num_images 30
    python scripts/visualize_best_sahi.py --split test --random
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations_best_sahi"

CLASS_NAMES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]
CLASS_COLORS = {
    0: (255, 0, 0),      # ship - blue (BGR)
    1: (0, 255, 0),      # car - green
    2: (0, 255, 255),    # cyclist - yellow
    3: (255, 0, 255),    # pedestrian - magenta
    4: (0, 165, 255),    # bus - orange
    5: (255, 255, 0),    # drone - cyan
    6: (128, 0, 128),    # plane - purple
}

# Best SAHI parameters from grid search
BEST_SAHI = {
    "slice_size": 512,
    "overlap": 0.2,
    "conf": 0.25,
    "postprocess_type": "GREEDYNMM",
    "postprocess_match_metric": "IOU",
    "postprocess_match_threshold": 0.7,
}

MODELS = {
    "naive_baseline": PROJECT_ROOT / "experiments" / "naive_baseline" / "weights" / "best.pt",
    "p2_baseline": PROJECT_ROOT / "experiments" / "p2_baseline" / "weights" / "best.pt",
}


# =============================================================================
# Inference
# =============================================================================
def run_yolo(model, img_path, conf):
    """Run standard YOLO inference."""
    results = model.predict(str(img_path), conf=conf, verbose=False)
    preds = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            preds.append({
                "class_id": int(box.cls[0].cpu().numpy()),
                "confidence": float(box.conf[0].cpu().numpy()),
                "bbox": xyxy.tolist(),
            })
    return preds


def run_sahi(detection_model, img_path):
    """Run SAHI inference with best grid-search parameters."""
    result = get_sliced_prediction(
        str(img_path),
        detection_model,
        slice_height=BEST_SAHI["slice_size"],
        slice_width=BEST_SAHI["slice_size"],
        overlap_height_ratio=BEST_SAHI["overlap"],
        overlap_width_ratio=BEST_SAHI["overlap"],
        perform_standard_pred=True,
        postprocess_type=BEST_SAHI["postprocess_type"],
        postprocess_match_metric=BEST_SAHI["postprocess_match_metric"],
        postprocess_match_threshold=BEST_SAHI["postprocess_match_threshold"],
        postprocess_class_agnostic=False,
        verbose=0,
    )

    preds = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        preds.append({
            "class_id": pred.category.id,
            "confidence": pred.score.value,
            "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        })
    return preds


# =============================================================================
# Drawing
# =============================================================================
def draw_boxes(image, preds, title=None):
    """Draw bounding boxes with class colors and confidence."""
    img = image.copy()

    for pred in preds:
        x1, y1, x2, y2 = [int(c) for c in pred["bbox"]]
        color = CLASS_COLORS.get(pred["class_id"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cls_name = CLASS_NAMES[pred["class_id"]] if pred["class_id"] < len(CLASS_NAMES) else f"cls{pred['class_id']}"
        label = f"{cls_name} {pred['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    if title:
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (5, 5), (tw + 15, th + 15), (0, 0, 0), -1)
        cv2.putText(img, title, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


def create_panel(img, naive_preds, p2_preds, sahi_preds, img_name=""):
    """Create a 3-panel (or 4-panel with GT) comparison image."""
    h, w = img.shape[:2]

    panel_naive = draw_boxes(img, naive_preds, f"Naive ({len(naive_preds)})")
    panel_p2 = draw_boxes(img, p2_preds, f"P2 ({len(p2_preds)})")
    panel_sahi = draw_boxes(img, sahi_preds, f"P2+SAHI ({len(sahi_preds)})")

    # Stack horizontally
    divider = np.full((h, 2, 3), 255, dtype=np.uint8)
    combined = np.hstack([panel_naive, divider, panel_p2, divider, panel_sahi])

    # Add filename bar at bottom
    bar_h = 30
    bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
    text = f"{img_name}  |  Naive: {len(naive_preds)}  P2: {len(p2_preds)}  P2+SAHI: {len(sahi_preds)}  (+{len(sahi_preds) - len(naive_preds)} vs naive)"
    cv2.putText(bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    combined = np.vstack([combined, bar])

    return combined


def load_ground_truth(labels_dir, img_name, img_w=640, img_h=512):
    """Load YOLO-format ground truth for a single image."""
    label_file = labels_dir / f"{img_name}.txt"
    boxes = []
    if label_file.exists():
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc = float(parts[1]) * img_w
                    yc = float(parts[2]) * img_h
                    bw = float(parts[3]) * img_w
                    bh = float(parts[4]) * img_h
                    boxes.append({
                        "class_id": cls_id,
                        "confidence": 1.0,
                        "bbox": [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2],
                    })
    return boxes


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Naive vs P2 vs P2+SAHI (best config)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Best SAHI configuration (from grid search):
  slice_size:  {BEST_SAHI['slice_size']}
  overlap:     {BEST_SAHI['overlap']}
  confidence:  {BEST_SAHI['conf']}
  postprocess: {BEST_SAHI['postprocess_type']}
  metric:      {BEST_SAHI['postprocess_match_metric']}
  threshold:   {BEST_SAHI['postprocess_match_threshold']}
        """,
    )
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--num_images", type=int, default=20, help="Number of images to visualize")
    parser.add_argument("--random", action="store_true", help="Pick random images instead of most-improved")
    parser.add_argument("--with_gt", action="store_true", help="Add a 4th panel with ground truth boxes")

    args = parser.parse_args()

    if not SAHI_AVAILABLE:
        print("ERROR: SAHI not installed. Run: pip install sahi")
        return

    # Paths
    images_dir = PROJECT_ROOT / "data" / args.split / "images" / "rgb"
    labels_dir = PROJECT_ROOT / "data" / args.split / "labels"
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    if not image_files:
        print(f"ERROR: No images found in {images_dir}")
        return

    out_dir = OUTPUT_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INFERENCE VISUALIZATION — BEST SAHI CONFIG")
    print("=" * 70)
    print(f"Split:      {args.split}")
    print(f"Images:     {len(image_files)} available, visualizing {args.num_images}")
    print(f"SAHI conf:  {BEST_SAHI}")
    print(f"Output:     {out_dir}")
    print("=" * 70)

    # Load models
    print("\nLoading models...")

    naive_model = YOLO(str(MODELS["naive_baseline"]))
    print(f"  Loaded naive_baseline")

    p2_model = YOLO(str(MODELS["p2_baseline"]))
    print(f"  Loaded p2_baseline")

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(MODELS["p2_baseline"]),
        confidence_threshold=BEST_SAHI["conf"],
        device="cuda:0",
    )
    print(f"  Loaded p2_baseline for SAHI")

    # First pass: run all 3 models on a candidate pool to rank by improvement
    pool_size = min(150, len(image_files))
    print(f"\nScanning {pool_size} images to rank by SAHI improvement...")

    candidates = []
    for i, img_path in enumerate(image_files[:pool_size]):
        if (i + 1) % 50 == 0:
            print(f"  Scanned {i + 1}/{pool_size}...")

        naive_preds = run_yolo(naive_model, img_path, BEST_SAHI["conf"])
        p2_preds = run_yolo(p2_model, img_path, BEST_SAHI["conf"])
        sahi_preds = run_sahi(sahi_model, img_path)

        # Score: how many more detections SAHI finds vs naive
        score = len(sahi_preds) - len(naive_preds)

        candidates.append({
            "path": img_path,
            "naive": naive_preds,
            "p2": p2_preds,
            "sahi": sahi_preds,
            "score": score,
        })

    # Sort: most-improved first, or shuffle for random
    if args.random:
        import random
        random.shuffle(candidates)
        print("  Selected random images.")
    else:
        candidates.sort(key=lambda c: c["score"], reverse=True)
        print(f"  Top improvement: +{candidates[0]['score']} detections")

    # Generate panels
    selected = candidates[: args.num_images]
    print(f"\nGenerating {len(selected)} comparison panels...")

    for i, cand in enumerate(selected):
        img = cv2.imread(str(cand["path"]))
        if img is None:
            continue

        if args.with_gt:
            gt_boxes = load_ground_truth(labels_dir, cand["path"].stem)
            panel_gt = draw_boxes(img, gt_boxes, f"GT ({len(gt_boxes)})")
            panel_naive = draw_boxes(img, cand["naive"], f"Naive ({len(cand['naive'])})")
            panel_p2 = draw_boxes(img, cand["p2"], f"P2 ({len(cand['p2'])})")
            panel_sahi = draw_boxes(img, cand["sahi"], f"P2+SAHI ({len(cand['sahi'])})")
            h = img.shape[0]
            div = np.full((h, 2, 3), 255, dtype=np.uint8)
            combined = np.hstack([panel_gt, div, panel_naive, div, panel_p2, div, panel_sahi])
        else:
            combined = create_panel(img, cand["naive"], cand["p2"], cand["sahi"], cand["path"].stem)

        out_path = out_dir / f"{i + 1:02d}_{cand['path'].stem}.jpg"
        cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])

        n, p, s = len(cand["naive"]), len(cand["p2"]), len(cand["sahi"])
        print(f"  [{i + 1:2d}] {cand['path'].stem}  Naive={n}  P2={p}  SAHI={s}  (+{s - n})")

    # Summary
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")
    print(f"Saved {len(selected)} comparison images to: {out_dir}")

    avg_naive = np.mean([len(c["naive"]) for c in selected])
    avg_p2 = np.mean([len(c["p2"]) for c in selected])
    avg_sahi = np.mean([len(c["sahi"]) for c in selected])
    print(f"\nAverage detections per image:")
    print(f"  Naive:    {avg_naive:.1f}")
    print(f"  P2:       {avg_p2:.1f}")
    print(f"  P2+SAHI:  {avg_sahi:.1f}")


if __name__ == "__main__":
    main()
