"""
Visualize SAHI vs Standard Inference Side-by-Side
==================================================
Creates comparison images showing what SAHI detects that standard inference misses.

Outputs:
- Side-by-side images with bounding boxes
- Green boxes: detections in both methods
- Red boxes: SAHI-only detections (what SAHI caught that standard missed)
- Blue boxes: Standard-only detections (what standard caught that SAHI missed)

Usage:
    python scripts/visualize_sahi_comparison.py --model experiments/p2_baseline/weights/best.pt
    python scripts/visualize_sahi_comparison.py --model experiments/p2_baseline/weights/best.pt --num_images 20
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
RESULTS_DIR = PROJECT_ROOT / "outputs" / "sahi_visual_comparison"

CLASS_NAMES = ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]
CLASS_COLORS = {
    0: (255, 0, 0),      # ship - blue
    1: (0, 255, 0),      # car - green
    2: (0, 255, 255),    # cyclist - yellow
    3: (255, 0, 255),    # pedestrian - magenta
    4: (0, 165, 255),    # bus - orange
    5: (255, 255, 0),    # drone - cyan
    6: (128, 0, 128),    # plane - purple
}


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


def match_detections(standard_preds, sahi_preds, iou_threshold=0.5):
    """
    Match detections between standard and SAHI predictions.

    Returns:
        matched: list of (std_box, sahi_box) tuples - detected by both
        sahi_only: list of sahi boxes - SAHI caught, standard missed
        standard_only: list of std boxes - standard caught, SAHI missed
    """
    matched = []
    sahi_only = []
    standard_only = []

    sahi_matched = [False] * len(sahi_preds)
    std_matched = [False] * len(standard_preds)

    # Match standard predictions to SAHI predictions
    for std_idx, std_pred in enumerate(standard_preds):
        best_iou = 0
        best_sahi_idx = -1

        for sahi_idx, sahi_pred in enumerate(sahi_preds):
            if sahi_matched[sahi_idx]:
                continue
            if std_pred['class_id'] != sahi_pred['class_id']:
                continue

            iou = compute_iou(std_pred['bbox'], sahi_pred['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_sahi_idx = sahi_idx

        if best_iou >= iou_threshold and best_sahi_idx >= 0:
            matched.append((std_pred, sahi_preds[best_sahi_idx]))
            sahi_matched[best_sahi_idx] = True
            std_matched[std_idx] = True

    # Collect unmatched
    for idx, pred in enumerate(sahi_preds):
        if not sahi_matched[idx]:
            sahi_only.append(pred)

    for idx, pred in enumerate(standard_preds):
        if not std_matched[idx]:
            standard_only.append(pred)

    return matched, sahi_only, standard_only


def draw_boxes(image, boxes, color, label_prefix="", thickness=2):
    """Draw bounding boxes on image."""
    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box['bbox']]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Label
        class_name = CLASS_NAMES[box['class_id']] if box['class_id'] < len(CLASS_NAMES) else f"cls{box['class_id']}"
        conf = box['confidence']
        label = f"{label_prefix}{class_name} {conf:.2f}"

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return image


def create_comparison_image(img_path, standard_preds, sahi_preds, matched, sahi_only, standard_only):
    """Create a side-by-side comparison image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    # Create three panels: Standard | SAHI | Difference
    panel_width = w
    combined = np.zeros((h, panel_width * 3, 3), dtype=np.uint8)

    # Panel 1: Standard inference
    img_std = img.copy()
    for pred in standard_preds:
        color = CLASS_COLORS.get(pred['class_id'], (255, 255, 255))
        draw_boxes(img_std, [pred], color)
    cv2.putText(img_std, f"Standard ({len(standard_preds)} det)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    combined[:, :panel_width] = img_std

    # Panel 2: SAHI inference
    img_sahi = img.copy()
    for pred in sahi_preds:
        color = CLASS_COLORS.get(pred['class_id'], (255, 255, 255))
        draw_boxes(img_sahi, [pred], color)
    cv2.putText(img_sahi, f"SAHI ({len(sahi_preds)} det)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    combined[:, panel_width:panel_width*2] = img_sahi

    # Panel 3: Difference visualization
    img_diff = img.copy()
    # Green: matched (both detected)
    for std_pred, sahi_pred in matched:
        draw_boxes(img_diff, [sahi_pred], (0, 255, 0), thickness=2)
    # Red: SAHI only (SAHI caught what standard missed)
    draw_boxes(img_diff, sahi_only, (0, 0, 255), thickness=3)
    # Blue: Standard only (standard caught what SAHI missed)
    draw_boxes(img_diff, standard_only, (255, 0, 0), thickness=2)

    # Legend
    cv2.putText(img_diff, "Green=Both | Red=SAHI only | Blue=Std only", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img_diff, f"+{len(sahi_only)} SAHI, -{len(standard_only)} Std", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    combined[:, panel_width*2:] = img_diff

    return combined


def run_visualization(model_path, split='val', num_images=10, slice_size=512, overlap=0.2, conf=0.3):
    """Run visualization comparison."""

    if not SAHI_AVAILABLE:
        print("ERROR: SAHI not installed. Run: pip install sahi")
        return

    model_path = Path(model_path)
    model_name = model_path.parent.parent.name

    output_dir = RESULTS_DIR / f"{model_name}_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAHI vs STANDARD VISUAL COMPARISON")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Slice size: {slice_size}")
    print(f"Confidence: {conf}")
    print(f"Output: {output_dir}")

    # Load models
    print("\nLoading models...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=conf,
        device='cuda:0'
    )
    model = YOLO(str(model_path))

    # Get images
    data_dir = PROJECT_ROOT / "data" / split / "images" / "rgb"
    image_files = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))

    if len(image_files) == 0:
        print(f"ERROR: No images found in {data_dir}")
        return

    # Select images with most difference (process a subset first to find interesting ones)
    print(f"\nAnalyzing {min(100, len(image_files))} images to find interesting comparisons...")

    candidates = []
    for img_path in image_files[:100]:
        # Standard inference
        results = model.predict(str(img_path), conf=conf, verbose=False)
        standard_preds = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                standard_preds.append({
                    'class_id': int(box.cls[0].cpu().numpy()),
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'bbox': xyxy.tolist()
                })

        # SAHI inference
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            perform_standard_pred=True,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            verbose=0
        )

        sahi_preds = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            sahi_preds.append({
                'class_id': pred.category.id,
                'confidence': pred.score.value,
                'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]]
            })

        # Match detections
        matched, sahi_only, standard_only = match_detections(standard_preds, sahi_preds)

        # Score by how many extra detections SAHI found
        score = len(sahi_only)
        candidates.append((img_path, standard_preds, sahi_preds, matched, sahi_only, standard_only, score))

    # Sort by score (most SAHI-only detections first)
    candidates.sort(key=lambda x: x[6], reverse=True)

    # Generate comparison images for top N
    print(f"\nGenerating {num_images} comparison images...")

    for i, (img_path, standard_preds, sahi_preds, matched, sahi_only, standard_only, score) in enumerate(candidates[:num_images]):
        combined = create_comparison_image(img_path, standard_preds, sahi_preds, matched, sahi_only, standard_only)

        if combined is not None:
            output_file = output_dir / f"comparison_{i+1:02d}_{img_path.stem}.jpg"
            cv2.imwrite(str(output_file), combined)
            print(f"  Saved: {output_file.name} (SAHI+{len(sahi_only)}, Std+{len(standard_only)})")

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_sahi_only = sum(len(c[4]) for c in candidates[:num_images])
    total_std_only = sum(len(c[5]) for c in candidates[:num_images])
    print(f"  Total SAHI-only detections: {total_sahi_only}")
    print(f"  Total Standard-only detections: {total_std_only}")
    print(f"\n  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SAHI vs Standard inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--num_images", type=int, default=10, help="Number of comparison images to generate")
    parser.add_argument("--slice_size", type=int, default=512, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")

    args = parser.parse_args()
    run_visualization(args.model, args.split, args.num_images, args.slice_size, args.overlap, args.conf)


if __name__ == "__main__":
    main()
