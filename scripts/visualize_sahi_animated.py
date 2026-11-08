"""
Animated SAHI vs Standard Inference Visualizations
===================================================
Creates cool visualizations comparing SAHI and standard inference:
1. Alternating GIF - flips between standard and SAHI detections
2. Slider image - vertical bar dividing standard (left) vs SAHI (right)
3. Animated slider GIF - the divider sweeps across the image

Usage:
    python scripts/visualize_sahi_animated.py --model experiments/p2_baseline/weights/best.pt
    python scripts/visualize_sahi_animated.py --model experiments/p2_baseline/weights/best.pt --num_images 10
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
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
RESULTS_DIR = PROJECT_ROOT / "outputs" / "sahi_animations"

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


def draw_detections(image, predictions, label_text=None):
    """Draw bounding boxes on image."""
    img = image.copy()

    for pred in predictions:
        x1, y1, x2, y2 = [int(c) for c in pred['bbox']]
        color = CLASS_COLORS.get(pred['class_id'], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label
        class_name = CLASS_NAMES[pred['class_id']] if pred['class_id'] < len(CLASS_NAMES) else f"cls{pred['class_id']}"
        conf = pred['confidence']
        label = f"{class_name} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Add label text in corner
    if label_text:
        cv2.rectangle(img, (5, 5), (200, 35), (0, 0, 0), -1)
        cv2.putText(img, label_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img


def create_alternating_gif(img_std, img_sahi, output_path, duration=800):
    """Create GIF that alternates between standard and SAHI."""
    # Convert BGR to RGB for PIL
    frame_std = Image.fromarray(cv2.cvtColor(img_std, cv2.COLOR_BGR2RGB))
    frame_sahi = Image.fromarray(cv2.cvtColor(img_sahi, cv2.COLOR_BGR2RGB))

    frames = [frame_std, frame_sahi]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )


def create_slider_image(img_std, img_sahi, output_path, slider_pos=0.5):
    """Create image with vertical slider dividing standard (left) and SAHI (right)."""
    h, w = img_std.shape[:2]
    slider_x = int(w * slider_pos)

    # Combine images
    combined = img_std.copy()
    combined[:, slider_x:] = img_sahi[:, slider_x:]

    # Draw slider line
    cv2.line(combined, (slider_x, 0), (slider_x, h), (255, 255, 255), 3)
    cv2.line(combined, (slider_x, 0), (slider_x, h), (0, 0, 0), 1)

    # Labels
    cv2.rectangle(combined, (5, 5), (120, 35), (0, 0, 0), -1)
    cv2.putText(combined, "Standard", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(combined, (w - 90, 5), (w - 5, 35), (0, 0, 0), -1)
    cv2.putText(combined, "SAHI", (w - 80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Slider handle
    handle_y = h // 2
    cv2.circle(combined, (slider_x, handle_y), 15, (255, 255, 255), -1)
    cv2.circle(combined, (slider_x, handle_y), 15, (0, 0, 0), 2)
    cv2.arrowedLine(combined, (slider_x - 8, handle_y), (slider_x - 2, handle_y), (0, 0, 0), 2)
    cv2.arrowedLine(combined, (slider_x + 8, handle_y), (slider_x + 2, handle_y), (0, 0, 0), 2)

    cv2.imwrite(str(output_path), combined)


def create_sweeping_slider_gif(img_std, img_sahi, output_path, num_frames=30, duration=50):
    """Create animated GIF where slider sweeps across the image."""
    h, w = img_std.shape[:2]
    frames = []

    # Sweep from left to right, then right to left
    positions = list(np.linspace(0.1, 0.9, num_frames // 2))
    positions += positions[::-1]  # Reverse for back sweep

    for pos in positions:
        slider_x = int(w * pos)

        # Combine images
        combined = img_std.copy()
        combined[:, slider_x:] = img_sahi[:, slider_x:]

        # Draw slider line
        cv2.line(combined, (slider_x, 0), (slider_x, h), (255, 255, 255), 3)
        cv2.line(combined, (slider_x, 0), (slider_x, h), (0, 0, 0), 1)

        # Labels
        cv2.rectangle(combined, (5, 5), (120, 35), (0, 0, 0), -1)
        cv2.putText(combined, "Standard", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(combined, (w - 90, 5), (w - 5, 35), (0, 0, 0), -1)
        cv2.putText(combined, "SAHI", (w - 80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Slider handle
        handle_y = h // 2
        cv2.circle(combined, (slider_x, handle_y), 12, (255, 255, 255), -1)
        cv2.circle(combined, (slider_x, handle_y), 12, (0, 0, 0), 2)

        # Convert to PIL
        frame = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        frames.append(frame)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )


def create_detection_count_comparison(img, std_preds, sahi_preds, output_path):
    """Create image highlighting what SAHI found that standard missed."""
    h, w = img.shape[:2]
    result = img.copy()

    # Find SAHI-only detections (IoU matching)
    sahi_only = []
    for sahi_pred in sahi_preds:
        matched = False
        for std_pred in std_preds:
            if std_pred['class_id'] != sahi_pred['class_id']:
                continue
            # Simple IoU check
            x1 = max(std_pred['bbox'][0], sahi_pred['bbox'][0])
            y1 = max(std_pred['bbox'][1], sahi_pred['bbox'][1])
            x2 = min(std_pred['bbox'][2], sahi_pred['bbox'][2])
            y2 = min(std_pred['bbox'][3], sahi_pred['bbox'][3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area1 = (std_pred['bbox'][2]-std_pred['bbox'][0]) * (std_pred['bbox'][3]-std_pred['bbox'][1])
            area2 = (sahi_pred['bbox'][2]-sahi_pred['bbox'][0]) * (sahi_pred['bbox'][3]-sahi_pred['bbox'][1])
            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
            if iou > 0.5:
                matched = True
                break
        if not matched:
            sahi_only.append(sahi_pred)

    # Draw all SAHI detections in green (dim)
    for pred in sahi_preds:
        x1, y1, x2, y2 = [int(c) for c in pred['bbox']]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 150, 0), 1)

    # Highlight SAHI-only detections in bright red with "NEW" label
    for pred in sahi_only:
        x1, y1, x2, y2 = [int(c) for c in pred['bbox']]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(result, "NEW!", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Stats box
    cv2.rectangle(result, (5, 5), (250, 70), (0, 0, 0), -1)
    cv2.putText(result, f"Standard: {len(std_preds)} detections", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(result, f"SAHI: {len(sahi_preds)} detections", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(result, f"SAHI found +{len(sahi_only)} extra!", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imwrite(str(output_path), result)
    return len(sahi_only)


def run_inference(model_path, img_path, conf=0.3, slice_size=512, overlap=0.2):
    """Run both standard and SAHI inference."""
    # Standard inference
    model = YOLO(str(model_path))
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
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=conf,
        device='cuda:0'
    )

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

    return standard_preds, sahi_preds


def main():
    parser = argparse.ArgumentParser(description="Create animated SAHI comparisons")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to process")
    parser.add_argument("--slice_size", type=int, default=512, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")

    args = parser.parse_args()

    if not SAHI_AVAILABLE:
        print("ERROR: SAHI not installed. Run: pip install sahi")
        return

    model_path = Path(args.model)
    model_name = model_path.parent.parent.name

    output_dir = RESULTS_DIR / f"{model_name}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANIMATED SAHI COMPARISON GENERATOR")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Creating: alternating GIFs, slider images, sweeping GIFs")
    print("=" * 70)

    # Get images
    data_dir = PROJECT_ROOT / "data" / args.split / "images" / "rgb"
    image_files = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))

    if len(image_files) == 0:
        print(f"ERROR: No images found in {data_dir}")
        return

    # Find images with most difference first
    print(f"\nScanning {min(50, len(image_files))} images to find best examples...")

    candidates = []
    for i, img_path in enumerate(image_files[:50]):
        std_preds, sahi_preds = run_inference(
            model_path, img_path, args.conf, args.slice_size, args.overlap
        )
        diff = len(sahi_preds) - len(std_preds)
        candidates.append((img_path, std_preds, sahi_preds, diff))
        print(f"  [{i+1}/50] {img_path.name}: Std={len(std_preds)}, SAHI={len(sahi_preds)}, Diff={diff:+d}")

    # Sort by difference (most SAHI improvement first)
    candidates.sort(key=lambda x: x[3], reverse=True)

    print(f"\nGenerating visualizations for top {args.num_images} images...")

    for i, (img_path, std_preds, sahi_preds, diff) in enumerate(candidates[:args.num_images]):
        print(f"\n[{i+1}/{args.num_images}] Processing {img_path.name} (SAHI found +{diff} more)")

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Draw detections
        img_std = draw_detections(img, std_preds, f"Standard ({len(std_preds)})")
        img_sahi = draw_detections(img, sahi_preds, f"SAHI ({len(sahi_preds)})")

        base_name = img_path.stem

        # 1. Alternating GIF
        gif_path = output_dir / f"{i+1:02d}_{base_name}_alternating.gif"
        create_alternating_gif(img_std, img_sahi, gif_path)
        print(f"    Created: {gif_path.name}")

        # 2. Slider image (50% split)
        slider_path = output_dir / f"{i+1:02d}_{base_name}_slider.jpg"
        create_slider_image(img_std, img_sahi, slider_path)
        print(f"    Created: {slider_path.name}")

        # 3. Sweeping slider GIF
        sweep_path = output_dir / f"{i+1:02d}_{base_name}_sweep.gif"
        create_sweeping_slider_gif(img_std, img_sahi, sweep_path)
        print(f"    Created: {sweep_path.name}")

        # 4. Detection highlight image
        highlight_path = output_dir / f"{i+1:02d}_{base_name}_highlight.jpg"
        extra = create_detection_count_comparison(img, std_preds, sahi_preds, highlight_path)
        print(f"    Created: {highlight_path.name} (+{extra} SAHI-only detections)")

    print(f"\n" + "=" * 70)
    print(f"DONE! All visualizations saved to: {output_dir}")
    print("=" * 70)
    print("\nFiles created per image:")
    print("  - *_alternating.gif : Flips between Standard and SAHI")
    print("  - *_slider.jpg      : Side-by-side with vertical divider")
    print("  - *_sweep.gif       : Animated slider sweeping across")
    print("  - *_highlight.jpg   : Shows SAHI-only detections in red")


if __name__ == "__main__":
    main()
