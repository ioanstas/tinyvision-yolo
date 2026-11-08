"""
Inference Visualization & Video Generation
============================================
Runs inference with 3 modes (naive_baseline, p2_baseline, p2_baseline+SAHI)
on the same images, draws bounding boxes, and optionally generates MP4 videos.

Usage:
    python scripts/visualize_inference.py --split test
    python scripts/visualize_inference.py --split test --max_frames 100
    python scripts/visualize_inference.py --split test --sequence DJI_0309_4
    python scripts/visualize_inference.py --split test --conf 0.25
    python scripts/visualize_inference.py --split test --video --fps 15
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

MODEL_CONFIGS = {
    "naive_baseline": {
        "weights": PROJECT_ROOT / "experiments" / "naive_baseline" / "weights" / "best.pt",
        "sahi": False,
    },
    "p2_baseline": {
        "weights": PROJECT_ROOT / "experiments" / "p2_baseline" / "weights" / "best.pt",
        "sahi": False,
    },
    "sahi_p2": {
        "weights": PROJECT_ROOT / "experiments" / "p2_baseline" / "weights" / "best.pt",
        "sahi": True,
    },
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"


def draw_detections(image, predictions, label_text=None):
    """Draw bounding boxes on image with class colors and confidence scores."""
    img = image.copy()

    for pred in predictions:
        x1, y1, x2, y2 = [int(c) for c in pred['bbox']]
        color = CLASS_COLORS.get(pred['class_id'], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        class_name = CLASS_NAMES[pred['class_id']] if pred['class_id'] < len(CLASS_NAMES) else f"cls{pred['class_id']}"
        conf = pred['confidence']
        label = f"{class_name} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if label_text:
        cv2.rectangle(img, (5, 5), (200, 35), (0, 0, 0), -1)
        cv2.putText(img, label_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img


def run_yolo_inference(model, img_path, conf):
    """Run standard YOLO inference on a single image."""
    results = model.predict(str(img_path), conf=conf, verbose=False)
    preds = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            preds.append({
                'class_id': int(box.cls[0].cpu().numpy()),
                'confidence': float(box.conf[0].cpu().numpy()),
                'bbox': xyxy.tolist()
            })
    return preds


def run_sahi_inference(detection_model, img_path, slice_size, overlap, conf):
    """Run SAHI sliced inference on a single image."""
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
        postprocess_class_agnostic=False,
        verbose=0
    )

    preds = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        preds.append({
            'class_id': pred.category.id,
            'confidence': pred.score.value,
            'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]]
        })
    return preds


P2_ONLY_COLOR = (128, 128, 255)  # light red in BGR


def find_p2_only(preds_left, preds_right, iou_thresh=0.5):
    """Return set of indices in preds_right that have no IoU match in preds_left."""
    unmatched = set(range(len(preds_right)))
    for j, rp in enumerate(preds_right):
        for lp in preds_left:
            if lp['class_id'] != rp['class_id']:
                continue
            x1 = max(lp['bbox'][0], rp['bbox'][0])
            y1 = max(lp['bbox'][1], rp['bbox'][1])
            x2 = min(lp['bbox'][2], rp['bbox'][2])
            y2 = min(lp['bbox'][3], rp['bbox'][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1 = (lp['bbox'][2] - lp['bbox'][0]) * (lp['bbox'][3] - lp['bbox'][1])
            a2 = (rp['bbox'][2] - rp['bbox'][0]) * (rp['bbox'][3] - rp['bbox'][1])
            union = a1 + a2 - inter
            if union > 0 and inter / union >= iou_thresh:
                unmatched.discard(j)
                break
    return unmatched


def draw_detections_highlight(image, predictions, highlight_indices, label_text=None):
    """Draw boxes; highlighted indices use light red, others use class colors."""
    img = image.copy()

    for idx, pred in enumerate(predictions):
        x1, y1, x2, y2 = [int(c) for c in pred['bbox']]
        is_new = idx in highlight_indices
        color = P2_ONLY_COLOR if is_new else CLASS_COLORS.get(pred['class_id'], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        class_name = CLASS_NAMES[pred['class_id']] if pred['class_id'] < len(CLASS_NAMES) else f"cls{pred['class_id']}"
        conf = pred['confidence']
        label = f"{class_name} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if label_text:
        cv2.rectangle(img, (5, 5), (200, 35), (0, 0, 0), -1)
        cv2.putText(img, label_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img


def draw_side_by_side(img, preds_left, preds_right, label_left, label_right):
    """Create a side-by-side image. P2-only detections drawn in light red."""
    p2_only_idx = find_p2_only(preds_left, preds_right)
    n_new = len(p2_only_idx)

    left = draw_detections(img, preds_left, f"{label_left} ({len(preds_left)})")
    right = draw_detections_highlight(
        img, preds_right, p2_only_idx,
        f"{label_right} ({len(preds_right)})"
    )
    h, w = img.shape[:2]

    combined = np.hstack([left, right])
    # Vertical divider
    cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)

    # Improvement badge (top center)
    if n_new > 0:
        badge = f"+{n_new} new (p2 only)"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bx = w - tw // 2
        cv2.rectangle(combined, (bx - 5, 5), (bx + tw + 5, th + 15), (0, 0, 180), -1)
        cv2.putText(combined, badge, (bx, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return combined


def make_video(frames_dir, video_path, fps=15):
    """Create MP4 video from a directory of frame images."""
    frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frame_files:
        print(f"  No frames found in {frames_dir}, skipping video.")
        return

    first = cv2.imread(str(frame_files[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for fp in frame_files:
        frame = cv2.imread(str(fp))
        if frame is not None:
            writer.write(frame)

    writer.release()
    print(f"  Video saved: {video_path} ({len(frame_files)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Visualize inference from 3 models")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--sequence", type=str, default=None, help="Filter to a DJI sequence prefix")
    parser.add_argument("--video", action="store_true", help="Generate MP4 videos from frames")
    parser.add_argument("--fps", type=int, default=15, help="Video FPS (default 15)")
    parser.add_argument("--slice_size", type=int, default=256, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    parser.add_argument("--top_k", type=int, default=50, help="Number of top p2-improvement images to save")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Which models to run (default: all)")
    args = parser.parse_args()

    # Collect image paths
    images_dir = PROJECT_ROOT / "data" / args.split / "images" / "rgb"
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    if not image_files:
        print(f"ERROR: No images found in {images_dir}")
        return

    # Filter by sequence if requested
    if args.sequence:
        image_files = [f for f in image_files if f.name.startswith(args.sequence)]
        if not image_files:
            print(f"ERROR: No images matching sequence '{args.sequence}' in {images_dir}")
            return

    # Limit frames
    if args.max_frames:
        image_files = image_files[:args.max_frames]

    print("=" * 70)
    print("INFERENCE VISUALIZATION")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Images: {len(image_files)}")
    print(f"Confidence: {args.conf}")
    if args.sequence:
        print(f"Sequence filter: {args.sequence}")
    active_configs = {k: v for k, v in MODEL_CONFIGS.items()
                      if args.models is None or k in args.models}
    print(f"Models: {', '.join(active_configs.keys())}")
    print("=" * 70)

    # Load models once
    print("\nLoading models...")
    models = {}
    sahi_model = None

    for name, cfg in active_configs.items():
        if not cfg["weights"].exists():
            print(f"  WARNING: Weights not found: {cfg['weights']}, skipping {name}")
            continue

        if cfg["sahi"]:
            if not SAHI_AVAILABLE:
                print(f"  WARNING: SAHI not available, skipping {name}")
                continue
            if sahi_model is None:
                sahi_model = AutoDetectionModel.from_pretrained(
                    model_type='yolov8',
                    model_path=str(cfg["weights"]),
                    confidence_threshold=args.conf,
                    device='cuda:0'
                )
                print(f"  Loaded SAHI model for {name}")
            models[name] = {"type": "sahi", "model": sahi_model}
        else:
            yolo = YOLO(str(cfg["weights"]))
            print(f"  Loaded YOLO model for {name}")
            models[name] = {"type": "yolo", "model": yolo}

    if not models:
        print("ERROR: No models loaded.")
        return

    # Create output directories
    out_dirs = {}
    for name in models:
        d = OUTPUT_DIR / name / args.split
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[name] = d

    # Process images — store predictions for comparison
    all_preds = {}  # {img_name: {model_name: preds_list}}
    print(f"\nProcessing {len(image_files)} images...")
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Could not read {img_path.name}, skipping.")
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  [{i + 1}/{len(image_files)}] {img_path.name}")

        img_preds = {}
        for name, m in models.items():
            if m["type"] == "yolo":
                preds = run_yolo_inference(m["model"], img_path, args.conf)
            else:
                preds = run_sahi_inference(
                    m["model"], img_path, args.slice_size, args.overlap, args.conf
                )

            img_preds[name] = preds
            annotated = draw_detections(img, preds, f"{name} ({len(preds)})")
            out_path = out_dirs[name] / img_path.name
            cv2.imwrite(str(out_path), annotated)

        all_preds[img_path.name] = img_preds

    print(f"\nFrames saved to {OUTPUT_DIR}")
    for name, d in out_dirs.items():
        count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
        print(f"  {name}/{args.split}: {count} frames")

    # =========================================================================
    # P2 Improvement picking: images where p2_baseline finds more than naive
    # =========================================================================
    if "naive_baseline" in models and "p2_baseline" in models:
        print("\n" + "=" * 70)
        print("P2 IMPROVEMENT ANALYSIS")
        print("=" * 70)

        improvements = []
        for img_name, preds_map in all_preds.items():
            n_naive = len(preds_map.get("naive_baseline", []))
            n_p2 = len(preds_map.get("p2_baseline", []))
            diff = n_p2 - n_naive
            if diff > 0:
                improvements.append((img_name, n_naive, n_p2, diff))

        # Sort by biggest improvement first
        improvements.sort(key=lambda x: x[3], reverse=True)

        top_k = min(args.top_k, len(improvements))
        print(f"Images where p2 found MORE detections: {len(improvements)} / {len(all_preds)}")
        print(f"Saving top {top_k} side-by-side comparisons...")

        improv_dir = OUTPUT_DIR / "p2_improvements" / args.split
        improv_dir.mkdir(parents=True, exist_ok=True)

        for rank, (img_name, n_naive, n_p2, diff) in enumerate(improvements[:top_k]):
            img_path = images_dir / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            combined = draw_side_by_side(
                img,
                all_preds[img_name]["naive_baseline"],
                all_preds[img_name]["p2_baseline"],
                "naive", "p2"
            )

            out_name = f"{rank + 1:03d}_{img_name}"
            cv2.imwrite(str(improv_dir / out_name), combined)

            if rank < 10:
                print(f"  #{rank + 1}: {img_name}  naive={n_naive}  p2={n_p2}  (+{diff})")

        print(f"\nSaved to: {improv_dir}")

        # Also save as video if requested
        if args.video:
            video_dir = OUTPUT_DIR / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"p2_improvements_{args.split}.mp4"
            make_video(improv_dir, video_path, fps=args.fps)

    # Generate per-model videos
    if args.video:
        print("\nGenerating per-model videos...")
        video_dir = OUTPUT_DIR / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        for name in models:
            video_path = video_dir / f"{name}_{args.split}.mp4"
            make_video(out_dirs[name], video_path, fps=args.fps)

    print("\nDone!")


if __name__ == "__main__":
    main()
