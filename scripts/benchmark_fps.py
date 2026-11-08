"""
Benchmark FPS for Different Inference Schemes
==============================================
Measures inference speed (FPS) for:
- Naive baseline (standard YOLOv8n)
- P2 baseline (YOLOv8n with P2 head)
- P2 baseline + SAHI (sliced inference)

Usage:
    python scripts/benchmark_fps.py
    python scripts/benchmark_fps.py --num_images 100 --warmup 10
"""

import argparse
import time
from datetime import datetime
import numpy as np
from pathlib import Path
from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("WARNING: SAHI not installed. SAHI benchmarks will be skipped.")

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent

MODELS = {
    "naive_baseline": PROJECT_ROOT / "experiments" / "naive_baseline" / "weights" / "best.pt",
    "p2_baseline": PROJECT_ROOT / "experiments" / "p2_baseline" / "weights" / "best.pt",
}

# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_standard_inference(model_path, image_files, warmup=10, conf=0.3):
    """Benchmark standard YOLO inference."""
    print(f"    Loading model...")
    model = YOLO(str(model_path))
    print(f"    Model loaded.")

    # Warmup runs (not counted)
    print(f"    Warming up ({warmup} images)...", end="", flush=True)
    for i, img_path in enumerate(image_files[:warmup]):
        _ = model.predict(str(img_path), conf=conf, verbose=False)
        print(".", end="", flush=True)
    print(" done")

    # Timed runs
    print(f"    Benchmarking ({len(image_files)} images)...", end="", flush=True)
    times = []
    for i, img_path in enumerate(image_files):
        start = time.perf_counter()
        _ = model.predict(str(img_path), conf=conf, verbose=False)
        end = time.perf_counter()
        times.append(end - start)
        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)
    print(" done")

    return times


def benchmark_sahi_inference(model_path, image_files, warmup=10, conf=0.3,
                              slice_size=512, overlap=0.2):
    """Benchmark SAHI sliced inference."""
    if not SAHI_AVAILABLE:
        return None

    print(f"    Loading SAHI model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=conf,
        device='cuda:0'
    )
    print(f"    SAHI model loaded.")

    # Warmup runs
    print(f"    Warming up ({warmup} images)...", end="", flush=True)
    for i, img_path in enumerate(image_files[:warmup]):
        _ = get_sliced_prediction(
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
        print(".", end="", flush=True)
    print(" done")

    # Timed runs
    print(f"    Benchmarking ({len(image_files)} images)...", end="", flush=True)
    times = []
    for i, img_path in enumerate(image_files):
        start = time.perf_counter()
        _ = get_sliced_prediction(
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
        end = time.perf_counter()
        times.append(end - start)
        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)
    print(" done")

    return times


def calculate_stats(times):
    """Calculate FPS statistics from timing data."""
    times = np.array(times)
    fps = 1.0 / times

    return {
        'mean_fps': np.mean(fps),
        'std_fps': np.std(fps),
        'min_fps': np.min(fps),
        'max_fps': np.max(fps),
        'mean_latency_ms': np.mean(times) * 1000,
        'std_latency_ms': np.std(times) * 1000,
    }


def generate_report(results, args, num_images):
    """Generate report text."""
    lines = []

    lines.append("=" * 80)
    lines.append("FPS BENCHMARK RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("CONFIGURATION:")
    lines.append(f"  - Dataset split: {args.split}")
    lines.append(f"  - Number of images: {num_images}")
    lines.append(f"  - Warmup images: {args.warmup}")
    lines.append(f"  - Confidence threshold: {args.conf}")
    lines.append(f"  - SAHI slice size: {args.slice_size}x{args.slice_size}")
    lines.append(f"  - SAHI overlap: {args.overlap}")
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"{'Model':<30} {'FPS (mean±std)':<20} {'Latency (ms)':<20} {'Speedup':<10}")
    lines.append("-" * 80)

    # Find baseline FPS for speedup calculation
    baseline_fps = None
    for name, stats in results.items():
        if stats is not None and 'naive' in name.lower() and 'sahi' not in name.lower():
            baseline_fps = stats['mean_fps']
            break

    for name, stats in results.items():
        if stats is None:
            lines.append(f"{name:<30} {'N/A':<20} {'N/A':<20} {'N/A':<10}")
            continue

        fps_str = f"{stats['mean_fps']:.1f} ± {stats['std_fps']:.1f}"
        latency_str = f"{stats['mean_latency_ms']:.1f} ± {stats['std_latency_ms']:.1f}"

        if baseline_fps:
            speedup = stats['mean_fps'] / baseline_fps
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "1.00x"

        lines.append(f"{name:<30} {fps_str:<20} {latency_str:<20} {speedup_str:<10}")

    lines.append("=" * 80)

    # Insights
    lines.append("")
    lines.append("INSIGHTS:")
    if 'p2_baseline' in results and 'naive_baseline' in results:
        overhead = (results['naive_baseline']['mean_fps'] - results['p2_baseline']['mean_fps']) / results['naive_baseline']['mean_fps'] * 100
        lines.append(f"  - P2 head overhead: {overhead:.1f}% slower than naive baseline")

    if 'p2_baseline + SAHI' in results and 'p2_baseline' in results:
        overhead = (results['p2_baseline']['mean_fps'] - results['p2_baseline + SAHI']['mean_fps']) / results['p2_baseline']['mean_fps'] * 100
        lines.append(f"  - SAHI overhead: {overhead:.1f}% slower than standard P2 inference")

    # Explanation
    lines.append("")
    lines.append("=" * 80)
    lines.append("METRIC EXPLANATIONS:")
    lines.append("=" * 80)
    lines.append("")
    lines.append("FPS (Frames Per Second):")
    lines.append("  How many images the model can process per second. Higher is better.")
    lines.append("  Real-time video (30 FPS) requires the model to process each frame in <33ms.")
    lines.append("")
    lines.append("Latency (ms):")
    lines.append("  Time to process a single image in milliseconds. Lower is better.")
    lines.append("  This is the inverse of FPS: Latency = 1000 / FPS")
    lines.append("")
    lines.append("Speedup:")
    lines.append("  Relative speed compared to naive_baseline (standard YOLOv8n).")
    lines.append("  Values <1.0x indicate slower inference, >1.0x indicate faster.")
    lines.append("")
    lines.append("Models tested:")
    lines.append("  - naive_baseline: Standard YOLOv8n with default detection heads (P3, P4, P5)")
    lines.append("  - p2_baseline: YOLOv8n with additional P2 detection head (stride 4)")
    lines.append("    The P2 head adds a higher resolution feature map (160x160 for 640x640 input)")
    lines.append("    for better small object detection, at the cost of some speed.")
    lines.append("  - p2_baseline + SAHI: P2 model with Slicing Aided Hyper Inference.")
    lines.append("    SAHI splits the image into overlapping tiles, runs inference on each,")
    lines.append("    and merges results. This improves small object detection but is slower.")
    lines.append("")

    return "\n".join(lines)


def print_results_table(results, args, num_images):
    """Print results in a nice table format."""
    report = generate_report(results, args, num_images)
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS for different inference schemes")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup images")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--slice_size", type=int, default=512, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")

    args = parser.parse_args()

    # Get images
    data_dir = PROJECT_ROOT / "data" / args.split / "images" / "rgb"
    image_files = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))
    image_files = image_files[:args.num_images]

    if len(image_files) == 0:
        print(f"ERROR: No images found in {data_dir}")
        return

    print("=" * 80)
    print("FPS BENCHMARK")
    print("=" * 80)
    print(f"Images: {len(image_files)} from {args.split} split")
    print(f"Warmup: {args.warmup} images")
    print(f"Confidence: {args.conf}")
    print(f"SAHI slice size: {args.slice_size}")
    print(f"SAHI overlap: {args.overlap}")
    print("=" * 80)

    results = {}

    # Benchmark each model
    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"\n[SKIP] {model_name}: Model not found at {model_path}")
            continue

        # Standard inference
        print(f"\n[{model_name}] Standard inference...")
        times = benchmark_standard_inference(model_path, image_files, args.warmup, args.conf)
        results[model_name] = calculate_stats(times)
        print(f"    Done: {results[model_name]['mean_fps']:.1f} FPS")

        # SAHI inference (only for p2_baseline to show the overhead)
        if 'p2' in model_name and SAHI_AVAILABLE:
            print(f"\n[{model_name} + SAHI] Sliced inference...")
            times = benchmark_sahi_inference(
                model_path, image_files, args.warmup, args.conf,
                args.slice_size, args.overlap
            )
            if times:
                results[f"{model_name} + SAHI"] = calculate_stats(times)
                print(f"    Done: {results[f'{model_name} + SAHI']['mean_fps']:.1f} FPS")

    # Print and save results
    report = print_results_table(results, args, len(image_files))

    # Save report
    output_dir = PROJECT_ROOT / "outputs" / "fps_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"fps_benchmark_{args.split}_{timestamp}.txt"

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
