"""
Dataset Exploration Script for RGBT-Tiny
========================================
Analyzes the dataset and displays statistics, distributions, and visualizations.

Run from project root:
    python scripts/explore_dataset.py
"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CLASSES = {
    0: "ship",
    1: "car",
    2: "cyclist",
    3: "pedestrian",
    4: "bus",
    5: "drone",
    6: "plane"
}

CLASS_COLORS = {
    0: "#1f77b4",  # ship - blue
    1: "#ff7f0e",  # car - orange
    2: "#2ca02c",  # cyclist - green
    3: "#d62728",  # pedestrian - red
    4: "#9467bd",  # bus - purple
    5: "#8c564b",  # drone - brown
    6: "#e377c2",  # plane - pink
}


def count_images(split_dir):
    """Count images in a split directory."""
    rgb_dir = split_dir / "images" / "rgb"
    thermal_dir = split_dir / "images" / "thermal"
    labels_dir = split_dir / "labels"

    rgb_count = len(list(rgb_dir.glob("*.jpg"))) if rgb_dir.exists() else 0
    thermal_count = len(list(thermal_dir.glob("*.jpg"))) if thermal_dir.exists() else 0
    labels_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

    return rgb_count, thermal_count, labels_count


def parse_yolo_labels(labels_dir):
    """
    Parse all YOLO format label files and extract statistics.
    Returns: list of (class_id, x_center, y_center, width, height) tuples
    """
    annotations = []
    images_with_objects = 0
    empty_images = 0

    if not labels_dir.exists():
        return annotations, 0, 0

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            empty_images += 1
            continue

        images_with_objects += 1
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append((class_id, x_center, y_center, width, height))

    return annotations, images_with_objects, empty_images


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def analyze_split(split_name, split_dir):
    """Analyze a single data split and return statistics."""
    rgb_count, thermal_count, labels_count = count_images(split_dir)
    annotations, imgs_with_obj, empty_imgs = parse_yolo_labels(split_dir / "labels")

    # Count by class
    class_counts = Counter(ann[0] for ann in annotations)

    # Collect bbox sizes
    bbox_sizes = [(ann[3], ann[4]) for ann in annotations]  # width, height
    bbox_areas = [w * h for w, h in bbox_sizes]

    return {
        'name': split_name,
        'rgb_count': rgb_count,
        'thermal_count': thermal_count,
        'labels_count': labels_count,
        'total_annotations': len(annotations),
        'images_with_objects': imgs_with_obj,
        'empty_images': empty_imgs,
        'class_counts': class_counts,
        'bbox_sizes': bbox_sizes,
        'bbox_areas': bbox_areas,
        'annotations': annotations
    }


def print_summary_table(stats_list):
    """Print a summary table of all splits."""
    print_header("DATASET SUMMARY")

    # Calculate totals
    total_rgb = sum(s['rgb_count'] for s in stats_list)
    total_thermal = sum(s['thermal_count'] for s in stats_list)
    total_labels = sum(s['labels_count'] for s in stats_list)
    total_annotations = sum(s['total_annotations'] for s in stats_list)

    print(f"\n{'Split':<10} {'RGB':>10} {'Thermal':>10} {'Labels':>10} {'Annotations':>12}")
    print("-" * 55)

    for stats in stats_list:
        print(f"{stats['name']:<10} {stats['rgb_count']:>10,} {stats['thermal_count']:>10,} "
              f"{stats['labels_count']:>10,} {stats['total_annotations']:>12,}")

    print("-" * 55)
    print(f"{'TOTAL':<10} {total_rgb:>10,} {total_thermal:>10,} "
          f"{total_labels:>10,} {total_annotations:>12,}")


def print_class_distribution(stats_list):
    """Print class distribution across all splits."""
    print_header("CLASS DISTRIBUTION")

    # Aggregate class counts
    total_by_class = Counter()
    for stats in stats_list:
        total_by_class.update(stats['class_counts'])

    total_objects = sum(total_by_class.values())

    print(f"\n{'Class':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 40)

    for class_id in sorted(total_by_class.keys()):
        class_name = CLASSES.get(class_id, f"class_{class_id}")
        count = total_by_class[class_id]
        pct = (count / total_objects) * 100 if total_objects > 0 else 0
        print(f"{class_name:<15} {count:>10,} {pct:>11.1f}%")

    print("-" * 40)
    print(f"{'TOTAL':<15} {total_objects:>10,} {'100.0%':>12}")

    return total_by_class


def print_bbox_statistics(stats_list):
    """Print bounding box size statistics."""
    print_header("BOUNDING BOX STATISTICS")

    # Combine all bbox data
    all_areas = []
    all_widths = []
    all_heights = []

    for stats in stats_list:
        all_areas.extend(stats['bbox_areas'])
        all_widths.extend([w for w, h in stats['bbox_sizes']])
        all_heights.extend([h for w, h in stats['bbox_sizes']])

    if not all_areas:
        print("\nNo bounding box data available.")
        return

    all_areas = np.array(all_areas)
    all_widths = np.array(all_widths)
    all_heights = np.array(all_heights)

    # Size categories (normalized values)
    # Small: area < 0.001 (roughly 32x32 in 640x512 = 0.003)
    # Medium: 0.001 <= area < 0.01
    # Large: area >= 0.01
    small = np.sum(all_areas < 0.001)
    medium = np.sum((all_areas >= 0.001) & (all_areas < 0.01))
    large = np.sum(all_areas >= 0.01)

    print(f"\nNormalized bbox statistics (values in 0-1 range):")
    print(f"\n{'Metric':<15} {'Width':>12} {'Height':>12} {'Area':>12}")
    print("-" * 55)
    print(f"{'Min':<15} {np.min(all_widths):>12.4f} {np.min(all_heights):>12.4f} {np.min(all_areas):>12.6f}")
    print(f"{'Max':<15} {np.max(all_widths):>12.4f} {np.max(all_heights):>12.4f} {np.max(all_areas):>12.6f}")
    print(f"{'Mean':<15} {np.mean(all_widths):>12.4f} {np.mean(all_heights):>12.4f} {np.mean(all_areas):>12.6f}")
    print(f"{'Median':<15} {np.median(all_widths):>12.4f} {np.median(all_heights):>12.4f} {np.median(all_areas):>12.6f}")
    print(f"{'Std':<15} {np.std(all_widths):>12.4f} {np.std(all_heights):>12.4f} {np.std(all_areas):>12.6f}")

    print(f"\nSize distribution:")
    total = len(all_areas)
    print(f"  Small  (area < 0.001):  {small:>8,} ({100*small/total:>5.1f}%)")
    print(f"  Medium (0.001-0.01):    {medium:>8,} ({100*medium/total:>5.1f}%)")
    print(f"  Large  (area >= 0.01):  {large:>8,} ({100*large/total:>5.1f}%)")

    # Convert to approximate pixel sizes (assuming 640x512)
    print(f"\nApproximate pixel sizes (640x512 image):")
    print(f"  Mean bbox:   {np.mean(all_widths)*640:.1f} x {np.mean(all_heights)*512:.1f} pixels")
    print(f"  Median bbox: {np.median(all_widths)*640:.1f} x {np.median(all_heights)*512:.1f} pixels")
    print(f"  Smallest:    {np.min(all_widths)*640:.1f} x {np.min(all_heights)*512:.1f} pixels")

    return all_widths, all_heights, all_areas


def print_objects_per_image(stats_list):
    """Analyze and print objects per image statistics."""
    print_header("OBJECTS PER IMAGE")

    for stats in stats_list:
        labels_dir = DATA_DIR / stats['name'] / "labels"
        if not labels_dir.exists():
            continue

        objects_per_image = []
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                count = len([l for l in f.readlines() if l.strip()])
                objects_per_image.append(count)

        if objects_per_image:
            objects_per_image = np.array(objects_per_image)
            print(f"\n{stats['name'].upper()}:")
            print(f"  Total images:     {len(objects_per_image):>8,}")
            print(f"  Empty images:     {np.sum(objects_per_image == 0):>8,}")
            print(f"  Mean objects:     {np.mean(objects_per_image):>8.1f}")
            print(f"  Median objects:   {np.median(objects_per_image):>8.0f}")
            print(f"  Max objects:      {np.max(objects_per_image):>8,}")


def create_visualizations(stats_list, class_counts):
    """Create and save visualization plots."""
    print_header("CREATING VISUALIZATIONS")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Combine all data
    all_areas = []
    all_widths = []
    all_heights = []
    for stats in stats_list:
        all_areas.extend(stats['bbox_areas'])
        all_widths.extend([w for w, h in stats['bbox_sizes']])
        all_heights.extend([h for w, h in stats['bbox_sizes']])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Class distribution bar chart
    ax1 = axes[0, 0]
    class_names = [CLASSES[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    colors = [CLASS_COLORS[i] for i in sorted(class_counts.keys())]
    bars = ax1.bar(class_names, counts, color=colors)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    ax1.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{count:,}', ha='center', va='bottom', fontsize=8)

    # 2. Bbox area histogram
    ax2 = axes[0, 1]
    ax2.hist(all_areas, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0.001, color='red', linestyle='--', label='Small/Medium threshold')
    ax2.axvline(x=0.01, color='orange', linestyle='--', label='Medium/Large threshold')
    ax2.set_xlabel('Normalized Bbox Area')
    ax2.set_ylabel('Count')
    ax2.set_title('Bounding Box Area Distribution')
    ax2.legend()
    ax2.set_xlim(0, 0.05)  # Focus on smaller objects

    # 3. Width vs Height scatter
    ax3 = axes[1, 0]
    sample_size = min(5000, len(all_widths))
    indices = np.random.choice(len(all_widths), sample_size, replace=False)
    sampled_w = np.array(all_widths)[indices]
    sampled_h = np.array(all_heights)[indices]
    ax3.scatter(sampled_w * 640, sampled_h * 512, alpha=0.3, s=10, c='steelblue')
    ax3.set_xlabel('Width (pixels)')
    ax3.set_ylabel('Height (pixels)')
    ax3.set_title('Bounding Box Dimensions (sampled)')
    ax3.set_xlim(0, 200)
    ax3.set_ylim(0, 200)
    ax3.axhline(y=32, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(x=32, color='red', linestyle='--', alpha=0.5, label='32px threshold')
    ax3.legend()

    # 4. Split distribution pie chart
    ax4 = axes[1, 1]
    split_names = [s['name'] for s in stats_list]
    split_counts = [s['rgb_count'] for s in stats_list]
    ax4.pie(split_counts, labels=split_names, autopct='%1.1f%%',
            colors=['#66b3ff', '#99ff99', '#ffcc99'])
    ax4.set_title('Train/Val/Test Split')

    plt.tight_layout()

    # Save figure
    fig_path = OUTPUTS_DIR / "dataset_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")

    # Additional plot: Class distribution by size
    fig2, ax = plt.subplots(figsize=(12, 6))

    # Group by class and size
    size_by_class = defaultdict(lambda: {'small': 0, 'medium': 0, 'large': 0})
    for stats in stats_list:
        for ann in stats['annotations']:
            class_id, _, _, w, h = ann
            area = w * h
            if area < 0.001:
                size_by_class[class_id]['small'] += 1
            elif area < 0.01:
                size_by_class[class_id]['medium'] += 1
            else:
                size_by_class[class_id]['large'] += 1

    x = np.arange(len(CLASSES))
    width = 0.25

    small_counts = [size_by_class[i]['small'] for i in range(len(CLASSES))]
    medium_counts = [size_by_class[i]['medium'] for i in range(len(CLASSES))]
    large_counts = [size_by_class[i]['large'] for i in range(len(CLASSES))]

    ax.bar(x - width, small_counts, width, label='Small', color='#ff6b6b')
    ax.bar(x, medium_counts, width, label='Medium', color='#4ecdc4')
    ax.bar(x + width, large_counts, width, label='Large', color='#45b7d1')

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Object Size Distribution by Class')
    ax.set_xticks(x)
    ax.set_xticklabels([CLASSES[i] for i in range(len(CLASSES))], rotation=45)
    ax.legend()

    plt.tight_layout()
    fig2_path = OUTPUTS_DIR / "size_by_class.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig2_path}")

    plt.close('all')


def main():
    """Run the complete dataset analysis."""
    print("\n" + "=" * 60)
    print(" RGBT-Tiny Dataset Analysis")
    print("=" * 60)

    # Check if data exists
    splits = ['train', 'val', 'test']
    stats_list = []

    for split in splits:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            stats = analyze_split(split, split_dir)
            stats_list.append(stats)
        else:
            print(f"Warning: {split_dir} not found")

    if not stats_list:
        print("\nNo data found! Run prepare_data.py first.")
        return

    # Print all statistics
    print_summary_table(stats_list)
    class_counts = print_class_distribution(stats_list)
    print_bbox_statistics(stats_list)
    print_objects_per_image(stats_list)

    # Create visualizations
    try:
        create_visualizations(stats_list, class_counts)
    except Exception as e:
        print(f"\nWarning: Could not create visualizations: {e}")
        print("Install matplotlib: pip install matplotlib")

    print_header("ANALYSIS COMPLETE")
    print(f"\nVisualization saved to: {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
