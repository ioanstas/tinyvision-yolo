"""
Data Preparation Script for RGBT-Tiny Dataset
==============================================
Extracts, processes, and organizes the RGBT-Tiny dataset for YOLO training.

Run from project root:
    python scripts/prepare_data.py
"""

import os
import json
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ZIPPED_DIR = DATA_DIR / "zipped"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# RGBT-Tiny class mapping
CLASSES = {
    0: "ship",
    1: "car",
    2: "cyclist",
    3: "pedestrian",
    4: "bus",
    5: "drone",
    6: "plane"
}

# Validation split ratio (from training set, since dataset only has train/test)
VAL_RATIO = 0.15


def unzip_all():
    """
    Step 1: Extract all zip files.
    - Extracts the main zips to raw/
    - Extracts nested zips (annotations, splits)
    """
    print("=" * 60)
    print("STEP 1: Extracting zip files")
    print("=" * 60)

    # Extract main zip files
    for zip_file in ZIPPED_DIR.glob("*.zip"):
        print(f"Extracting: {zip_file.name}")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(RAW_DIR)

    # Extract nested zips in RGBT-Tiny folder
    rgbt_dir = RAW_DIR / "RGBT-Tiny"
    if rgbt_dir.exists():
        for nested_zip in rgbt_dir.glob("*.zip"):
            print(f"Extracting nested: {nested_zip.name}")
            with zipfile.ZipFile(nested_zip, 'r') as zf:
                zf.extractall(rgbt_dir)

    print("Done!\n")


def load_coco_annotations(json_path):
    """
    Load COCO format annotations and organize by image_id.
    Returns: (images_dict, annotations_by_image)
    """
    with open(json_path, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}

    annotations_by_image = defaultdict(list)
    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    return images, annotations_by_image


def coco_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format.
    YOLO format: [x_center, y_center, width, height] normalized to [0, 1]
    """
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Clamp values to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    w_norm = max(0, min(1, w_norm))
    h_norm = max(0, min(1, h_norm))

    return x_center, y_center, w_norm, h_norm


def convert_annotations_to_yolo():
    """
    Step 2: Convert COCO annotations to YOLO format.
    Creates .txt files in processed/annotations/
    """
    print("=" * 60)
    print("STEP 2: Converting annotations to YOLO format")
    print("=" * 60)

    rgbt_dir = RAW_DIR / "RGBT-Tiny"
    annotations_dir = PROCESSED_DIR / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Process both RGB (00) and thermal (01) annotations
    # We'll use RGB annotations as the primary (they should be identical)
    coco_files = [
        rgbt_dir / "instances_00_train2017.json",
        rgbt_dir / "instances_00_test2017.json"
    ]

    total_annotations = 0

    for coco_file in coco_files:
        if not coco_file.exists():
            print(f"Warning: {coco_file} not found, skipping...")
            continue

        print(f"Processing: {coco_file.name}")
        images, annotations_by_image = load_coco_annotations(coco_file)

        for img_id, img_info in tqdm(images.items(), desc="Converting"):
            # Get annotations for this image
            anns = annotations_by_image.get(img_id, [])

            # Create output filename (flatten path structure)
            # DJI_0022_1/00/00000.jpg -> DJI_0022_1_00_00000.txt
            file_name = img_info['file_name']
            txt_name = file_name.replace('/', '_').replace('.jpg', '.txt')
            txt_path = annotations_dir / txt_name

            # Write YOLO format annotations
            with open(txt_path, 'w') as f:
                for ann in anns:
                    if ann.get('ignore', 0) == 1:
                        continue

                    class_id = ann['category_id']
                    bbox = ann['bbox']

                    x_center, y_center, w, h = coco_to_yolo(
                        bbox, img_info['width'], img_info['height']
                    )

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    total_annotations += 1

    print(f"Created {len(list(annotations_dir.glob('*.txt')))} annotation files")
    print(f"Total annotations: {total_annotations}\n")


def organize_images():
    """
    Step 3: Copy and organize images into processed/rgb and processed/thermal.
    Flattens the directory structure for easier handling.
    """
    print("=" * 60)
    print("STEP 3: Organizing images")
    print("=" * 60)

    rgb_out = PROCESSED_DIR / "rgb"
    thermal_out = PROCESSED_DIR / "thermal"
    rgb_out.mkdir(parents=True, exist_ok=True)
    thermal_out.mkdir(parents=True, exist_ok=True)

    # Find all sequence directories
    image_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.startswith("DJI_")]

    rgb_count = 0
    thermal_count = 0

    for seq_dir in tqdm(image_dirs, desc="Processing sequences"):
        seq_name = seq_dir.name

        # Process RGB (00 subfolder)
        rgb_dir = seq_dir / "00"
        if rgb_dir.exists():
            for img in rgb_dir.glob("*.jpg"):
                # Flatten: DJI_0022_1/00/00000.jpg -> DJI_0022_1_00_00000.jpg
                new_name = f"{seq_name}_00_{img.name}"
                shutil.copy2(img, rgb_out / new_name)
                rgb_count += 1

        # Process thermal (01 subfolder)
        thermal_dir = seq_dir / "01"
        if thermal_dir.exists():
            for img in thermal_dir.glob("*.jpg"):
                new_name = f"{seq_name}_01_{img.name}"
                shutil.copy2(img, thermal_out / new_name)
                thermal_count += 1

    print(f"Copied {rgb_count} RGB images")
    print(f"Copied {thermal_count} thermal images\n")


def create_splits():
    """
    Step 4: Create train/val/test splits.
    Uses the dataset's official train/test split, creates val from train.
    """
    print("=" * 60)
    print("STEP 4: Creating dataset splits")
    print("=" * 60)

    rgbt_dir = RAW_DIR / "RGBT-Tiny"

    # Read official splits
    train_file = rgbt_dir / "train.txt"
    test_file = rgbt_dir / "test.txt"

    if not train_file.exists() or not test_file.exists():
        print("Error: Split files not found!")
        return

    # Parse split files (format: DJI_0022_1/00/00000)
    def read_split(filepath):
        with open(filepath, 'r') as f:
            # Convert path format to our flattened naming
            return [line.strip().replace('/', '_') for line in f if line.strip()]

    train_ids = read_split(train_file)
    test_ids = read_split(test_file)

    # Create validation split from training data
    import random
    random.seed(42)
    random.shuffle(train_ids)

    val_size = int(len(train_ids) * VAL_RATIO)
    val_ids = train_ids[:val_size]
    train_ids = train_ids[val_size:]

    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # Create split directories and copy/link files
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    for split_name, ids in splits.items():
        print(f"\nCreating {split_name} split...")

        split_dir = DATA_DIR / split_name
        (split_dir / "images" / "rgb").mkdir(parents=True, exist_ok=True)
        (split_dir / "images" / "thermal").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels" / "rgb").mkdir(parents=True, exist_ok=True)

        copied = 0
        for img_id in tqdm(ids, desc=f"  {split_name}"):
            # Source paths (RGB)
            rgb_src = PROCESSED_DIR / "rgb" / f"{img_id}.jpg"
            thermal_id = img_id.replace("_00_", "_01_")
            thermal_src = PROCESSED_DIR / "thermal" / f"{thermal_id}.jpg"
            label_src = PROCESSED_DIR / "annotations" / f"{img_id}.txt"

            # Destination paths
            rgb_dst = split_dir / "images" / "rgb" / f"{img_id}.jpg"
            thermal_dst = split_dir / "images" / "thermal" / f"{thermal_id}.jpg"
            label_dst = split_dir / "labels" / "rgb" / f"{img_id}.txt"

            # Copy files if they exist
            if rgb_src.exists():
                shutil.copy2(rgb_src, rgb_dst)
            if thermal_src.exists():
                shutil.copy2(thermal_src, thermal_dst)
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
                copied += 1

        print(f"  Copied {copied} samples")

    print()


def create_dataset_yaml():
    """
    Step 5: Create dataset.yaml configuration file for YOLO.
    """
    print("=" * 60)
    print("STEP 5: Creating dataset.yaml")
    print("=" * 60)

    yaml_content = f"""# RGBT-Tiny Dataset Configuration
# Auto-generated by prepare_data.py

path: {DATA_DIR.absolute()}

train: train/images/rgb
val: val/images/rgb
test: test/images/rgb

# Thermal image paths (for custom fusion dataloader)
train_thermal: train/images/thermal
val_thermal: val/images/thermal
test_thermal: test/images/thermal

# Class names
names:
  0: ship
  1: car
  2: cyclist
  3: pedestrian
  4: bus
  5: drone
  6: plane

nc: 7
"""

    yaml_path = PROJECT_ROOT / "configs" / "model" / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created: {yaml_path}\n")


def cleanup_intermediate():
    """
    Step 6 (Optional): Remove intermediate processed files to save space.
    Keeps only the final train/val/test splits.
    """
    print("=" * 60)
    print("STEP 6: Cleanup (optional)")
    print("=" * 60)

    # Uncomment to enable cleanup
    # shutil.rmtree(PROCESSED_DIR / "rgb", ignore_errors=True)
    # shutil.rmtree(PROCESSED_DIR / "thermal", ignore_errors=True)
    # shutil.rmtree(PROCESSED_DIR / "annotations", ignore_errors=True)
    # print("Cleaned up intermediate files")

    print("Skipped (uncomment in code to enable)\n")


def main():
    """Run the complete data preparation pipeline."""
    print("\n" + "=" * 60)
    print("RGBT-Tiny Data Preparation Pipeline")
    print("=" * 60 + "\n")

    # Run all steps
    unzip_all()
    convert_annotations_to_yolo()
    organize_images()
    create_splits()
    create_dataset_yaml()
    cleanup_intermediate()

    print("=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nFinal structure:")
    print(f"  {DATA_DIR}/train/  - Training data")
    print(f"  {DATA_DIR}/val/    - Validation data")
    print(f"  {DATA_DIR}/test/   - Test data")
    print(f"\nConfig file: configs/model/dataset.yaml")


if __name__ == "__main__":
    main()
