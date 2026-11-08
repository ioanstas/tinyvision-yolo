# Scripts Documentation

## Prerequisites

```bash
pip install tqdm matplotlib numpy scikit-learn
```

## Run Order

```bash
python scripts/prepare_data.py      # First - prepares the dataset
python scripts/explore_dataset.py   # Second - analyzes the dataset
```

---

## prepare_data.py - Data Preparation Pipeline

Extracts, processes, and organizes the RGBT-Tiny dataset for YOLO training.

### Pipeline Steps

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `unzip_all()` | Extracts both zip files + nested annotation/split zips |
| 2 | `convert_annotations_to_yolo()` | Converts COCO format `[x,y,w,h]` to YOLO format `[x_center, y_center, w, h]` normalized 0-1 |
| 3 | `organize_images()` | Flattens folder structure: `DJI_0022_1/00/00000.jpg` → `DJI_0022_1_00_00000.jpg` |
| 4 | `create_splits()` | Uses official train/test split, carves 15% of train for validation |
| 5 | `create_dataset_yaml()` | Generates YOLO config with paths and class names |
| 6 | `cleanup_intermediate()` | Optional cleanup of intermediate files (disabled by default) |

### Output Structure

```
data/
├── train/
│   ├── images/
│   │   ├── rgb/           # RGB training images
│   │   └── thermal/       # Matching thermal images
│   └── labels/            # YOLO format labels (.txt)
├── val/
│   ├── images/
│   │   ├── rgb/
│   │   └── thermal/
│   └── labels/
└── test/
    ├── images/
    │   ├── rgb/
    │   └── thermal/
    └── labels/
```

### YOLO Label Format

Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to [0, 1] relative to image dimensions.

### Classes

| ID | Class |
|----|-------|
| 0 | ship |
| 1 | car |
| 2 | cyclist |
| 3 | pedestrian |
| 4 | bus |
| 5 | drone |
| 6 | plane |

---

## explore_dataset.py - Dataset Analysis

Analyzes the prepared dataset and displays statistics, distributions, and visualizations.

### Analysis Features

| Feature | Description |
|---------|-------------|
| **Summary table** | Image counts per split (RGB, thermal, labels) |
| **Class distribution** | Count and percentage per class |
| **Bbox statistics** | Min/max/mean/median size, small/medium/large breakdown |
| **Objects per image** | Mean, median, max objects per image |
| **Visualizations** | Saves 2 PNG plots to `outputs/` |

### Output Files

```
outputs/
├── dataset_analysis.png   # 4-panel figure with main statistics
└── size_by_class.png      # Object size distribution per class
```

### Visualizations Included

1. **Class Distribution Bar Chart** - Number of objects per class
2. **Bbox Area Histogram** - Distribution of bounding box sizes
3. **Width vs Height Scatter** - Bounding box dimensions in pixels
4. **Train/Val/Test Pie Chart** - Split proportions

### Size Categories

Objects are categorized by normalized area:
- **Small**: area < 0.001 (roughly < 32x32 pixels in 640x512 image)
- **Medium**: 0.001 ≤ area < 0.01
- **Large**: area ≥ 0.01

---

## Troubleshooting

### Missing modules
```bash
pip install tqdm matplotlib numpy scikit-learn
```

### No data found
Make sure zip files are in `data/zipped/` before running `prepare_data.py`

### Visualization errors
Matplotlib requires a display or backend. On headless servers:
```python
import matplotlib
matplotlib.use('Agg')  # Add before importing pyplot
```
