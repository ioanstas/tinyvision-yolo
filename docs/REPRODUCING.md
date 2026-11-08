# Running the experiments

Run commands from the repository root. Python 3.10+ is the intended environment; the original code contains Python 3.10-era artifacts. The exact original environment was not recovered, so `requirements.txt` is a starting environment, not a reproducibility lockfile. NWD patches use Ultralytics internals and may require additional adaptation between versions.

## 1. Install

Create and activate a virtual environment. Install PyTorch/torchvision appropriate for your hardware, then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Training scripts select CUDA when available and otherwise CPU. Several evaluation and SAHI scripts explicitly use `cuda:0`; use a CUDA-enabled machine for that workflow. Do not assume Jetson compatibility or real-time performance from the desktop results.

## 2. Prepare RGBT-Tiny

Obtain the dataset from [its authors](https://github.com/XinyiYing/RGBT-Tiny), following their access and usage terms. Place the original dataset archives in `data/zipped/`. Only extract trusted archives.

```bash
python scripts/prepare_data.py
python scripts/explore_dataset.py
```

The preparation code expects `data/raw/RGBT-Tiny/` to contain `instances_00_train2017.json`, `instances_00_test2017.json`, `train.txt` and `test.txt`. Image sequence directories such as `DJI_0022_1/00/` and `DJI_0022_1/01/` must be directly under `data/raw/`. Check the extracted layout if the publisher's archive packaging changes.

It converts RGB COCO boxes to normalized YOLO labels, flattens image names, and reserves 15% of the official training list for validation using seed 42. It copies RGB and thermal images, but the headline detector uses RGB only. Thermal annotations are not generated for fusion training here.

Generated paths:

```text
data/train/images/rgb/   data/train/labels/rgb/
data/val/images/rgb/     data/val/labels/rgb/
data/test/images/rgb/    data/test/labels/rgb/
configs/model/dataset.yaml
```

`dataset.yaml` is generated with your local absolute data path and intentionally ignored by Git. Confirm all three splits contain images and matching labels before training. The exploratory plotting script has older size-bin definitions; use the evaluation scripts for the reported AP metrics.

## 3. Train the two models

```bash
python scripts/naive_baseline.py
python scripts/p2_baseline.py
```

Both scripts configure up to 500 epochs, patience 20, batch size 8, image size 640 and AdamW. They write to `experiments/naive_baseline/` and `experiments/p2_baseline/` and can reuse existing run directories; archive previous runs before repeating them.

The standard model starts from `yolov8n.pt`. The P2 script constructs a model from YAML and passes `pretrained=True`; this alone is not evidence that every compatible COCO weight was transferred. Inspect the actual training logs before treating initialization as a controlled variable. Pretrained assets may download automatically. Trained project checkpoints are not included in Git.

## 4. Evaluate

```bash
python scripts/evaluate_full.py --model experiments/naive_baseline/weights/best.pt --split val --conf 0.001
python scripts/evaluate_full.py --model experiments/p2_baseline/weights/best.pt --split val --conf 0.001
python scripts/evaluate_full_obss_sahi.py --model experiments/p2_baseline/weights/best.pt --split val --slice_size 512 --overlap 0.2 --conf 0.25 --postprocess_type GREEDYNMM --postprocess_metric IOU --postprocess_threshold 0.7
```

Use `evaluate_full_obss_sahi.py` for the official SAHI library comparison. Older `evaluate_sahi.py` and `evaluate_full_sahi.py` are retained as exploratory code, not interchangeable benchmark implementations. Save the generated metrics with their exact command, package versions and checkpoint hash.

## 5. Search and visualize

`sahi_grid_search_stage1.py` explores slice size/overlap, `stage1b` confidence, and `stage2` merging. Inspect their configuration before starting a long GPU run.

```bash
python scripts/visualize_sahi_animated.py --model experiments/p2_baseline/weights/best.pt --split val --num_images 5 --slice_size 512 --overlap 0.2 --conf 0.3
python scripts/benchmark_fps.py --split test --num_images 50 --warmup 10
```

Visualization and benchmark defaults differ from the selected accuracy experiment. Treat them as qualitative/timing tools, not direct reproductions of its metrics. The benchmark reports the mean of per-image FPS, not aggregate throughput.

## Reproduction status

Publication checks cover Python syntax, bounding-box conversion, saved metric consistency, and documentation assets. Full GPU training/inference has not been rerun for this portfolio release. For a stronger follow-up: use sequence-disjoint validation, pin the working environment, record hardware, synchronize GPU timing, and evaluate the selected configuration once on a held-out test set.
