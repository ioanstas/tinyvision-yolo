"""
P2 + Hybrid CIoU+NWD Training Script
=====================================
Experiment 4b: P2 detection head + hybrid CIoU+NWD loss.

Combines CIoU (stable for medium/large) with NWD (fair for tiny objects).

Run from project root:
    python scripts/p2_hybrid_nwd.py
"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ultralytics import YOLO
from utils.hybrid_nwd_loss import patch_ultralytics_hybrid_loss

# =============================================================================
# Configuration
# =============================================================================
CONFIG_PATH = PROJECT_ROOT / "configs" / "model" / "dataset.yaml"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model" / "yolov8n-p2.yaml"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

# =============================================================================
# Device Setup
# =============================================================================
def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("WARNING: No GPU found, using CPU")
    return device


# =============================================================================
# Hyperparameters
# =============================================================================
HYPERPARAMS = {
    "epochs": 500,
    "patience": 20,
    "batch": 8,
    "imgsz": 640,

    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,

    "dropout": 0.1,

    "augment": True,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.1,

    "val": True,
    "plots": True,
    "save_period": 10,
    "workers": 2,
    "amp": True,
}

# Hybrid loss settings
CIOU_WEIGHT = 0.5
NWD_WEIGHT = 0.5
NWD_CONSTANT = 42.0


# =============================================================================
# Training
# =============================================================================
def train():
    """Run P2 + Hybrid CIoU+NWD training."""
    print("=" * 60)
    print("EXPERIMENT 4b: P2 + Hybrid CIoU+NWD Loss")
    print("=" * 60)
    print(f"\nLoss config: CIoU×{CIOU_WEIGHT} + NWD×{NWD_WEIGHT} (C={NWD_CONSTANT})")

    if not CONFIG_PATH.exists():
        print(f"ERROR: Dataset config not found at {CONFIG_PATH}")
        return None

    if not MODEL_CONFIG.exists():
        print(f"ERROR: Model config not found at {MODEL_CONFIG}")
        return None

    device = get_device()

    # Patch loss BEFORE creating model
    print("\nPatching Ultralytics with Hybrid CIoU+NWD loss...")
    success = patch_ultralytics_hybrid_loss(
        ciou_weight=CIOU_WEIGHT,
        nwd_weight=NWD_WEIGHT,
        constant=NWD_CONSTANT
    )

    if not success:
        print("WARNING: Hybrid patch failed. Continuing with default loss...")

    # Build model
    print(f"\nBuilding YOLOv8n-P2...")
    model = YOLO(str(MODEL_CONFIG))

    # Start training
    print(f"\nStarting training for {HYPERPARAMS['epochs']} epochs...")
    print(f"Output: {EXPERIMENT_DIR}/p2_hybrid_nwd/")
    print("-" * 60)

    results = model.train(
        data=str(CONFIG_PATH),
        device=device,
        project=str(EXPERIMENT_DIR),
        name="p2_hybrid_nwd",
        exist_ok=True,
        pretrained=True,

        **HYPERPARAMS
    )

    return model, results


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(model):
    """Evaluate trained model."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    print("\nValidating on val set...")
    val_metrics = model.val(data=str(CONFIG_PATH), split="val")

    print("\n" + "-" * 40)
    print("VALIDATION RESULTS")
    print("-" * 40)
    print(f"mAP50:      {val_metrics.box.map50:.4f}")
    print(f"mAP50-95:   {val_metrics.box.map:.4f}")
    print(f"Precision:  {val_metrics.box.mp:.4f}")
    print(f"Recall:     {val_metrics.box.mr:.4f}")

    print("\n" + "-" * 40)
    print("PER-CLASS AP50")
    print("-" * 40)
    for i, ap in enumerate(val_metrics.box.ap50):
        print(f"  {model.names[i]:<12}: {ap:.4f}")

    print("\n" + "-" * 40)
    print("TEST SET RESULTS")
    print("-" * 40)
    test_metrics = model.val(data=str(CONFIG_PATH), split="test")
    print(f"mAP50:      {test_metrics.box.map50:.4f}")
    print(f"mAP50-95:   {test_metrics.box.map:.4f}")

    return val_metrics, test_metrics


# =============================================================================
# Main
# =============================================================================
def main():
    result = train()

    if result is None:
        return

    model, train_results = result
    val_metrics, test_metrics = evaluate(model)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest weights: {EXPERIMENT_DIR}/p2_hybrid_nwd/weights/best.pt")

    print("\n" + "-" * 40)
    print("P2 + HYBRID CIoU+NWD METRICS")
    print("-" * 40)
    print(f"Val  mAP50: {val_metrics.box.map50:.4f}")
    print(f"Test mAP50: {test_metrics.box.map50:.4f}")

    print("\nCompare all experiments:")
    print("  python scripts/evaluate_by_size.py --model experiments/p2_baseline/weights/best.pt")
    print("  python scripts/evaluate_by_size.py --model experiments/p2_nwd_baseline/weights/best.pt")
    print("  python scripts/evaluate_by_size.py --model experiments/p2_hybrid_nwd/weights/best.pt")


if __name__ == "__main__":
    main()
