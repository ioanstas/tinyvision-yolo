"""
Naive Baseline Training Script - YOLOv8n on RGB Only
=====================================================
Experiment 1: Establish baseline performance using standard YOLOv8n.

Run from project root:
    python scripts/naive_baseline.py
"""

import torch
from pathlib import Path
from ultralytics import YOLO

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model" / "dataset.yaml"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

# =============================================================================
# Device Setup
# =============================================================================
def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        device = 0  # Use first GPU
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("WARNING: No GPU found, using CPU (training will be slow)")
    return device


# =============================================================================
# Hyperparameters
# =============================================================================
HYPERPARAMS = {
    # Training
    "epochs": 500,
    "patience": 20,          # Early stopping patience
    "batch": 8,             # Reduce to 8 if OOM
    "imgsz": 640,            # Standard YOLO input size

    # Optimizer
    "optimizer": "AdamW",    # AdamW works well for fine-tuning
    "lr0": 0.001,            # Initial learning rate
    "lrf": 0.01,             # Final LR = lr0 * lrf
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Regularization
    "dropout": 0.1,          # Dropout for overfitting prevention

    # Augmentation (tuned for small objects)
    "augment": True,
    "hsv_h": 0.015,          # Hue augmentation
    "hsv_s": 0.7,            # Saturation augmentation
    "hsv_v": 0.4,            # Value augmentation
    "degrees": 0.0,          # No rotation (can hurt small objects)
    "translate": 0.1,        # Translation
    "scale": 0.5,            # Scale augmentation
    "fliplr": 0.5,           # Horizontal flip
    "flipud": 0.0,           # No vertical flip
    "mosaic": 1.0,           # Mosaic augmentation
    "mixup": 0.1,            # Light mixup

    # Other
    "val": True,
    "plots": True,
    "save_period": 10,       # Save checkpoint every N epochs
    "workers": 2,            # DataLoader workers (reduce if RAM issues)
    "amp": True,             # Mixed precision training
}


# =============================================================================
# Training
# =============================================================================
def train():
    """Run baseline training."""
    print("=" * 60)
    print("EXPERIMENT 1: Naive Baseline (YOLOv8n RGB)")
    print("=" * 60)

    # Verify config exists
    if not CONFIG_PATH.exists():
        print(f"ERROR: Dataset config not found at {CONFIG_PATH}")
        print("Run 'python scripts/prepare_data.py' first.")
        return None

    # Setup device
    device = get_device()

    # Load pretrained model
    print("\nLoading pretrained YOLOv8n...")
    model = YOLO("yolov8n.pt")

    # Start training
    print(f"\nStarting training for {HYPERPARAMS['epochs']} epochs...")
    print(f"Config: {CONFIG_PATH}")
    print(f"Output: {EXPERIMENT_DIR}/naive_baseline/")
    print("-" * 60)

    results = model.train(
        data=str(CONFIG_PATH),
        device=device,
        project=str(EXPERIMENT_DIR),
        name="naive_baseline",
        exist_ok=True,  # Overwrite if exists

        # Unpack hyperparameters
        **HYPERPARAMS
    )

    return model, results


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(model):
    """Evaluate trained model and print results."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Validate on validation set
    print("\nValidating on val set...")
    val_metrics = model.val(data=str(CONFIG_PATH), split="val")

    # Print summary
    print("\n" + "-" * 40)
    print("VALIDATION RESULTS")
    print("-" * 40)
    print(f"mAP50:      {val_metrics.box.map50:.4f}")
    print(f"mAP50-95:   {val_metrics.box.map:.4f}")
    print(f"Precision:  {val_metrics.box.mp:.4f}")
    print(f"Recall:     {val_metrics.box.mr:.4f}")

    # Per-class results
    print("\n" + "-" * 40)
    print("PER-CLASS AP50")
    print("-" * 40)
    class_names = model.names
    for i, ap in enumerate(val_metrics.box.ap50):
        print(f"  {class_names[i]:<12}: {ap:.4f}")

    # Test set evaluation
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
    # Train
    result = train()

    if result is None:
        return

    model, train_results = result

    # Evaluate
    val_metrics, test_metrics = evaluate(model)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest weights: {EXPERIMENT_DIR}/naive_baseline/weights/best.pt")
    print(f"Training plots: {EXPERIMENT_DIR}/naive_baseline/")

    print("\n" + "-" * 40)
    print("BASELINE METRICS TO RECORD")
    print("-" * 40)
    print(f"Val  mAP50: {val_metrics.box.map50:.4f}")
    print(f"Test mAP50: {test_metrics.box.map50:.4f}")
    print("\nUse these as reference for future experiments.")


if __name__ == "__main__":
    main()
