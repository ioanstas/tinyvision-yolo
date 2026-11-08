"""
P2 Baseline Training Script - YOLOv8n with P2 Head
===================================================
Experiment 2: Add P2 detection head for better tiny object detection.

The P2 head operates at stride 4 (vs stride 8 for P3), giving 4x more
spatial resolution for detecting objects < 16px.

Run from project root:
    python scripts/p2_baseline.py
"""

import torch
from pathlib import Path
from ultralytics import YOLO

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
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
        print("WARNING: No GPU found, using CPU (training will be slow)")
    return device


# =============================================================================
# Hyperparameters
# =============================================================================
HYPERPARAMS = {
    # Training
    "epochs": 500,
    "patience": 20,
    "batch": 8,              # Reduced due to larger model (P2 adds memory)
    "imgsz": 640,

    # Optimizer
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Regularization
    "dropout": 0.1,

    # Augmentation
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

    # Other
    "val": True,
    "plots": True,
    "save_period": 10,
    "workers": 2,
    "amp": True,
}


# =============================================================================
# Training
# =============================================================================
def train():
    """Run P2 baseline training."""
    print("=" * 60)
    print("EXPERIMENT 2: P2 Baseline (YOLOv8n + P2 Head)")
    print("=" * 60)

    # Verify configs exist
    if not CONFIG_PATH.exists():
        print(f"ERROR: Dataset config not found at {CONFIG_PATH}")
        print("Run 'python scripts/prepare_data.py' first.")
        return None

    if not MODEL_CONFIG.exists():
        print(f"ERROR: Model config not found at {MODEL_CONFIG}")
        return None

    device = get_device()

    # Build model from custom P2 config
    print(f"\nBuilding YOLOv8n-P2 from: {MODEL_CONFIG}")
    model = YOLO(str(MODEL_CONFIG))

    # Print model info
    print("\nModel Architecture:")
    print(f"  Detection heads: P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)")
    print(f"  P2 feature map: 160x160 (for 640 input)")
    print(f"  Expected improvement: Better detection of objects < 16px")

    # Start training
    print(f"\nStarting training for {HYPERPARAMS['epochs']} epochs...")
    print(f"Config: {CONFIG_PATH}")
    print(f"Output: {EXPERIMENT_DIR}/p2_baseline/")
    print("-" * 60)

    results = model.train(
        data=str(CONFIG_PATH),
        device=device,
        project=str(EXPERIMENT_DIR),
        name="p2_baseline",
        exist_ok=True,
        pretrained=True,  # Use pretrained backbone weights where possible

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

    # Validate
    print("\nValidating on val set...")
    val_metrics = model.val(data=str(CONFIG_PATH), split="val")

    print("\n" + "-" * 40)
    print("VALIDATION RESULTS")
    print("-" * 40)
    print(f"mAP50:      {val_metrics.box.map50:.4f}")
    print(f"mAP50-95:   {val_metrics.box.map:.4f}")
    print(f"Precision:  {val_metrics.box.mp:.4f}")
    print(f"Recall:     {val_metrics.box.mr:.4f}")

    # Per-class
    print("\n" + "-" * 40)
    print("PER-CLASS AP50")
    print("-" * 40)
    class_names = model.names
    for i, ap in enumerate(val_metrics.box.ap50):
        print(f"  {class_names[i]:<12}: {ap:.4f}")

    # Test set
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

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest weights: {EXPERIMENT_DIR}/p2_baseline/weights/best.pt")

    print("\n" + "-" * 40)
    print("P2 BASELINE METRICS")
    print("-" * 40)
    print(f"Val  mAP50: {val_metrics.box.map50:.4f}")
    print(f"Test mAP50: {test_metrics.box.map50:.4f}")

    print("\nNext: Run evaluate_by_size.py to compare tiny object performance:")
    print("  python scripts/evaluate_by_size.py --model experiments/p2_baseline/weights/best.pt")


if __name__ == "__main__":
    main()
