"""CPU-only sanity checks; these do not validate detector accuracy."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import coco_to_yolo


class DataTests(unittest.TestCase):
    def test_box_conversion(self):
        self.assertEqual(coco_to_yolo([10, 20, 20, 40], 100, 200),
                         (0.2, 0.2, 0.2, 0.2))

    def test_full_image_box(self):
        self.assertEqual(coco_to_yolo([0, 0, 640, 512], 640, 512),
                         (0.5, 0.5, 1.0, 1.0))


class ArtifactTests(unittest.TestCase):
    def load(self, name):
        return json.loads((ROOT / "docs/results" / name).read_text())

    def test_baseline_weighted_scores(self):
        for name in ("naive-validation.json", "p2-validation.json"):
            metrics = self.load(name)
            bins = metrics["size_metrics"].values()
            score = sum(b["ap"] * b["total_gt"] for b in bins)
            count = sum(b["total_gt"] for b in bins)
            self.assertAlmostEqual(score / count, metrics["weighted_map50"])
            self.assertEqual(metrics["split"], "val")

    def test_sahi_comparison_uses_same_ground_truth_counts(self):
        metrics = self.load("sahi-validation.json")
        for size, values in metrics["standard_size_metrics"].items():
            self.assertEqual(values["total_gt"],
                             metrics["sahi_size_metrics"][size]["total_gt"])
        self.assertEqual(metrics["sahi_config"]["confidence_threshold"], 0.25)

    def test_media_present(self):
        for name in ("p2-detections.gif", "standard-vs-sahi.gif",
                     "sahi-size-comparison.png"):
            self.assertGreater((ROOT / "docs/assets" / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
