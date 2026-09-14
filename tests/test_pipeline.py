"""Small synthetic checks; no study records or images are included."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from retina_study.data import Record, load_manifest, select_patient_fraction, validate_records
from retina_study.metrics import binary_metrics, borderline_scenarios
from retina_study.model import load_image
from retina_study.prepare import classify_glaucoma, prepare_manifest


class PipelineTests(unittest.TestCase):
    def test_prepare_published_definitions_and_private_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png", "c.png", "d.png"):
                (root / name).touch()
            with (root / "clinical.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([
                    ["sample_id", "md", "ght", "psd_p_value"],
                    ["eye-a", -3.01, "abnormal", 0.05],
                    ["eye-b", -1, "normal", 0.051],
                    ["eye-c", -2, "abnormal", 0.2],
                    ["eye-d", -2, "normal", 0.2],
                ])
            with (root / "images.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([
                    ["patient_id", "sample_id", "image_path", "split", "image_type"],
                    ["person-a", "eye-a", "a.png", "train", "cfp"],
                    ["person-b", "eye-b", "b.png", "test", "cfp"],
                    ["person-c", "eye-c", "c.png", "test", "cfp"],
                    ["person-d", "eye-d", "d.png", "train", "cfp"],
                ])
            output = root / "elsewhere" / "manifest.csv"
            self.assertEqual(prepare_manifest(root / "clinical.csv", root / "images.csv", output), 3)
            records = load_manifest(output)
            self.assertEqual([record.label for record in records], [1, 0, 2])
            self.assertEqual(records[0].image_path, (root / "a.png").resolve())

    def test_published_label_boundaries(self):
        self.assertEqual(classify_glaucoma(-3.01, "3", 0.05), 1)
        self.assertEqual(classify_glaucoma(-3, "abnormal", 0.05), 2)
        self.assertEqual(classify_glaucoma(-1, "1", 0.0501), 0)
        self.assertEqual(classify_glaucoma(-1, "normal", 0.05), 2)
        with self.assertRaises(ValueError):
            classify_glaucoma(-4, "unknown", 0.01)

    def test_patient_leakage_and_eye_specific_labels(self):
        a = Record("person-a", "left", Path("a"), 0, "train", "cfp")
        b = Record("person-a", "right", Path("b"), 1, "train", "cfp")
        validate_records([a, b])
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_records([a, Record("person-a", "right", Path("b"), 1, "test", "cfp")])

    def test_low_data_keeps_all_eyes_of_selected_patient(self):
        records = [
            Record("p1", "p1-left", Path("a"), 0, "train", "cfp"),
            Record("p1", "p1-right", Path("b"), 1, "train", "cfp"),
            Record("p2", "p2-left", Path("c"), 0, "train", "cfp"),
            Record("p3", "p3-left", Path("d"), 1, "train", "cfp"),
        ]
        selected = select_patient_fraction(records, 0.5, 1337)
        by_patient = {r.patient_id for r in selected}
        self.assertEqual(len(by_patient), 2)
        self.assertEqual(sum(r.patient_id == "p1" for r in selected), 2 if "p1" in by_patient else 0)

    def test_metrics_and_borderline_scenarios(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        scenarios = borderline_scenarios([0, 1, 2], [0.1, 0.9, 0.6])
        self.assertEqual(set(scenarios), {
            "borderline_excluded", "borderline_as_healthy", "borderline_as_glaucoma"
        })
        self.assertEqual(scenarios["borderline_excluded"]["n_images"], 2)

    def test_png_and_npz_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.full((8, 8, 3), 128, dtype=np.uint8)
            Image.fromarray(pixels).save(root / "image.png")
            np.savez(root / "image.npz", colorfundus=pixels)
            for name in ("image.png", "image.npz"):
                image = load_image(root / name, 224)
                self.assertEqual(image.shape, (224, 224, 3))
                self.assertAlmostEqual(float(image[0, 0, 0]), 128 / 255)


if __name__ == "__main__":
    unittest.main()
