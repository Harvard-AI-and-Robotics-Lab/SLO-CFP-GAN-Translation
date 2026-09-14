"""Evaluate a trained binary model on ordered, real-image test records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import filter_records, load_manifest
from .metrics import binary_metrics, borderline_scenarios
from .model import make_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Private aggregate JSON output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-specificity", type=float, default=0.95)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.threshold <= 1:
        parser.error("Invalid batch size or threshold")

    records = filter_records(load_manifest(args.manifest), "test")
    import tensorflow as tf

    sequence = make_sequence(records, args.batch_size, 224, shuffle=False, augment=False, seed=0)
    model = tf.keras.models.load_model(args.model)
    probabilities = np.asarray(model.predict(sequence, verbose=0)).reshape(-1)
    labels = np.asarray([r.label for r in records])
    if len(probabilities) != len(labels):
        raise RuntimeError("Prediction and label counts differ")
    kwargs = {"threshold": args.threshold, "target_specificity": args.target_specificity}
    metrics = (borderline_scenarios(labels, probabilities, **kwargs)
               if 2 in labels else binary_metrics(labels, probabilities, **kwargs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("Aggregate evaluation complete. Review output before sharing.")


if __name__ == "__main__":
    main()
