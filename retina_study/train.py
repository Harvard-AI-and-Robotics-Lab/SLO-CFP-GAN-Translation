"""Train a binary CFP classifier with patient-level subset experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import filter_records, load_manifest, select_patient_fraction
from .model import build_model, make_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=["vgg19", "resnet101", "efficientnetb0"], default="vgg19")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of train patients per class")
    parser.add_argument("--include-synthetic", action="store_true", help="Add synthetic CFP images for chosen train patients")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    records = load_manifest(args.manifest)
    chosen = select_patient_fraction(records, args.fraction, args.seed)
    selected_ids = {r.patient_id for r in chosen}
    train = [r for r in chosen if r.image_type == "cfp"]
    if args.include_synthetic:
        train += [r for r in records if r.split == "train"
                  and r.image_type == "synthetic_cfp" and r.patient_id in selected_ids]
    val = filter_records(records, "val")
    if not train or set(r.label for r in train) != {0, 1} or set(r.label for r in val) != {0, 1}:
        parser.error("Training and validation each require both binary classes")

    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    train_seq = make_sequence(train, args.batch_size, 224, shuffle=True, augment=True, seed=args.seed)
    val_seq = make_sequence(val, args.batch_size, 224, shuffle=False, augment=False, seed=args.seed)
    model = build_model(args.backbone)
    counts = np.bincount([r.label for r in train], minlength=2)
    weights = {i: len(train) / (2 * int(counts[i])) for i in (0, 1)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.CSVLogger(args.output_dir / "history.csv"),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
    ]
    model.fit(train_seq, validation_data=val_seq, epochs=args.epochs,
              class_weight=weights, callbacks=callbacks, verbose=2)
    summary = {
        "backbone": args.backbone,
        "fraction": args.fraction,
        "include_synthetic": args.include_synthetic,
        "seed": args.seed,
        "n_train_images": len(train),
        "n_train_patients": len(selected_ids),
        "n_val_images": len(val),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Training complete. Keep the model and run directory private until reviewed.")


if __name__ == "__main__":
    main()
