"""Create a PRIVATE labeled manifest from clinical and image-index CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .data import load_manifest


def classify_glaucoma(md: float, ght: str, psd_p_value: float) -> int:
    """Return 1 glaucoma, 0 non-glaucoma, or 2 borderline/unclear."""
    if not math.isfinite(md) or not math.isfinite(psd_p_value) or not 0 <= psd_p_value <= 1:
        raise ValueError("MD and PSD p-value must be finite; PSD p-value must be in [0, 1]")
    ght = (ght or "").strip().lower()
    if ght in {"normal", "1"}:
        ght = "normal"
    elif ght in {"abnormal", "3"}:
        ght = "abnormal"
    else:
        raise ValueError("GHT must be normal/1 or abnormal/3")
    if md < -3 and ght == "abnormal" and psd_p_value <= 0.05:
        return 1
    if md >= -1 and ght == "normal" and psd_p_value > 0.05:
        return 0
    return 2


def prepare_manifest(clinical_csv: Path, images_csv: Path, output_csv: Path) -> int:
    """Apply the published MD/GHT/PSD definitions to each private sample."""
    labels: dict[str, int] = {}
    with clinical_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "md", "ght", "psd_p_value"}.issubset(reader.fieldnames or []):
            raise ValueError("Clinical CSV needs sample_id, md, ght, and psd_p_value columns")
        for row in reader:
            sample = row["sample_id"].strip()
            if not sample:
                raise ValueError("Blank sample ID in clinical CSV")
            try:
                md = float(row["md"])
                psd_p_value = float(row["psd_p_value"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Missing or invalid MD or PSD p-value in clinical CSV") from exc
            label = classify_glaucoma(md, row["ght"], psd_p_value)
            if sample in labels and labels[sample] != label:
                raise ValueError("Conflicting clinical labels for a sample")
            labels[sample] = label

    rows: list[dict[str, str | int]] = []
    with images_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"patient_id", "sample_id", "image_path", "split", "image_type"}.issubset(reader.fieldnames or []):
            raise ValueError("Image index needs patient_id, sample_id, image_path, split, image_type")
        for row in reader:
            sample = row["sample_id"].strip()
            if sample not in labels:
                raise ValueError("An image lacks a clinical label")
            split = row["split"].strip().lower()
            # Unclear cases are reserved for the test-set sensitivity scenarios.
            if labels[sample] == 2 and split in {"train", "val"}:
                continue
            image_path = Path(row["image_path"].strip()).expanduser()
            if not image_path.is_absolute():
                image_path = images_csv.resolve().parent / image_path
            rows.append({
                "patient_id": row["patient_id"].strip(),
                "sample_id": sample,
                "image_path": str(image_path.resolve()),
                "label": labels[sample],
                "split": split,
                "image_type": row["image_type"].strip(),
            })
    if not rows:
        raise ValueError("Image index is empty")
    # Validate before publishing even to a private destination.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "sample_id", "image_path", "label", "split", "image_type"])
        writer.writeheader()
        writer.writerows(rows)
    try:
        load_manifest(output_csv)
    except Exception:
        output_csv.unlink(missing_ok=True)
        raise
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = prepare_manifest(args.clinical, args.images, args.output)
    print(f"Validated {count} private image records")


if __name__ == "__main__":
    main()
