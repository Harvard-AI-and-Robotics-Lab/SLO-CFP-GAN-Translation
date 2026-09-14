"""Private manifest validation, loading, and patient-level subset selection."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = {"patient_id", "sample_id", "image_path", "label", "split", "image_type"}
SPLITS = {"train", "val", "test"}
IMAGE_TYPES = {"cfp", "slo", "synthetic_cfp"}


@dataclass(frozen=True)
class Record:
    patient_id: str
    sample_id: str
    image_path: Path
    label: int
    split: str
    image_type: str


def load_manifest(path: str | Path, *, check_files: bool = True) -> list[Record]:
    """Load a local manifest; never print identifiers or image paths."""
    manifest = Path(path).expanduser().resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain {sorted(REQUIRED_COLUMNS)}")
        records = []
        for row_number, row in enumerate(reader, start=2):
            try:
                patient_id = row["patient_id"].strip()
                sample_id = row["sample_id"].strip()
                raw_path = row["image_path"].strip()
                label = int(row["label"])
                split = row["split"].strip().lower()
                image_type = row["image_type"].strip().lower()
                if not patient_id or not sample_id or not raw_path:
                    raise ValueError("patient_id, sample_id, and image_path are required")
                if label not in (0, 1, 2) or split not in SPLITS or image_type not in IMAGE_TYPES:
                    raise ValueError("invalid label, split, or image_type")
                image_path = Path(raw_path).expanduser()
                if not image_path.is_absolute():
                    image_path = manifest.parent / image_path
                if check_files and not image_path.is_file():
                    raise ValueError("image file does not exist")
                records.append(Record(patient_id, sample_id, image_path.resolve(), label, split, image_type))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid manifest row {row_number}: {exc}") from exc
    validate_records(records)
    return records


def validate_records(records: list[Record]) -> None:
    if not records:
        raise ValueError("Manifest is empty")
    patient_split: dict[str, str] = {}
    sample_label: dict[str, int] = {}
    sample_patient: dict[str, str] = {}
    seen: set[tuple[str, Path, str]] = set()
    for record in records:
        if record.patient_id in patient_split and patient_split[record.patient_id] != record.split:
            raise ValueError("A patient appears in multiple splits")
        if record.sample_id in sample_label and sample_label[record.sample_id] != record.label:
            raise ValueError("A sample has inconsistent labels")
        if record.sample_id in sample_patient and sample_patient[record.sample_id] != record.patient_id:
            raise ValueError("A sample is assigned to multiple patients")
        key = (record.patient_id, record.image_path, record.split)
        if key in seen:
            raise ValueError("Duplicate manifest image")
        seen.add(key)
        patient_split[record.patient_id] = record.split
        sample_label[record.sample_id] = record.label
        sample_patient[record.sample_id] = record.patient_id
        if record.split == "train" and record.label == 2:
            raise ValueError("Borderline labels are not supported in binary training")
        if record.split != "train" and record.image_type == "synthetic_cfp":
            raise ValueError("Synthetic images are allowed only in training")


def select_patient_fraction(records: list[Record], fraction: float, seed: int) -> list[Record]:
    """Stratify a training subset by patient; include every chosen patient's images."""
    import random

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    patient_label: dict[str, int] = {}
    for record in records:
        if record.split != "train":
            continue
        # A patient with either eye positive belongs to the positive stratum.
        patient_label[record.patient_id] = max(patient_label.get(record.patient_id, 0), record.label)
    by_label: dict[int, list[str]] = {
        label: [patient for patient, value in patient_label.items() if value == label]
        for label in (0, 1)
    }
    if not all(by_label.values()):
        raise ValueError("Training requires patients in both binary classes")
    rng = random.Random(seed)
    chosen: set[str] = set()
    for patient_ids in by_label.values():
        rng.shuffle(patient_ids)
        count = max(1, round(len(patient_ids) * fraction))
        chosen.update(patient_ids[:count])
    return [record for record in records if record.split == "train" and record.patient_id in chosen]


def filter_records(records: list[Record], split: str, image_type: str = "cfp") -> list[Record]:
    selected = [r for r in records if r.split == split and r.image_type == image_type]
    if not selected:
        raise ValueError(f"No {image_type} images in {split} split")
    return selected
