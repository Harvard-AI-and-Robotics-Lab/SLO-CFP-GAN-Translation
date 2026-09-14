"""Match private SLO and CFP indexes by an explicit, exact study key."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_index(path: Path) -> dict[str, str]:
    """Index CSV columns: match_key, image_path. Reject ambiguous matches."""
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"match_key", "image_path"}.issubset(reader.fieldnames or []):
            raise ValueError("Index needs match_key and image_path columns")
        for row in reader:
            key = row["match_key"].strip()
            image = row["image_path"].strip()
            if not key or not image or key in result:
                raise ValueError("Blank or duplicate match key or image path")
            image_path = Path(image).expanduser()
            if not image_path.is_absolute():
                image_path = path.resolve().parent / image_path
            if not image_path.is_file():
                raise ValueError("An indexed image file is missing")
            result[key] = str(image_path.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slo-index", type=Path, required=True)
    parser.add_argument("--cfp-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Private pair index")
    args = parser.parse_args()
    slo = read_index(args.slo_index)
    cfp = read_index(args.cfp_index)
    keys = sorted(slo.keys() & cfp.keys())
    if not keys:
        raise ValueError("No exact SLO/CFP matches")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["match_key", "slo_path", "cfp_path"])
        writer.writerows((key, slo[key], cfp[key]) for key in keys)
    print(f"Matched {len(keys)} private image pairs")


if __name__ == "__main__":
    main()
