"""Inventory imported Petri photos without claiming filename counts as verified truth."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "raw" / "unlabeled" / "Petri_plates"
OUT = ROOT / "annotations" / "metadata"
REPORT = ROOT / "outputs" / "audit"
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def read_count_token(name):
    match = re.search(r"_(\d+)(_naop)?\.[^.]+$", name, flags=re.I)
    if not match:
        return "", "unparsed"
    count = int(match.group(1))
    if match.group(2):
        return count, "special_suffix_needs_review"
    if count >= 300:
        return count, "possible_capped_count_needs_review"
    return count, "filename_count_unverified"


def category(name):
    lower = name.lower()
    for key in ("ecoli", "saureus"):
        if key in lower:
            return key
    if "paeruginosa" in lower or "paureginosa" in lower:
        return "paeruginosa"
    return "other_or_unknown"


def main():
    if not SOURCE.exists():
        raise SystemExit(f"Source directory not found: {SOURCE}")
    images = sorted(p for p in SOURCE.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"No images in {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in images:
        n, status = read_count_token(path.name)
        rows.append({
            "filename": path.name,
            "relative_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "filename_species_hint": category(path.name),
            "filename_count_token": n,
            "count_status": status,
            "verified_ground_truth": "no",
            "has_bbox_annotation": "no",
        })
    output = OUT / "source_inventory.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "images": len(rows),
        "bytes": sum(int(r["bytes"]) for r in rows),
        "categories_from_filenames_only": dict(Counter(r["filename_species_hint"] for r in rows)),
        "count_status": dict(Counter(r["count_status"] for r in rows)),
        "annotation_status": "No per-colony bounding boxes found in imported source directory",
        "caution": "Filename count tokens are not verified reference labels. Values >=300 may be censored.",
    }
    path = REPORT / "source_inventory.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("CSV:", output)
    print("JSON:", path)


if __name__ == "__main__":
    main()
