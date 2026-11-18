"""Choose a reproducible diverse pilot set; filename tokens are NOT ground truth."""
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "annotations" / "metadata" / "source_inventory.csv"
OUTPUT = ROOT / "annotations" / "metadata" / "pilot_24_manifest.csv"
rng = random.Random(20261118)

def bucket(raw):
    if raw in ("", None):
        return "unknown"
    n = int(raw)
    if n <= 50:
        return "low_0_50_hint"
    if n <= 150:
        return "medium_51_150_hint"
    if n < 300:
        return "high_151_299_hint"
    return "capped_300_plus_hint"

def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in rows:
        key = (row["filename_species_hint"], bucket(row["filename_count_token"]))
        groups[key].append(row)
    chosen = []
    for species in ("ecoli", "saureus", "paeruginosa"):
        for density in ("low_0_50_hint", "medium_51_150_hint",
                        "high_151_299_hint", "capped_300_plus_hint"):
            candidates = sorted(groups.get((species, density), []),
                                key=lambda r: r["filename"])
            for row in rng.sample(candidates, min(2, len(candidates))):
                chosen.append({
                    "image": row["filename"],
                    "relative_path": row["relative_path"],
                    "species_filename_hint": species,
                    "count_filename_hint": row["filename_count_token"],
                    "density_filename_hint": density,
                    "annotation_status": "unlabeled",
                    "ground_truth": "NOT_VERIFIED",
                    "review_status": "pending",
                })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(chosen[0].keys()))
        writer.writeheader()
        writer.writerows(chosen)
    print("Pilot images:", len(chosen))
    for row in chosen:
        print(row["species_filename_hint"], row["density_filename_hint"], row["image"])
    print("Manifest:", OUTPUT)

if __name__ == "__main__":
    main()
