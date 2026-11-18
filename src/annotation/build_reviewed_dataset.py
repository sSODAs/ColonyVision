"""Build YOLO train/val/test only from human-approved and grouped colony reviews.
1) --init-plan creates a manifest to fill with split and plate_group.
2) --build checks labels, bytes, group leakage and copies approved files only.
"""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from review_colonies import ROOT, REVIEWED, DRAFT, IMAGES, norm_box, yolo_line

PLAN = ROOT / "annotations" / "metadata" / "reviewed_splits.csv"
DEST = ROOT / "dataset"
FIELDS = ["image", "split", "plate_group", "reviewed_count", "notes"]
ALLOWED = {"train", "val", "test"}

def reviewed_items():
    items = {}
    if not REVIEWED.exists():
        return items
    for path in sorted(REVIEWED.glob("*.jpg.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "reviewed":
            continue
        name = data.get("image")
        if not name or name in items:
            raise ValueError("Missing or duplicate reviewed image: " + str(path))
        draft = DRAFT / (name + ".json")
        if draft.exists() and draft.stat().st_mtime >= path.stat().st_mtime:
            raise ValueError("A newer unsanctioned draft exists; re-approve before training: " + name)
        image = IMAGES / name
        if not image.is_file():
            raise ValueError("Reviewed source image missing: " + str(image))
        if hashlib.sha256(image.read_bytes()).hexdigest() != data["sha256"]:
            raise ValueError("Reviewed image changed since approval: " + name)
        w,h=int(data["width"]),int(data["height"])
        boxes=data["boxes"]
        if len(boxes) != data["box_count"]:
            raise ValueError("Reviewed box count mismatch: " + name)
        for box in boxes:
            if norm_box(box,w,h) != box:
                raise ValueError("Invalid box: " + name)
        label = REVIEWED / (name + ".txt")
        if not label.exists():
            raise ValueError("Missing YOLO label: " + str(label))
        wanted="\n".join(yolo_line(b,w,h) for b in boxes).strip()
        actual=label.read_text(encoding="utf-8").strip()
        if actual != wanted:
            raise ValueError("YOLO labels do not match approved JSON: " + name)
        items[name]=(image,label,len(boxes))
    return items

def init_plan(items):
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        with PLAN.open(encoding="utf-8-sig", newline="") as f:
            existing=list(csv.DictReader(f))
        if existing:
            present={r["image"] for r in existing}
            missing=sorted(set(items)-present)
            if not missing:
                print("Split plan exists; no overwrite:", PLAN)
                return
            with PLAN.open("a",encoding="utf-8-sig",newline="") as f:
                writer=csv.writer(f)
                for name in missing:writer.writerow([name,"","",items[name][2],""])
            print("Appended",len(missing),"reviewed images:", PLAN)
            return
    with PLAN.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(FIELDS)
        for name in sorted(items):writer.writerow([name,"","",items[name][2],""])
    print("Created split plan:",PLAN)
    print("Fill split=train/val/test and plate_group=unique physical plate/session ID before --build.")

def build(items):
    if not items:
        raise ValueError("No human-approved images. Review and approve first.")
    if not PLAN.exists():
        raise ValueError("Run --init-plan then assign split and plate_group.")
    with PLAN.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError("Split plan has no rows")
    seen=set()
    groups=defaultdict(set)
    splits=Counter()
    tasks=[]
    for row in rows:
        name=row["image"]
        split=row.get("split","").strip().lower()
        group=row.get("plate_group","").strip()
        if name in seen:
            raise ValueError("Duplicate row: "+name)
        seen.add(name)
        if name not in items:
            raise ValueError("No valid human-approved label for "+name)
        if split not in ALLOWED:
            raise ValueError("Assign train, val or test for "+name)
        if not group or group.upper() in ("UNKNOWN","UNSET","TBD"):
            raise ValueError("Assign physical plate/session group for "+name)
        groups[group].add(split)
        splits[split]+=1
        source,label,_=items[name]
        tasks.extend([
            (source,DEST/"images"/split/name),
            (label,DEST/"labels"/split/(Path(name).stem+".txt"))
        ])
    if set(items)!=seen:
        raise ValueError("Split plan missing human-approved images: "+str(sorted(set(items)-seen)[:5]))
    overlap={key:value for key,value in groups.items() if len(value)>1}
    if overlap:
        raise ValueError("Group leakage across splits: "+str(overlap))
    if not splits["train"] or not splits["val"]:
        raise ValueError("Both train and val need at least one reviewed image")
    # Preflight all paths to avoid overwriting existing data or accidental dataset mixing.
    for source,target in tasks:
        if target.exists():
            if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():
                raise ValueError("Different file already exists; refused to overwrite: "+str(target))
    for source,target in tasks:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
    print("Built verified YOLO dataset:",dict(splits))
    print("Class 0 = colony; source images and reviewed annotations left unchanged.")

def main():
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--init-plan",action="store_true")
    action.add_argument("--build",action="store_true")
    args=parser.parse_args()
    items=reviewed_items()
    print("Human-approved images:",len(items))
    if args.init_plan:init_plan(items)
    else:build(items)

if __name__=="__main__":
    main()
