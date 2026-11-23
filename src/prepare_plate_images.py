"""ColonyVision: non-destructive ROI / preprocessing / Teacher ablation.

Place this file at <project>/src/roi_teacher_v2.py.
Only writes into NEW outputs/roi_v2 and outputs/teacher/roi_v2 paths.
Does NOT edit raw/, existing Teacher predictions, dataset/, or training weights.

Commands:
    python src/roi_teacher_v2.py preview --limit 20
    python src/roi_teacher_v2.py compare --limit 20
    python src/roi_teacher_v2.py run --limit 25
    python src/roi_teacher_v2.py run --limit 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing.plate_roi import estimate_plate
from preprocessing.plate_hough import estimate_plate_hough

DEFAULT_MANIFEST = ROOT / "annotations" / "metadata" / "teacher_split_1200.csv"
DEFAULT_OLD_TEACHER = ROOT / "outputs" / "teacher" / "adbc_1200" / "detections"
DEFAULT_WEIGHTS = ROOT / "weights" / "teacher_adbc_frozen.pt"
DEFAULT_EXPERIMENT = ROOT / "outputs" / "roi_v2"
DEFAULT_TEACHER_OUT = ROOT / "outputs" / "teacher" / "roi_v2"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path, splits: tuple[str, ...]) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Missing split CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        entries = list(csv.DictReader(f))
    required = {"image", "split", "source"}
    if not entries or not required.issubset(entries[0]):
        raise SystemExit("Manifest must include image, split, source")
    names = [r["image"].lower() for r in entries]
    if len(names) != len(set(names)):
        raise SystemExit("Duplicate image names in manifest")
    return [r for r in entries if r["split"] in splits]


def sample_manifest(entries: list[dict], k: int, old_detections: Path) -> list[dict]:
    """Deterministic species-stratified examples across low/high Teacher counts."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in entries:
        group = row.get("species_filename_hint") or "unknown"
        path = old_detections / (row["image"] + ".json")
        count = -1
        if path.is_file():
            try:
                count = int(json.loads(path.read_text(encoding="utf-8")).get("count") or 0)
            except (ValueError, TypeError, OSError):
                pass
        by_group[group].append({**row, "_teacher_count": count})
    for group in by_group:
        by_group[group].sort(key=lambda x: (x["_teacher_count"], x["image"].lower()))
    groups = sorted(by_group)
    selected = []
    # Alternate low/high and median count within each species.
    cycle = (0.03, 0.98, 0.50, 0.22, 0.76, 0.37, 0.63, 0.11, 0.89)
    used: set[str] = set()
    for j in range(min(k, len(entries))):
        group = groups[j % len(groups)]
        values = by_group[group]
        candidates = [v for v in values if v["image"] not in used]
        if not candidates:
            group = next((g for g in groups if any(v["image"] not in used for v in by_group[g])), None)
            if group is None:
                break
            values = by_group[group]
            candidates = [v for v in values if v["image"] not in used]
        fraction = cycle[(j // len(groups)) % len(cycle)]
        pos = round(fraction * (len(candidates) - 1))
        record = candidates[pos]
        used.add(record["image"])
        selected.append(record)
    return selected


def read_upright(row: dict) -> Image.Image:
    path = Path(row["source"])
    if not path.is_file():
        raise FileNotFoundError(f"Source image missing: {path}")
    with Image.open(path) as f:
        return ImageOps.exif_transpose(f).convert("RGB")


def locate_plate(image: Image.Image):
    """Image-only dish circle.

    Prefer the scored Hough rim fit adapted from AST-MEASURE. The older
    threshold estimate remains a fallback only.
    """
    circle = estimate_plate_hough(image)
    if circle is not None:
        cx, cy, radius = [float(v) for v in circle[:3]]
        method = "hough_scored_ast_style"
    else:
        circle, status = estimate_plate(image)
        if circle is None:
            return None, "roi_not_found"
        cx, cy, radius = [float(v) for v in circle[:3]]
        method = "threshold_fallback_unverified"
    w, h = image.size
    if (not all(math.isfinite(x) for x in (cx, cy, radius))
        or not 0 <= cx <= w or not 0 <= cy <= h
        or radius < min(w, h)*0.20 or radius > min(w, h)*0.65):
        return None, "roi_geometry_rejected"
    return (cx, cy, radius), method


def crop_plate(image: Image.Image, circle: tuple[float, float, float], padding: float = 1.06):
    cx, cy, r = circle
    r *= padding
    x0 = max(0, int(math.floor(cx-r)))
    y0 = max(0, int(math.floor(cy-r)))
    x1 = min(image.width, int(math.ceil(cx+r)))
    y1 = min(image.height, int(math.ceil(cy+r)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid plate crop")
    return image.crop((x0, y0, x1, y1)), (x0, y0)


def clahe_luma(image: Image.Image, clip: float, grid: int, gamma: float, glare_clip: int):
    """Conservative L-channel enhancement: no inpainting or object fabrication."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    light = lab[:, :, 0]
    if clip > 0:
        light = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(light)
    if gamma != 1.0:
        # gamma>1 brightens dark pixels; gamma<1 darkens them.
        lut = np.rint(255.0 * (np.arange(256) / 255.0) ** (1.0 / gamma)).astype(np.uint8)
        light = cv2.LUT(light, lut)
    if glare_clip < 255:
        # Optional upper-L compression, deliberately disabled by default:
        # bright white colonies can be genuine colonies, not glare.
        high = np.maximum(0, light.astype(np.int16) - glare_clip)
        light = np.uint8(np.clip(light.astype(np.int16) - high // 2, 0, 255))
    lab[:, :, 0] = light
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def mask_circle(image: Image.Image, circle: tuple[float, float, float], offset: tuple[int, int], ratio: float):
    cx, cy, r = circle
    x0, y0 = offset
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    rad = r*ratio
    draw.ellipse((cx-x0-rad, cy-y0-rad, cx-x0+rad, cy-y0+rad), fill=255)
    return Image.composite(image, Image.new("RGB", image.size, "black"), mask)


def get_blur_score(crop: Image.Image) -> float:
    thumb = crop.copy()
    thumb.thumbnail((650, 650), Image.Resampling.BILINEAR)
    gray = cv2.cvtColor(np.asarray(thumb), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preprocess(image: Image.Image, circle, args):
    base, offset = crop_plate(image, circle)
    changed = clahe_luma(base, args.clahe_clip, args.clahe_grid, args.gamma, args.glare_clip)
    masked = mask_circle(changed, circle, offset, args.inner_ratio)
    return base, changed, masked, offset


def old_teacher_predictions(path: Path):
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if doc.get("status") != "pseudo_unverified":
        return None
    data = doc.get("boxes_xyxy_conf", [])
    return [list(map(float, box)) for box in data if len(box) == 5]


def box_iou(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    overlap = max(0.0, x1-x0) * max(0.0, y1-y0)
    area_a = max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1])
    area_b = max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    return overlap / max(1e-9, area_a + area_b - overlap)


def filter_boxes(boxes: list[list[float]], circle, image_size, args):
    """Only accept confident center-inside-radius candidates; post-NMS duplicates.

    No claim of complete colony detection; missed boxes cannot be recovered here.
    """
    cx, cy, r = circle
    w, h = image_size
    candidates = []
    rejected: list[dict] = []
    for box in boxes:
        if len(box) != 5:
            continue
        try:
            x0, y0, x1, y1, score = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        b = [x0, y0, x1, y1, score]
        reason = None
        if not all(math.isfinite(t) for t in b) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            reason = "invalid_geometry"
        elif score < args.keep_conf:
            reason = "low_confidence"
        else:
            midx, midy = (x0+x1)/2.0, (y0+y1)/2.0
            if math.hypot(midx-cx, midy-cy) > r*args.inner_ratio:
                reason = "outside_inner_circle"
            else:
                bw, bh = x1-x0, y1-y0
                if min(bw, bh) < args.min_side:
                    reason = "too_small"
                elif max(bw, bh) > r*args.max_side_fraction:
                    reason = "too_large"
                elif max(bw,bh)/max(1e-6,min(bw,bh)) > args.max_aspect:
                    reason = "extreme_aspect_ratio"
        if reason:
            rejected.append({"box": b, "reason": reason})
        else:
            candidates.append(b)
    candidates.sort(key=lambda b: b[4], reverse=True)
    accepted=[]
    for b in candidates:
        if any(box_iou(b, prior) > args.post_nms_iou for prior in accepted):
            rejected.append({"box": b, "reason": "duplicate_nms"})
        else:
            accepted.append(b)
    return accepted, rejected


def draw_circle(draw, circle, scale_x, scale_y, origin_x=0, origin_y=0, inner_ratio=.93):
    cx, cy, r = circle
    for radius, color in ((r, "#ffb74d"), (r*inner_ratio, "#5edcff")):
        bbox = ((cx-radius-origin_x)*scale_x, (cy-radius-origin_y)*scale_y,
                (cx+radius-origin_x)*scale_x, (cy+radius-origin_y)*scale_y)
        draw.ellipse(bbox, outline=color, width=3)


def annotated_thumb(image: Image.Image, circle, accepted, rejected, ratio, size=530, offset=(0,0), show_circle=True):
    result = image.copy()
    src_w, src_h = result.size
    result.thumbnail((size,size),Image.Resampling.BILINEAR)
    sx, sy = result.width/src_w, result.height/src_h
    draw = ImageDraw.Draw(result)
    ox, oy = offset
    if show_circle:
        draw_circle(draw, circle, sx, sy, ox, oy, ratio)
    for item in rejected:
        box = item["box"]
        draw.rectangle(((box[0]-ox)*sx, (box[1]-oy)*sy,
                        (box[2]-ox)*sx, (box[3]-oy)*sy),
                       outline="#f65765", width=2)
    for box in accepted:
        draw.rectangle(((box[0]-ox)*sx, (box[1]-oy)*sy,
                        (box[2]-ox)*sx, (box[3]-oy)*sy),
                       outline="#40f99b", width=2)
    return result


def make_montage(panels, subtitles, outfile: Path, header: str, footer: str):
    """Panels: four PIL images 2x2; optionally separate model predictions."""
    side = 580
    gutter = 10
    top = 56
    cell_h = 610
    canvas = Image.new("RGB",(side*2+gutter*3,top+cell_h*2+95),(23,28,33))
    d=ImageDraw.Draw(canvas)
    d.text((20,18),header[:155],fill="white")
    for idx, panel in enumerate(panels):
        row, col = divmod(idx,2)
        x = gutter+(side+gutter)*col
        y = top+cell_h*row
        frame = Image.new("RGB",(side,side),(38,43,49))
        pic = panel.copy()
        pic.thumbnail((side,side),Image.Resampling.LANCZOS)
        frame.paste(pic,((side-pic.width)//2,(side-pic.height)//2))
        canvas.paste(frame,(x,y))
        d.text((x+2,y+side+9),subtitles[idx][:92],fill="#deebf1")
    d.text((20,top+cell_h*2+18),footer[:165],fill="#f7ca82")
    outfile.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(outfile,"JPEG",quality=85)


def common_settings(args, model_sha: str):
    return {
        "teacher_sha256": model_sha,
        "imgsz": args.imgsz, "tile": args.tile,
        "stride": args.stride, "candidate_conf": args.candidate_conf,
        "global_nms_iou": args.global_nms_iou,
        "keep_conf": args.keep_conf,
        "post_nms_iou": args.post_nms_iou,
        "inner_ratio": args.inner_ratio,
        "min_side": args.min_side,
        "max_side_fraction": args.max_side_fraction,
        "max_aspect": args.max_aspect,
        "clahe_clip": args.clahe_clip,
        "clahe_grid": args.clahe_grid,
        "gamma": args.gamma,
        "glare_clip": args.glare_clip,
        "input_variant": "LAB_CLAHE_inner_mask",
    }


def baseline_preview(args, entries):
    out = args.preview_out
    report = []
    for idx, row in enumerate(entries, 1):
        original = read_upright(row)
        plate, method = locate_plate(original)
        old = old_teacher_predictions(args.old_teacher / (row["image"] + ".json"))
        if plate is None:
            report.append([row["image"],row["split"],"roi_missing",0,0,0])
            print(f"[{idx}/{len(entries)}] {row['image']}: NO ROI, skipped",flush=True)
            continue
        crop, enhanced, masked, offset = preprocess(original, plate, args)
        accepted, rejected = filter_boxes(old or [], plate, original.size, args)
        blur = round(get_blur_score(crop), 2)
        low_focus = blur < args.blur_warn_threshold
        panels = [annotated_thumb(original,plate,[],[],args.inner_ratio),
                  annotated_thumb(masked,plate,[],[],args.inner_ratio,offset=offset),
                  annotated_thumb(original,plate,accepted,rejected,args.inner_ratio),
                  annotated_thumb(masked,plate,accepted,rejected,args.inner_ratio,offset=offset)]
        subtitles = ["Original + OUTER (orange) / INNER (cyan)",
                     "LAB CLAHE + black outside INNER (NO inference)",
                     "EXISTING Teacher boxes: kept green / rejected red",
                     "SAME old boxes on enhanced image (NOT re-inferred)"]
        outfile = out / (Path(row["image"]).stem + "_roi_preview.jpg")
        make_montage(panels,subtitles,outfile,
                     f"PREVIEW {idx}/{len(entries)} | {row['image']}",
                     f"old candidate={len(old or [])}; kept={len(accepted)}; rejected={len(rejected)}; "
                     f"blur_score={blur}; low_focus_REVIEW={low_focus}; roi={method}")
        report.append([row["image"],row["split"],method,len(old or []),len(accepted),len(rejected)])
        print(f"[{idx}/{len(entries)}] {row['image']}: old {len(old or [])} / kept {len(accepted)} / rejected {len(rejected)}",flush=True)
    write_report(out / "roi_preview_report.csv",
                 ["image","split","roi_method","old_teacher_boxes","accepted","rejected"],report)
    print("PREVIEW OUTPUT:",out)
    print("Preview 'enhanced' uses OLD boxes ONLY. Use compare to measure new inference.")


def write_report(path, header, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)


def predict_mode(model, device, original, plate, args, mode):
    """Raw plate crop vs masked/enhanced plate crop, same Teacher."""
    crop, enhanced, masked, offset = preprocess(original,plate,args)
    if mode == "raw":
        work = crop
    elif mode == "mask":
        work = mask_circle(crop, plate, offset, args.inner_ratio)
    elif mode == "processed":
        work = masked
    else:
        raise ValueError("Unknown processing mode: " + mode)
    # Uses the existing tested ColonyVision tiling + cross-tile NMS.
    from predict_tiled import infer
    predictions, raw_total = infer(model,work,args.tile,args.stride,
                                   args.candidate_conf,args.imgsz,args.global_nms_iou,device)
    ox, oy = offset
    shifted = [[b[0]+ox,b[1]+oy,b[2]+ox,b[3]+oy,b[4]] for b in predictions]
    accepted,rejected=filter_boxes(shifted,plate,original.size,args)
    return {"accepted":accepted, "rejected":rejected,
            "candidates_before_tile_nms":raw_total,
            "tile_nms_boxes":len(predictions),
            "blur_score":round(get_blur_score(crop),2)}, (crop,masked,offset)


def load_teacher(args):
    if not args.model.is_file():
        raise SystemExit(f"Frozen ADBC Teacher missing: {args.model}")
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))
    return model,device,digest_file(args.model)


def compare(args, entries):
    model,device,teacher_sha=load_teacher(args)
    out=args.compare_out
    rows=[]
    for index,row in enumerate(entries,1):
        original=read_upright(row)
        plate,method=locate_plate(original)
        if plate is None:
            rows.append([row["image"],row["split"],"roi_missing","","",""])
            print(f"[{index}/{len(entries)}] {row['image']} - ROI MISSING",flush=True)
            continue
        before,(crop,masked,offset)=predict_mode(model,device,original,plate,args,"raw")
        masked_only,_=predict_mode(model,device,original,plate,args,"mask")
        after,_=predict_mode(model,device,original,plate,args,"processed")
        raw_boxes=before["accepted"];mask_boxes=masked_only["accepted"];enh_boxes=after["accepted"]
        orig=annotated_thumb(original,plate,[],[],args.inner_ratio)
        left=annotated_thumb(original,plate,raw_boxes,before["rejected"],args.inner_ratio)
        middle=annotated_thumb(original,plate,mask_boxes,masked_only["rejected"],args.inner_ratio)
        right=annotated_thumb(original,plate,enh_boxes,after["rejected"],args.inner_ratio)
        outfile=out/"overlays"/(Path(row["image"]).stem+"_comparison.jpg")
        make_montage([orig,left,middle,right],
                     ["Original image + outer / inner circles",
                      f"A RAW input + inner filter: kept={len(raw_boxes)}, dropped={len(before['rejected'])}",
                      f"B INNER MASK only: kept={len(mask_boxes)}, dropped={len(masked_only['rejected'])}",
                      f"C CLAHE + INNER MASK: kept={len(enh_boxes)}, dropped={len(after['rejected'])}"],
                     outfile,f"A/B/C Teacher {index}/{len(entries)} | {row['image']}",
                     "Green=kept; Red=rejected; orange=outer; cyan=inner. Count difference is NOT accuracy.")
        payload={"source":row["image"],"source_sha256":digest_file(Path(row["source"])),
                 "teacher_sha256":teacher_sha,"split":row["split"],
                 "image_size":{"width":original.width,"height":original.height},
                 "plate_estimate":list(plate),"roi_method":method,
                 "settings":common_settings(args,teacher_sha),
                 "raw":before,"mask_only":masked_only,"processed":after,
                 "status":"comparison_unverified"}
        dest=out/"detections"/(row["image"]+".json")
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        rows.append([row["image"],row["split"],method,len(raw_boxes),len(mask_boxes),
                     len(enh_boxes),len(mask_boxes)-len(raw_boxes),len(enh_boxes)-len(mask_boxes)])
        print(f"[{index}/{len(entries)}] {row['image']}: raw={len(raw_boxes)} mask={len(mask_boxes)} clahe_mask={len(enh_boxes)}",flush=True)
    write_report(out/"compare_counts.csv",
                 ["image","split","roi_method","raw_filtered_count",
                  "inner_mask_count","clahe_inner_mask_count",
                  "mask_minus_raw","clahe_minus_mask"],rows)
    print("COMPARISON OUTPUT:",out)
    print("NOTE: Neither raw nor processed count is independently verified accuracy.")


def saved_rows(manifest, out):
    rows=[]
    for row in manifest:
        source=out/"detections"/(row["image"]+".json")
        if not source.is_file():
            continue
        try:
            data=json.loads(source.read_text(encoding="utf-8"))
        except (ValueError,OSError):
            continue
        rows.append([row["image"],row["split"],data.get("status",""),
                     data.get("count",""),data.get("roi_method","")])
    return rows


def run_teacher(args, entries):
    model,device,teacher_sha=load_teacher(args)
    settings=common_settings(args,teacher_sha)
    out=args.teacher_out
    (out/"detections").mkdir(parents=True,exist_ok=True)
    (out/"overlays").mkdir(parents=True,exist_ok=True)
    processed_new=0;skipped=0;failed=0
    for index,row in enumerate(entries,1):
        original_file=Path(row["source"])
        destination=out/"detections"/(row["image"]+".json")
        if destination.is_file():
            try:
                previous=json.loads(destination.read_text(encoding="utf-8"))
            except (ValueError,OSError):
                raise SystemExit(f"Unreadable prior result: {destination}; do not overwrite blindly")
            if (previous.get("settings") != settings or
                previous.get("source_sha256") != digest_file(original_file) or
                previous.get("teacher_sha256") != teacher_sha):
                raise SystemExit("INCOMPATIBLE EXISTING OUTPUT. Use a NEW --teacher-out for changed settings/model: "
                                 +str(destination))
            skipped+=1
            continue
        if args.limit and processed_new>=args.limit:
            break
        try:
            original=read_upright(row)
            plate,method=locate_plate(original)
            if plate is None:
                doc={"source":row["image"],"source_sha256":digest_file(original_file),
                     "split":row["split"],"status":"roi_missing", "count":None,
                     "image_size":{"width":original.width,"height":original.height},
                     "boxes_xyxy_conf":[],"teacher_sha256":teacher_sha,
                     "settings":settings,"roi_method":method}
            else:
                detection,_=predict_mode(model,device,original,plate,args,"processed")
                boxes=detection["accepted"]
                doc={"source":row["image"],"source_sha256":digest_file(original_file),
                     "teacher_sha256":teacher_sha,"status":"pseudo_unverified",
                     "split":row["split"],"count":len(boxes),
                     "image_size":{"width":original.width,"height":original.height},
                     "boxes_xyxy_conf":boxes,"rejected_candidates":detection["rejected"],
                     "plate_estimate":list(plate),"roi_method":method,
                     "settings":settings,
                     "quality":{"blur_score":detection["blur_score"],
                                "review_low_focus":detection["blur_score"] < args.blur_warn_threshold,
                                "blur_requires_manual_calibration":True},
                     "note":"Pseudo labels, not independently verified GT; 200 test images MUST NOT train Student"}
                overlay=annotated_thumb(original,plate,boxes,detection["rejected"],args.inner_ratio,size=1350)
                overpath=out/"overlays"/(Path(row["image"]).stem+"_preview.jpg")
                overlay.save(overpath,"JPEG",quality=90)
            tmp=destination.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc,indent=2),encoding="utf-8")
            tmp.replace(destination)
            processed_new+=1
            print(f"[{index}/{len(entries)}] {row['split']} {row['image']} status={doc['status']} count={doc['count']}",flush=True)
        except Exception as exc:
            failed+=1
            print(f"[ERROR] {row['image']}: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
    rows=saved_rows(entries,out)
    write_report(out/"counts.csv",
                 ["image","split","status","predicted_count","roi_method"],rows)
    print(f"Teacher new={processed_new} skipped_prior={skipped} errors={failed} "
          f"total_results={len(rows)}/{len(entries)}; saved to {out}")
    if failed:
        raise SystemExit(f"{failed} images failed; check logs and rerun with same settings")


def parser_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("preview","compare","run"))
    parser.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument("--old-teacher",type=Path,default=DEFAULT_OLD_TEACHER)
    parser.add_argument("--model",type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument("--preview-out",type=Path,default=DEFAULT_EXPERIMENT/"preview")
    parser.add_argument("--compare-out",type=Path,default=DEFAULT_EXPERIMENT/"compare")
    parser.add_argument("--teacher-out",type=Path,default=DEFAULT_TEACHER_OUT)
    parser.add_argument("--limit",type=int,default=20,help="preview/compare examples OR number of NEW run outputs; 0=all")
    parser.add_argument("--inner-ratio",type=float,default=0.93)
    parser.add_argument("--clahe-clip",type=float,default=2.0)
    parser.add_argument("--clahe-grid",type=int,default=8)
    parser.add_argument("--gamma",type=float,default=1.0)
    parser.add_argument("--glare-clip",type=int,default=255,
                        help="255 disables; lower values compress highlights but can damage white colonies")
    parser.add_argument("--tile",type=int,default=768)
    parser.add_argument("--stride",type=int,default=640)
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--candidate-conf",type=float,default=0.20)
    parser.add_argument("--global-nms-iou",type=float,default=0.60)
    parser.add_argument("--keep-conf",type=float,default=0.35)
    parser.add_argument("--post-nms-iou",type=float,default=0.48)
    parser.add_argument("--blur-warn-threshold",type=float,default=35.0,
                        help="Flag low-focus images for review; do not automatically remove")
    parser.add_argument("--min-side",type=float,default=3.0)
    parser.add_argument("--max-side-fraction",type=float,default=0.25)
    parser.add_argument("--max-aspect",type=float,default=3.5)
    args=parser.parse_args()
    if not 0.80<=args.inner_ratio<=0.99:
        parser.error("--inner-ratio must be between 0.80 and 0.99")
    if not 0 < args.stride<=args.tile or args.limit<0:
        parser.error("Require 0 < stride <= tile, limit >= 0")
    if args.gamma<=0 or args.clahe_grid<1 or not 0<=args.glare_clip<=255:
        parser.error("Invalid preprocessing parameters")
    if not 0<=args.candidate_conf<=args.keep_conf<=1 or not 0<args.post_nms_iou<1:
        parser.error("Require candidate-conf <= keep-conf and 0 < NMS IoU < 1")
    splits=("train","val","test") if args.mode=="run" else ("train","val")
    entries=read_manifest(args.manifest,splits)
    if args.mode in ("preview","compare"):
        if args.limit==0:
            parser.error("preview/compare need --limit > 0 to prevent accidental large GPU run")
        entries=sample_manifest(entries,args.limit,args.old_teacher)
    if args.mode=="preview":
        baseline_preview(args,entries)
    elif args.mode=="compare":
        compare(args,entries)
    else:
        if len(entries)!=1200:
            raise SystemExit(f"Expected exactly 1200 train/val/test rows, got {len(entries)}")
        run_teacher(args,entries)


if __name__=="__main__":
    parser_main()
