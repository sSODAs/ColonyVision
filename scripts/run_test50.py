from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import final_pipeline as ev
import predict_tiled as pt
from prepare_plate_images import crop_plate, clahe_luma

GT_DIR = ROOT / "evaluation" / "test50"
SELECTION = GT_DIR / "selection.csv"
GT_POINTS = GT_DIR / "ground_truth_points.csv"
DEV_CFG = ROOT / "config" / "final_pipeline.json"
MODEL = ROOT / "weights" / "colonyvision_student_v1.pt"

DEFAULT_OUT = ROOT / "output" / "Test50_Final"
METHODS = ("YOLO_baseline", "Step4_refined", "Final_iterative_resolved")
MAIN_TOL = 0.50


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
        ),
        encoding="utf-8",
    )


def density_band(n: int):
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 100:
        return "51-100"
    if n <= 200:
        return "101-200"
    return ">200"


def aggregate(rows):
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    abs_err = np.asarray([float(r["count_abs_error"]) for r in rows], dtype=float)
    sq_err = np.asarray([float(r["count_error"]) ** 2 for r in rows], dtype=float)
    rel_err = np.asarray([float(r["relative_error_pct"]) for r in rows], dtype=float)

    return {
        "images": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": float(np.mean([float(r["f1"]) for r in rows])),
        "count_mae": float(np.mean(abs_err)),
        "count_rmse": float(np.sqrt(np.mean(sq_err))),
        "mean_relative_count_error_pct": float(np.mean(rel_err)),
    }


def crop_geometry(im: Image.Image, outer_circle, padding: float):
    crop, offset = crop_plate(im, tuple(outer_circle), padding=padding)
    ox, oy = map(float, offset)
    return crop, ox, oy


def shift_circle(circle, ox, oy):
    return (
        float(circle[0]) - ox,
        float(circle[1]) - oy,
        float(circle[2]),
    )


def shift_box(box, ox, oy):
    return [
        float(box[0]) - ox,
        float(box[1]) - oy,
        float(box[2]) - ox,
        float(box[3]) - oy,
    ]


def box_center_radius(box):
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    radius = max(3.0, 0.25 * ((x2 - x1) + (y2 - y1)))
    return cx, cy, radius


def ui_font(size=24):
    candidates = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def rect_area(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def rect_intersection(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def containment_ratio(parent_box, child_box):
    child_area = rect_area(child_box)
    if child_area <= 0:
        return 0.0
    return rect_intersection(parent_box, child_box) / child_area


def log_area_distance(area, typical_area):
    if area <= 0 or typical_area <= 0:
        return float("inf")
    return abs(math.log(area / typical_area))


def iterative_nested_recheck(
    boxes,
    crop,
    offset,
    typical_area,
    max_iter=5,
):
    """Final conservative post-pass for parent/child duplicate geometry.

    - Multiple contained children replace a parent only when the existing image
      evidence confirms distinct lobes/peaks.
    - A single parent/child duplicate is resolved by selecting the box whose
      area is more plausible for the plate's typical colony size.
    - The pass repeats because removing one level can expose another nested
      relationship.
    """
    current = [dict(q) for q in boxes]
    audit = []

    for iteration in range(1, max_iter + 1):
        if len(current) < 2:
            break

        areas = np.asarray([rect_area(q["box"]) for q in current], dtype=float)
        order = np.argsort(-areas)
        removed = set()
        changed = False

        for ii in order:
            i = int(ii)
            if i in removed:
                continue

            parent = current[i]
            parent_box = parent["box"]
            parent_area = areas[i]
            if parent_area <= 0:
                removed.add(i)
                changed = True
                continue

            child_ids = []
            for j, child in enumerate(current):
                if j == i or j in removed:
                    continue
                child_area = areas[j]
                if child_area <= 0 or child_area >= parent_area * 0.92:
                    continue
                if containment_ratio(parent_box, child["box"]) >= 0.82:
                    child_ids.append(j)

            if not child_ids:
                continue

            # One child: change only when both boxes are essentially
            # centered on the same object. Offset centers are left untouched
            # because they can be two genuinely touching colonies.
            if len(child_ids) == 1:
                j = child_ids[0]
                child = current[j]
                child_area = areas[j]
                pb = np.asarray(parent_box, dtype=float)
                cb = np.asarray(child["box"], dtype=float)
                pc = (pb[:2] + pb[2:]) * 0.5
                cc = (cb[:2] + cb[2:]) * 0.5
                parent_min_side = max(1.0, float(np.min(pb[2:] - pb[:2])))
                normalized_center_distance = float(
                    np.linalg.norm(pc - cc) / parent_min_side
                )
                ratio = child_area / max(parent_area, 1e-9)
                containment = containment_ratio(parent_box, child["box"])

                if (
                    ratio <= 0.78
                    and containment >= 0.95
                    and normalized_center_distance <= 0.12
                ):
                    parent_score = log_area_distance(parent_area, typical_area)
                    child_score = log_area_distance(child_area, typical_area)
                    if child_score + 0.10 < parent_score:
                        removed.add(i)
                        action = "keep_child_remove_parent"
                    elif parent_score + 0.10 < child_score:
                        removed.add(j)
                        action = "keep_parent_remove_child"
                    else:
                        continue
                    audit.append(
                        {
                            "iteration": iteration,
                            "action": action,
                            "parent_id": parent.get("id"),
                            "child_ids": [child.get("id")],
                            "normalized_center_distance": normalized_center_distance,
                            "containment": containment,
                            "parent_area": parent_area,
                            "child_areas": [child_area],
                            "typical_area": typical_area,
                        }
                    )
                    changed = True
                continue

            # Multiple children: authorize child replacement only when local
            # image evidence confirms separate lobes. Otherwise prefer parent
            # if children do not jointly look like distinct colonies.
            children = [current[j] for j in child_ids]
            try:
                supported, evidence = pt.nested_image_evidence(
                    crop,
                    offset,
                    {"box": parent_box},
                    [{"box": q["box"]} for q in children],
                )
            except Exception as exc:
                supported = False
                evidence = {"reason": f"evidence_error:{type(exc).__name__}"}

            if supported:
                removed.add(i)
                audit.append(
                    {
                        "iteration": iteration,
                        "action": "keep_children_remove_parent",
                        "parent_id": parent.get("id"),
                        "child_ids": [q.get("id") for q in children],
                        "evidence": evidence,
                    }
                )
                changed = True
            else:
                # Failed/ambiguous multi-lobe evidence is not permission to
                # delete either representation. Leave the group unchanged.
                audit.append(
                    {
                        "iteration": iteration,
                        "action": "review_unchanged",
                        "parent_id": parent.get("id"),
                        "child_ids": [q.get("id") for q in children],
                        "evidence": evidence,
                    }
                )

        if not changed:
            break

        current = [q for idx, q in enumerate(current) if idx not in removed]

    return current, audit


def draw_full_image_crop_circle(full_im, outer_circle):
    out = full_im.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    cx, cy, radius = map(float, outer_circle)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline="#f5a623",
        width=6,
    )
    draw.rectangle((0, 0, min(out.width, 1800), 72), fill="white")
    draw.text(
        (16, 18),
        "Original full image | Orange = crop boundary",
        fill="black",
        font=ui_font(26),
    )
    return out


def draw_roi_ring(draw: ImageDraw.ImageDraw, circle, color="#f5a623", width=5):
    cx, cy, r = map(float, circle)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=width)


def draw_colonies(
    crop: Image.Image,
    boxes,
    ox,
    oy,
    roi_circle=None,
    title=None,
    number_labels=False,
    accepted_color="#17c964",
):
    out = crop.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    if roi_circle is not None:
        draw_roi_ring(draw, shift_circle(roi_circle, ox, oy), "#f5a623", 6)

    # Small AST-style tag: black rectangle with only the sequence number.
    # Drawn larger before final down-scaling so the number remains readable.
    font_label = ui_font(28)
    for idx, item in enumerate(boxes, 1):
        box = item["box"] if isinstance(item, dict) else item
        x1, y1, x2, y2 = shift_box(box, ox, oy)

        # Final visualization uses rectangular colony boxes.
        draw.rectangle((x1, y1, x2, y2), outline=accepted_color, width=3)

        if number_labels:
            label = str(idx)
            tb = draw.textbbox((0, 0), label, font=font_label)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            pad_x, pad_y = 6, 3

            bx1 = max(0, min(out.width - (tw + 2 * pad_x), x1))
            by1 = y1 - th - 2 * pad_y - 3
            if by1 < 0:
                by1 = min(out.height - (th + 2 * pad_y), y1 + 3)
            bx2 = bx1 + tw + 2 * pad_x
            by2 = by1 + th + 2 * pad_y

            draw.rectangle((bx1, by1, bx2, by2), fill="black")
            draw.text(
                (bx1 + pad_x, by1 + pad_y),
                label,
                fill=accepted_color,
                font=font_label,
            )

    if title:
        bar_h = 68
        draw.rectangle((0, 0, min(out.width, 1700), bar_h), fill="white")
        draw.text((16, 18), title, fill="black", font=ui_font(26))

    return out


def draw_crop_roi(crop, outer_circle, eval_circle, ox, oy):
    out = crop.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    draw_roi_ring(draw, shift_circle(outer_circle, ox, oy), "#7b7b7b", 4)
    draw_roi_ring(draw, shift_circle(eval_circle, ox, oy), "#f5a623", 6)
    draw.rectangle((0, 0, min(out.width, 1700), 68), fill="white")
    draw.text(
        (16, 18),
        "Gray = detected plate rim | Orange = colony counting ROI",
        fill="black",
        font=ui_font(24),
    )
    return out


def add_white_title(im: Image.Image, title: str):
    src = im.convert("RGB")
    canvas = Image.new("RGB", (src.width, src.height + 54), "white")
    canvas.paste(src, (0, 54))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 15), title, fill="black", font=ui_font(24))
    return canvas


def save_jpg(im: Image.Image, path: Path, max_side=1500, quality=95):
    out = im.copy().convert("RGB")
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out.save(path, quality=quality)


def make_ast_steps(panel_items, out_path: Path, plate_id: str):
    panels = []
    for title, image in panel_items:
        p = image.copy().convert("RGB")
        p.thumbnail((560, 520), Image.Resampling.LANCZOS)
        panels.append((title, p))

    cols = 4
    rows = math.ceil(len(panels) / cols)
    cell_w = 590
    cell_h = 590
    header_h = 70

    sheet = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (18, 16),
        f"Image: {plate_id} | ColonyVision Pipeline",
        fill="black",
        font=ui_font(30),
    )

    for i, (title, panel) in enumerate(panels):
        col = i % cols
        row = i // cols
        x0 = col * cell_w
        y0 = header_h + row * cell_h
        draw.text((x0 + 10, y0 + 8), title, fill="black", font=ui_font(22))
        x = x0 + (cell_w - panel.width) // 2
        y = y0 + 42 + (cell_h - 52 - panel.height) // 2
        sheet.paste(panel, (x, y))

    sheet.save(out_path, quality=95)


def make_plate_stage_graph(plate_id, counts, out_path):
    import matplotlib.pyplot as plt

    labels = ["YOLO", "Refined", "Lobe/Touching", "Seeded", "Re-check"]
    values = [
        counts["yolo"],
        counts["refined"],
        counts["lobe"],
        counts["seeded"],
        counts["final"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(labels, values, marker="o", linewidth=2.2)
    for x, y in zip(labels, values):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_title(f"{plate_id} | Detection count through pipeline", loc="left")
    ax.set_ylabel("Detected colonies")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def make_eval_graphs(eval_dir: Path, aggregate_rows, density_rows, per_image):
    import matplotlib.pyplot as plt

    graph_dir = eval_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    method_labels = ["YOLO baseline", "Refined", "Final"]
    x = np.arange(len(method_labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    for i, key in enumerate(("precision", "recall", "f1")):
        vals = [100 * float(aggregate_rows[j][key]) for j in range(3)]
        ax.bar(x + (i - 1) * width, vals, width, label=key.capitalize())
    ax.set_xticks(x, method_labels)
    ax.set_ylim(70, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Test50 | Detection performance", loc="left")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(graph_dir / "01_precision_recall_f1.png", dpi=180, facecolor="white")
    plt.close(fig)

    final_rows = [r for r in per_image if r["method"] == "Final_iterative_resolved"]
    gt = np.asarray([float(r["gt_in_roi"]) for r in final_rows])
    pred = np.asarray([float(r["pred_count"]) for r in final_rows])
    mx = max(float(gt.max()), float(pred.max())) * 1.03

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(gt, pred, s=38)
    ax.plot([0, mx], [0, mx], linestyle="--")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel("Ground Truth count")
    ax.set_ylabel("Predicted count")
    ax.set_title("Final model | Ground Truth vs Predicted", loc="left")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(graph_dir / "02_ground_truth_vs_prediction.png", dpi=180, facecolor="white")
    plt.close(fig)

    bands = ["1-10", "11-50", "51-100", "101-200", ">200"]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    for method, label in zip(METHODS, method_labels):
        vals = []
        for band in bands:
            row = next(
                (r for r in density_rows if r["method"] == method and r["density"] == band),
                None,
            )
            vals.append(100 * float(row["f1"]) if row else np.nan)
        ax.plot(bands, vals, marker="o", linewidth=2, label=label)
    ax.set_ylim(75, 100)
    ax.set_xlabel("Ground Truth colonies in evaluation ROI")
    ax.set_ylabel("F1-score (%)")
    ax.set_title("F1-score by colony density", loc="left")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(graph_dir / "03_f1_by_density.png", dpi=180, facecolor="white")
    plt.close(fig)

    ids = [r["eval_id"] for r in final_rows]
    err = np.asarray([float(r["count_abs_error"]) for r in final_rows])
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(ids, err)
    ax.set_xlabel("Test image")
    ax.set_ylabel("Absolute count error")
    ax.set_title("Final model | Absolute count error per plate", loc="left")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(graph_dir / "04_absolute_count_error.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run ColonyVision Test50 and save AST-style per-plate output."
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Default: output/Test50_Final",
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= 50:
        parser.error("--limit must be 1..50")

    out_root = args.output if args.output.is_absolute() else ROOT / args.output
    eval_dir = out_root / "_evaluation"

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    import torch
    from ultralytics import YOLO

    selection = read_csv(SELECTION)[: args.limit]
    cfg = json.loads(DEV_CFG.read_text(encoding="utf-8-sig"))
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL))

    raw_gt = defaultdict(list)
    for row in read_csv(GT_POINTS):
        if int(row["uncertain"]):
            continue
        raw_gt[row["eval_id"]].append(
            np.asarray([float(row["x"]), float(row["y"])], dtype=float)
        )

    gt_by = {}
    for row in selection:
        sid = row["eval_id"]
        src = Path(row["source"])
        if not src.is_absolute():
            src = ROOT / src
        with Image.open(src) as raw:
            w, h = raw.size
            orientation = int(raw.getexif().get(274, 1) or 1)
        gt_by[sid] = [
            ev.point_to_exif_upright(p[0], p[1], w, h, orientation)
            for p in raw_gt[sid]
        ]

    per_image = []
    summary_rows = []

    print("=" * 72)
    print("ColonyVision Test50 | AST-style output")
    print("Output:", out_root)
    print("=" * 72)

    for index, row in enumerate(selection, 1):
        sid = row["eval_id"]
        src = Path(row["source"])
        if not src.is_absolute():
            src = ROOT / src
        plate_dir = out_root / sid
        debug_dir = plate_dir / f"{sid}_debug"
        plate_dir.mkdir(parents=True)
        debug_dir.mkdir(parents=True)

        with Image.open(src) as f:
            im = ImageOps.exif_transpose(f).convert("RGB")

        z = ev.candidate_pipeline(im, model, device, cfg)

        outer = tuple(z["initial_outer"])
        eval_circle = tuple(z["eval_circle"])
        model_circle = tuple(z["model_circle"])

        crop, ox, oy = crop_geometry(
            im,
            outer,
            padding=float(cfg["crop_padding"]),
        )
        clahe = clahe_luma(
            crop,
            float(cfg["lab_clahe_clip"]),
            int(cfg["clahe_grid"]),
            float(cfg["gamma"]),
            int(cfg["glare_clip"]),
        )

        seeded_boxes = ev.boxes_inside(z["final"], eval_circle)
        seeded_areas = [rect_area(q["box"]) for q in seeded_boxes if rect_area(q["box"]) > 0]
        typical_area = float(np.median(seeded_areas)) if seeded_areas else 1.0

        final_boxes, nested_audit = iterative_nested_recheck(
            seeded_boxes,
            crop,
            (ox, oy),
            typical_area,
            max_iter=5,
        )

        # Main final result: cropped plate + orange ROI + rectangular detections.
        final_result = draw_colonies(
            crop,
            final_boxes,
            ox,
            oy,
            roi_circle=eval_circle,
            title=f"{sid} | Final colonies: {len(final_boxes)}",
            number_labels=True,
            accepted_color="#17c964",
        )
        save_jpg(
            final_result,
            plate_dir / f"{sid}_01_final_result.jpg",
            max_side=1800,
            quality=96,
        )

        # Individual debug panels.
        original_panel = draw_full_image_crop_circle(im, outer)
        roi_panel = draw_crop_roi(crop, outer, eval_circle, ox, oy)
        preprocess_panel = add_white_title(clahe, "CLAHE / preprocessing")

        baseline_panel = draw_colonies(
            crop,
            z["baseline"],
            ox,
            oy,
            roi_circle=eval_circle,
            title=f"YOLO baseline | {len(z['baseline'])} detections",
            accepted_color="#f04444",
        )
        refined_panel = draw_colonies(
            crop,
            z["step4"],
            ox,
            oy,
            roi_circle=model_circle,
            title=f"Refinement / filtering | {len(z['step4'])} kept",
            accepted_color="#f0a020",
        )
        lobe_panel = draw_colonies(
            crop,
            z["lobe_final"],
            ox,
            oy,
            roi_circle=model_circle,
            title=f"Lobe / touching-colony logic | {len(z['lobe_final'])}",
            accepted_color="#8a5cf6",
        )
        seeded_panel = draw_colonies(
            crop,
            seeded_boxes,
            ox,
            oy,
            roi_circle=eval_circle,
            title=f"Seeded-connected | {len(seeded_boxes)} before nested re-check",
            accepted_color="#2d7ff9",
        )
        recheck_panel = draw_colonies(
            crop,
            final_boxes,
            ox,
            oy,
            roi_circle=eval_circle,
            title=f"Iterative nested re-check | {len(seeded_boxes)} -> {len(final_boxes)}",
            accepted_color="#17c964",
        )
        final_panel = draw_colonies(
            crop,
            final_boxes,
            ox,
            oy,
            roi_circle=eval_circle,
            title=f"Final accepted colonies | {len(final_boxes)}",
            number_labels=True,
            accepted_color="#17c964",
        )

        debug_panels = [
            ("01_original_full_crop_circle.jpg", original_panel),
            ("02_cropped_plate_roi.jpg", roi_panel),
            ("03_preprocess.jpg", preprocess_panel),
            ("04_yolo_baseline.jpg", baseline_panel),
            ("05_refined.jpg", refined_panel),
            ("06_lobe_touching.jpg", lobe_panel),
            ("07_seeded_before_recheck.jpg", seeded_panel),
            ("08_iterative_nested_recheck.jpg", recheck_panel),
            ("09_final.jpg", final_panel),
        ]
        for name, panel in debug_panels:
            save_jpg(panel, debug_dir / name, max_side=1600, quality=94)

        # AST-like white step sheet.
        make_ast_steps(
            [
                ("Step 1: Full Image + Crop Boundary", original_panel),
                ("Step 2: Cropped Plate + Counting ROI", roi_panel),
                ("Step 3: Preprocessing / CLAHE", preprocess_panel),
                ("Step 4: YOLO Baseline", baseline_panel),
                ("Step 5: Refinement / Filtering", refined_panel),
                ("Step 6: Lobe / Touching Logic", lobe_panel),
                ("Step 7: Seeded-Connected", seeded_panel),
                ("Step 8: Iterative Nested Re-check / Final", final_panel),
            ],
            plate_dir / f"{sid}_02_steps.jpg",
            sid,
        )

        # Per-plate white graph: real pipeline counts.
        stage_counts = {
            "yolo": len(ev.boxes_inside(z["baseline"], eval_circle)),
            "refined": len(ev.boxes_inside(z["step4"], eval_circle)),
            "lobe": len(ev.boxes_inside(z["lobe_final"], eval_circle)),
            "seeded": len(seeded_boxes),
            "final": len(final_boxes),
        }
        make_plate_stage_graph(
            sid,
            stage_counts,
            plate_dir / f"{sid}_03_pipeline_graph.png",
        )

        gt_all = gt_by[sid]
        gt = [p for p in gt_all if ev.inside_pt(p, eval_circle)]
        ref_d = (
            ev.typical_diameter(z["baseline"], eval_circle)
            or ev.typical_diameter(z["step4"], eval_circle)
            or ev.typical_diameter(final_boxes, eval_circle)
            or 0.06 * eval_circle[2]
        )

        method_boxes = {
            "YOLO_baseline": z["baseline"],
            "Step4_refined": z["step4"],
            "Final_iterative_resolved": final_boxes,
        }

        final_metric = None
        method_metrics = {}
        for method, boxes0 in method_boxes.items():
            boxes = ev.boxes_inside(boxes0, eval_circle)
            pred_pts = [ev.center(q["box"]) for q in boxes]
            metric = ev.metrics_one(gt, pred_pts, MAIN_TOL * ref_d)

            gt_count = len(gt)
            pred_count = len(pred_pts)
            error = pred_count - gt_count

            metric_row = {
                "eval_id": sid,
                "species": row["species_filename_hint"],
                "density": density_band(gt_count),
                "method": method,
                "gt_in_roi": gt_count,
                "pred_count": pred_count,
                "count_error": error,
                "count_abs_error": abs(error),
                "relative_error_pct": abs(error) / gt_count * 100 if gt_count else 0.0,
                "tp": metric["tp"],
                "fp": metric["fp"],
                "fn": metric["fn"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
            }
            per_image.append(metric_row)
            method_metrics[method] = metric_row
            if method == "Final_iterative_resolved":
                final_metric = metric_row

        measured = {
            "plate_id": sid,
            "source_image": row["image"],
            "species": row["species_filename_hint"],
            "final_count": len(final_boxes),
            "counting_roi": {
                "center_x": eval_circle[0] - ox,
                "center_y": eval_circle[1] - oy,
                "radius": eval_circle[2],
                "coordinate_frame": "cropped_plate",
            },
            "pipeline_counts": stage_counts,
            "nested_recheck": {
                "typical_area_px2": typical_area,
                "before_count": len(seeded_boxes),
                "after_count": len(final_boxes),
                "actions": nested_audit,
            },
            "final_colonies": [
                {
                    "id": i,
                    "center_x": box_center_radius(shift_box(q["box"], ox, oy))[0],
                    "center_y": box_center_radius(shift_box(q["box"], ox, oy))[1],
                    "radius": box_center_radius(shift_box(q["box"], ox, oy))[2],
                    "box_xyxy": shift_box(q["box"], ox, oy),
                }
                for i, q in enumerate(final_boxes, 1)
            ],
            "evaluation": {
                "ground_truth_count_in_roi": final_metric["gt_in_roi"],
                "prediction_count": final_metric["pred_count"],
                "absolute_count_error": final_metric["count_abs_error"],
                "precision": final_metric["precision"],
                "recall": final_metric["recall"],
                "f1": final_metric["f1"],
            },
        }
        write_json(plate_dir / f"{sid}_04_measured.json", measured)

        summary_rows.append(
            {
                "plate_id": sid,
                "ground_truth": final_metric["gt_in_roi"],
                "prediction": final_metric["pred_count"],
                "absolute_error": final_metric["count_abs_error"],
                "precision": final_metric["precision"],
                "recall": final_metric["recall"],
                "f1": final_metric["f1"],
            }
        )
        write_csv(eval_dir / "per_plate_summary.csv", summary_rows)
        write_csv(eval_dir / "per_image_metrics.csv", per_image)

        print(
            f"[{index:02d}/{len(selection):02d}] {sid} "
            f"FINAL={len(final_boxes)} "
            f"F1={final_metric['f1']:.4f}"
        )

    aggregate_rows = []
    for method in METHODS:
        rows = [r for r in per_image if r["method"] == method]
        aggregate_rows.append({"method": method, **aggregate(rows)})
    write_csv(eval_dir / "aggregate_metrics.csv", aggregate_rows)

    density_rows = []
    for method in METHODS:
        method_rows = [r for r in per_image if r["method"] == method]
        for band in ("1-10", "11-50", "51-100", "101-200", ">200"):
            rows = [r for r in method_rows if r["density"] == band]
            if rows:
                density_rows.append(
                    {"method": method, "density": band, **aggregate(rows)}
                )
    write_csv(eval_dir / "metrics_by_density.csv", density_rows)

    make_eval_graphs(eval_dir, aggregate_rows, density_rows, per_image)

    final = next(
        r for r in aggregate_rows if r["method"] == "Final_iterative_resolved"
    )
    summary_text = (
        "ColonyVision Test50 - Final Evaluation\n\n"
        f"Images: {len(selection)}\n\n"
        f"Precision: {final['precision']*100:.2f}%\n"
        f"Recall: {final['recall']*100:.2f}%\n"
        f"F1-score: {final['f1']*100:.2f}%\n"
        f"Count MAE: {final['count_mae']:.2f} colonies/plate\n"
        f"Count RMSE: {final['count_rmse']:.2f}\n\n"
        "Each Txxx folder contains:\n"
        "  Txxx_01_final_result.jpg\n"
        "  Txxx_02_steps.jpg\n"
        "  Txxx_03_pipeline_graph.png\n"
        "  Txxx_04_measured.json\n"
        "  Txxx_debug/\n"
    )
    (eval_dir / "SUMMARY.txt").write_text(summary_text, encoding="utf-8")

    print("=" * 72)
    print("DONE")
    print(f"Precision = {final['precision']*100:.2f}%")
    print(f"Recall    = {final['recall']*100:.2f}%")
    print(f"F1        = {final['f1']*100:.2f}%")
    print(f"Count MAE = {final['count_mae']:.2f}")
    print("OUTPUT:", out_root)
    print("=" * 72)


if __name__ == "__main__":
    main()
