from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

import detection as pt
from postprocessing import touching as lgr
from postprocessing import connected as scr
from preprocessing.plate import locate_plate, crop_plate, clahe_luma


def center(box):
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], float)


def inside_pt(point, circle):
    return (
        (point[0] - circle[0]) ** 2 + (point[1] - circle[1]) ** 2
        <= circle[2] ** 2
    )


def point_to_exif_upright(x, y, w, h, orientation):
    x, y, w, h = map(float, (x, y, w, h))
    if orientation == 1:
        return np.array([x, y], float)
    if orientation == 2:
        return np.array([w - 1 - x, y], float)
    if orientation == 3:
        return np.array([w - 1 - x, h - 1 - y], float)
    if orientation == 4:
        return np.array([x, h - 1 - y], float)
    if orientation == 5:
        return np.array([y, x], float)
    if orientation == 6:
        return np.array([h - 1 - y, x], float)
    if orientation == 7:
        return np.array([h - 1 - y, w - 1 - x], float)
    if orientation == 8:
        return np.array([y, w - 1 - x], float)
    return np.array([x, y], float)


def boxes_inside(boxes, circle):
    return [
        q
        for q in boxes
        if inside_pt(center(q["box"] if isinstance(q, dict) else q), circle)
    ]


def typical_diameter(boxes, circle):
    selected = boxes_inside(boxes, circle)
    if not selected:
        return None

    wh = []
    for q in selected:
        box = np.asarray(q["box"] if isinstance(q, dict) else q, float)
        w, h = box[2] - box[0], box[3] - box[1]
        if w > 0 and h > 0 and math.isfinite(w * h):
            wh.append((w, h))

    if not wh:
        return None

    values = np.asarray(wh, float)
    return float(
        math.sqrt(
            float(np.median(values[:, 0])) * float(np.median(values[:, 1]))
        )
    )


def max_cardinality_match(gt_pts, pred_pts, tolerance):
    n, m = len(gt_pts), len(pred_pts)
    adjacency = []

    for gt in gt_pts:
        candidates = []
        for j, pred in enumerate(pred_pts):
            distance = float(np.linalg.norm(gt - pred))
            if distance <= tolerance:
                candidates.append((distance, j))
        candidates.sort()
        adjacency.append([j for _, j in candidates])

    pair_u = [-1] * n
    pair_v = [-1] * m
    distance_layer = [0] * n
    infinity = 10**9

    def bfs():
        queue = deque()
        found = False
        for u in range(n):
            if pair_u[u] < 0:
                distance_layer[u] = 0
                queue.append(u)
            else:
                distance_layer[u] = infinity
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                u2 = pair_v[v]
                if u2 < 0:
                    found = True
                elif distance_layer[u2] == infinity:
                    distance_layer[u2] = distance_layer[u] + 1
                    queue.append(u2)
        return found

    def dfs(u):
        for v in adjacency[u]:
            u2 = pair_v[v]
            if u2 < 0 or (
                distance_layer[u2] == distance_layer[u] + 1 and dfs(u2)
            ):
                pair_u[u] = v
                pair_v[v] = u
                return True
        distance_layer[u] = infinity
        return False

    while bfs():
        for u in range(n):
            if pair_u[u] < 0:
                dfs(u)

    matches = [
        (u, v, float(np.linalg.norm(gt_pts[u] - pred_pts[v])))
        for u, v in enumerate(pair_u)
        if v >= 0
    ]
    return matches, pair_u, pair_v


def metrics_one(gt_pts, pred_pts, tolerance):
    matches, pair_u, pair_v = max_cardinality_match(gt_pts, pred_pts, tolerance)
    tp = len(matches)
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gt_pts else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "pair_u": pair_u,
        "pair_v": pair_v,
    }


def candidate_pipeline(image, model, device, cfg):
    outer, roi_source = locate_plate(image)
    if outer is None:
        raise RuntimeError("plate ROI not found")

    inner = (
        float(outer[0]),
        float(outer[1]),
        float(outer[2]) * float(cfg["roi_ratio"]),
    )
    eval_circle = inner

    crop, offset = crop_plate(
        image, outer, padding=float(cfg["crop_padding"])
    )
    enhanced = clahe_luma(
        crop,
        float(cfg["lab_clahe_clip"]),
        int(cfg["clahe_grid"]),
        float(cfg["gamma"]),
        int(cfg["glare_clip"]),
    )

    ab = []
    for view, view_image in (("original", crop), ("clahe", enhanced)):
        raw = []
        kept, n = pt.infer(
            model,
            view_image,
            int(cfg["tile"]),
            int(cfg["stride"]),
            float(cfg["confidence"]),
            int(cfg["imgsz"]),
            float(cfg["global_nms_iou"]),
            device,
            pil_input=True,
            audit=raw,
        )
        if len(raw) != n or sum(q["global_nms_kept"] for q in raw) != len(kept):
            raise RuntimeError("audit mismatch " + view)
        ab.extend(pt.coverage_records(raw, offset, inner, view))

    extra = []
    for view, confidence in (("local_enhance", 0.25), ("local_enhance_low", 0.15)):
        raw = []
        kept, n = pt.infer(
            model,
            crop,
            768,
            640,
            confidence,
            int(cfg["imgsz"]),
            float(cfg["global_nms_iou"]),
            device,
            pil_input=True,
            audit=raw,
            patch_transform=pt.enhance_tile,
        )
        if len(raw) != n or sum(q["global_nms_kept"] for q in raw) != len(kept):
            raise RuntimeError("enhance audit mismatch " + view)
        extra.append(
            [
                dict(
                    q,
                    tile_size=768,
                    stride=640,
                    confidence_threshold=confidence,
                )
                for q in pt.coverage_records(raw, offset, inner, view)
            ]
        )

    c, d = extra
    pt.tile_enhancement_flags(c, ab, [])
    pt.tile_enhancement_flags(d, ab, c)
    candidates = [
        dict(q, tile_size=q.get("tile_size", 768), stride=q.get("stride", 640))
        for q in ab
    ] + c + d

    baseline = [
        q
        for q in ab
        if q["view"] == "original"
        and q["inside_inner_circle"]
        and q["global_nms_kept"]
    ]

    outer2, circle2, roi_refinement = pt.refinement_refit_roi(
        image, candidates, tuple(outer), tuple(inner)
    )

    recandidates = []
    for q in candidates:
        candidate = dict(q)
        candidate["inside_inner_circle"] = bool(
            inside_pt(center(candidate["box"]), circle2)
        )
        recandidates.append(candidate)

    crop2, offset2 = crop_plate(image, outer2, padding=1.02)
    bounds = (
        *offset2,
        offset2[0] + crop2.width,
        offset2[1] + crop2.height,
    )

    kept, suppressed, records = pt.refine_candidates(recandidates, bounds)
    kept, suppressed, records, _ = pt.refine_residual_fragments(
        kept, suppressed, records
    )
    kept, suppressed, records, _ = pt.refine_nested_parents(
        kept, suppressed, records, crop2, offset2
    )
    kept, suppressed, records, _ = pt.e3_finalize(
        image, circle2, kept, records
    )

    gray = np.asarray(image.convert("L"))
    lobe_final, _, lobe_debug, _, _, _ = lgr.resolve_lobe_graph(
        gray, kept, circle2, outer2
    )
    frozen_base = [
        q for q in lobe_final if q.get("resolution") != "counted_split"
    ]
    final_boxes, seed_debug, _ = scr.resolve_seeded_connected(
        gray,
        kept,
        circle2,
        outer2,
        frozen_base_boxes=frozen_base,
        frozen_old_debug=lobe_debug,
    )

    return {
        "eval_circle": list(map(float, eval_circle)),
        "initial_outer": list(map(float, outer)),
        "model_circle": list(map(float, circle2)),
        "model_outer": list(map(float, outer2)),
        "roi_source": roi_source,
        "roi_refinement": roi_refinement,
        "baseline": baseline,
        "step4": kept,
        "lobe_final": lobe_final,
        "final": final_boxes,
        "lobe_debug": lobe_debug,
        "seeded_debug": seed_debug,
    }

# Test50 runner dependencies.
import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from evaluation import aggregate, density_band, make_eval_graphs
from postprocessing.nested import iterative_nested_recheck, rect_area
from visualization import (
    add_white_title,
    box_center_radius,
    crop_geometry,
    draw_colonies,
    draw_crop_roi,
    draw_final_count,
    draw_full_image_crop_circle,
    make_ast_steps,
    make_plate_stage_graph,
    save_jpg,
    shift_box,
)

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "evaluation" / "test50"
SELECTION = GT_DIR / "selection.csv"
GT_POINTS = GT_DIR / "ground_truth_points.csv"
DEV_CFG = ROOT / "config" / "config.json"
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
            point_to_exif_upright(p[0], p[1], w, h, orientation)
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

        z = candidate_pipeline(im, model, device, cfg)

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

        seeded_boxes = boxes_inside(z["final"], eval_circle)
        seeded_areas = [rect_area(q["box"]) for q in seeded_boxes if rect_area(q["box"]) > 0]
        typical_area = float(np.median(seeded_areas)) if seeded_areas else 1.0

        final_boxes, nested_audit = iterative_nested_recheck(
            seeded_boxes,
            crop,
            (ox, oy),
            typical_area,
            max_iter=5,
        )

        # Main final result: cropped plate + rectangular detections only.
        # ROI is intentionally hidden here; it remains visible in step-by-step output.
        final_result = draw_colonies(
            crop,
            final_boxes,
            ox,
            oy,
            roi_circle=None,
            title=None,
            number_labels=True,
            accepted_color="#17c964",
        )
        final_result = draw_final_count(final_result, len(final_boxes))
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
            "yolo": len(boxes_inside(z["baseline"], eval_circle)),
            "refined": len(boxes_inside(z["step4"], eval_circle)),
            "lobe": len(boxes_inside(z["lobe_final"], eval_circle)),
            "seeded": len(seeded_boxes),
            "final": len(final_boxes),
        }
        make_plate_stage_graph(
            sid,
            stage_counts,
            plate_dir / f"{sid}_03_pipeline_graph.png",
        )

        gt_all = gt_by[sid]
        gt = [p for p in gt_all if inside_pt(p, eval_circle)]
        ref_d = (
            typical_diameter(z["baseline"], eval_circle)
            or typical_diameter(z["step4"], eval_circle)
            or typical_diameter(final_boxes, eval_circle)
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
            boxes = boxes_inside(boxes0, eval_circle)
            pred_pts = [center(q["box"]) for q in boxes]
            metric = metrics_one(gt, pred_pts, MAIN_TOL * ref_d)

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
