from __future__ import annotations

import math
from collections import deque

import numpy as np

import predict_tiled as pt
import lobe_graph_resolver as lgr
import seeded_connected_resolver as scr
from prepare_plate_images import locate_plate, crop_plate, clahe_luma


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
