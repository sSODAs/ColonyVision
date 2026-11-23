"""Lobe-graph colony completion prototype.

Existing Step-4 detections are LOCKED and never deleted by this module.

The module discovers missing touching colonies by:
  1. building a whole-plate local foreground once,
  2. extracting distance-transform lobe maxima once,
  3. assigning existing Step-4 detections to lobe nodes,
  4. connecting nearby lobes only when the foreground path has a short gap and
     a real neck/saddle,
  5. recursively walking from locked nodes to unrepresented valid lobe nodes.

This avoids mutable ROI expansion and avoids component-area counting on giant
dense components.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np

import colony_evidence as base


@dataclass
class LobeGraphConfig:
    # Foreground response.
    response_fraction: float = 0.17
    response_floor: float = 1.5
    close_fraction: float = 0.045

    # Lobe candidate extraction.
    peak_kernel_fraction: float = 0.38
    peak_radius_min_fraction: float = 0.10
    peak_radius_max_fraction: float = 0.70
    peak_nms_fraction: float = 0.43

    # Existing seed ownership.
    seed_snap_fraction: float = 0.55
    represented_fraction: float = 0.50

    # Graph edge.
    edge_distance_min_fraction: float = 0.48
    edge_distance_max_fraction: float = 1.32
    edge_gap_max_fraction: float = 0.22
    edge_saddle_max: float = 0.80

    # New lobe validation.
    signal_min: float = 6.0
    filled_fraction_min: float = 0.38
    core_min: float = 3.5
    rim_band_fraction: float = 1.55


def _center(box):
    b = np.asarray(box, dtype=float)
    return (b[:2] + b[2:]) * 0.5


def _clip_box(box, w, h):
    x1, y1, x2, y2 = map(float, box)
    return [
        max(0.0, min(float(w - 1), x1)),
        max(0.0, min(float(h - 1), y1)),
        max(1.0, min(float(w), x2)),
        max(1.0, min(float(h), y2)),
    ]


def _inside_circle_mask(shape, circle, margin=0.0):
    h, w = shape[:2]
    cx, cy, r = map(float, circle)
    r = max(0.0, r - margin)
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r



def _consensus_detector_review_support(q, ev):
    """Admit a review box when several detector views agree on a colony core.

    This path is intentionally independent of component_fraction because dense
    colonies can merge into very small or very large connected components.
    """
    radial = ev.get("radial", {})
    return bool(
        not ev.get("artifact", False)
        and float(q.get("confidence", 0.0)) >= 0.75
        and len(q.get("supporting_ids", [])) >= 3
        and ev.get("signal", 0.0) >= 7.0
        and radial.get("kind") != "reject"
        and radial.get("core", 0.0) >= 3.5
        and radial.get("filled_fraction", 0.0) >= 0.40
        and radial.get("aspect", 0.0) >= 0.75
        and radial.get("ring_fraction", 1.0) <= 0.15
    )



def _interior_consensus_review_support(q, ev, circle):
    """Second consensus path for true colonies missed by dense-component rules.

    It is restricted to the interior of the dish so strong rim/glass detections
    cannot enter through this path.
    """
    b = np.asarray(q["box"], dtype=float)
    c = _center(b)
    cx, cy, cr = map(float, circle)
    radial_position = math.hypot(
        float(c[0]) - cx,
        float(c[1]) - cy,
    ) / max(cr, 1e-6)
    if radial_position > 0.88:
        return False

    radial = ev.get("radial", {})
    return bool(
        not ev.get("artifact", False)
        and float(q.get("confidence", 0.0)) >= 0.60
        and len(q.get("supporting_ids", [])) >= 3
        and ev.get("signal", 0.0) >= 5.0
        and radial.get("kind") != "reject"
        and radial.get("core", 0.0) >= 2.0
        and radial.get("filled_fraction", 0.0) >= 0.30
        and radial.get("aspect", 0.0) >= 0.65
        and radial.get("ring_fraction", 1.0) <= 0.15
    )



def _interior_highconf_review_support(q, ev, circle):
    """Recall-preserving final admission path for an existing detector box.

    This path never creates a new detection. It only prevents post-processing
    from discarding a high-confidence Step-4 box deep inside the dish when the
    local appearance is at least colony-compatible.
    """
    b = np.asarray(q["box"], dtype=float)
    c = _center(b)
    cx, cy, cr = map(float, circle)
    radial_position = math.hypot(
        float(c[0]) - cx,
        float(c[1]) - cy,
    ) / max(cr, 1e-6)
    if radial_position > 0.85:
        return False

    radial = ev.get("radial", {})
    return bool(
        not ev.get("artifact", False)
        and float(q.get("confidence", 0.0)) >= 0.60
        and ev.get("signal", 0.0) >= 4.0
        and radial.get("kind") != "reject"
        and radial.get("core", 0.0) >= 1.5
        and radial.get("filled_fraction", 0.0) >= 0.25
        and radial.get("aspect", 0.0) >= 0.55
        and radial.get("ring_fraction", 1.0) <= 0.15
    )


def _dense_touching_review_support(q, ev):
    """Second path for a detector-supported colony embedded in dense growth.

    The ordinary evidence() intentionally rejects a connected bright component
    when component_fraction > 1.40. That is useful for isolated-object filtering
    but wrong for touching colonies. Here we require multi-view detector support
    plus filled colony morphology, and interpret an oversized component as
    'touching/dense' rather than 'not a colony'.
    """
    radial = ev.get("radial", {})
    component_fraction = float(ev.get("component_fraction", 0.0))
    return bool(
        not ev.get("artifact", False)
        and len(q.get("supporting_ids", [])) >= 2
        and float(q.get("confidence", 0.0)) >= 0.45
        and ev.get("signal", 0.0) >= 9.0
        and radial.get("kind") != "reject"
        and radial.get("core", 0.0) >= 6.0
        and radial.get("filled_fraction", 0.0) >= 0.55
        and radial.get("aspect", 0.0) >= 0.62
        and radial.get("ring_drop", 99.0) < 10.0
        and 1.10 <= component_fraction <= 3.20
    )



def _detector_consensus_peak_support(
    gray,
    q,
    ev,
    circle,
    outer_circle,
    typical,
):
    """Recall rescue for an existing Step-4 box using detector consensus + image peak.

    This path never invents a new detection. It only protects an existing
    multi-view detector box when dense/irregular morphology makes the generic
    evidence.support boolean too strict.
    """
    radial = ev.get("radial", {})
    b = np.asarray(q["box"], dtype=float)
    wh = np.maximum(1.0, b[2:] - b[:2])
    box_aspect = float(min(wh) / max(wh))
    box_area_ratio = float(
        np.prod(wh) / max(1.0, float(np.prod(typical)))
    )
    if not (
        not ev.get("artifact", False)
        and box_aspect >= 0.55
        and 0.35 <= box_area_ratio <= 1.80
        and float(q.get("confidence", 0.0)) >= 0.55
        and len(q.get("supporting_ids", [])) >= 3
        and ev.get("signal", 0.0) >= 10.0
        and radial.get("kind") != "reject"
        and radial.get("core", 0.0) >= 4.0
        and radial.get("filled_fraction", 0.0) >= 0.45
        and radial.get("aspect", 0.0) >= 0.30
        and radial.get("ring_fraction", 1.0) <= 0.55
        and 0.12 <= float(ev.get("component_fraction", 0.0)) <= 2.60
    ):
        return False, None

    centers, diag = base._strong_local_centers(
        gray,
        q["box"],
        typical,
        circle,
        outer_circle,
        expand=0.02,
    )
    if not centers:
        return False, diag

    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    qc = _center(q["box"])
    nearest = min(
        float(np.linalg.norm(np.asarray(c["pt"], dtype=float) - qc))
        for c in centers
    )
    ok = bool(
        int(diag.get("nms_peak_count", 0)) >= 1
        and nearest <= 0.55 * side
    )
    return ok, diag


def _trusted_rows(gray, rows, circle, outer_circle):
    """Lock Step-4 detections except deterministic near-white glare.

    Review detections have three admission paths:
      1) ordinary local evidence support;
      2) dense-touching support backed by >=2 detector views;
      3) detector-consensus rescue: high-confidence multi-view detection with
         a real local image peak, even when dense morphology is irregular.

    The later same-lobe checker remains responsible for duplicate removal.
    """
    trusted = []
    review_rejected = []
    specular_rejected = []
    typical = base.typical_geometry(rows)

    for row in rows:
        q = dict(row)

        spec = _specular_highlight_artifact(gray, q["box"])
        q["_specular"] = spec
        if spec.get("is_artifact", False):
            q["_seed_status"] = "specular_filtered"
            specular_rejected.append(q)
            continue

        ev = base.evidence(gray, q, circle)
        reason = base._hard_artifact_reason(
            gray, q, circle, outer_circle, ev
        )
        q["_evidence"] = ev
        q["_artifact_reason"] = reason

        if not q.get("review_needed", False):
            q["_seed_status"] = "accepted_step4_locked"
            trusted.append(q)
            continue

        radial = ev.get("radial", {})
        ordinary_ok = bool(
            reason is None
            and ev.get("support", False)
            and ev.get("signal", 0.0) >= 7.0
            and radial.get("kind") != "reject"
            and radial.get("filled_fraction", 0.0) >= 0.40
        )
        consensus_ok = bool(
            reason is None
            and _consensus_detector_review_support(q, ev)
        )
        interior_consensus_ok = bool(
            reason is None
            and _interior_consensus_review_support(q, ev, circle)
        )
        interior_highconf_ok = bool(
            reason is None
            and _interior_highconf_review_support(q, ev, circle)
        )
        dense_ok = bool(
            reason is None
            and _dense_touching_review_support(q, ev)
        )
        consensus_peak_ok, consensus_peak_diag = _detector_consensus_peak_support(
            gray,
            q,
            ev,
            circle,
            outer_circle,
            typical,
        )
        consensus_peak_ok = bool(
            reason is None
            and consensus_peak_ok
        )

        if ordinary_ok:
            q["_seed_status"] = "review_step4_locked"
            trusted.append(q)
        elif consensus_ok:
            q["_seed_status"] = "consensus_detector_seed"
            trusted.append(q)
        elif interior_consensus_ok:
            q["_seed_status"] = "interior_consensus_seed"
            trusted.append(q)
        elif interior_highconf_ok:
            q["_seed_status"] = "interior_highconf_seed"
            trusted.append(q)
        elif dense_ok:
            q["_seed_status"] = "dense_touching_seed"
            trusted.append(q)
        elif consensus_peak_ok:
            q["_seed_status"] = "detector_consensus_peak_seed"
            q["_consensus_peak_diag"] = consensus_peak_diag
            trusted.append(q)
        else:
            q["_seed_status"] = "review_not_locked"
            review_rejected.append(q)

    return trusted, review_rejected, specular_rejected


def _foreground(gray, trusted, typical, circle, cfg):
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    f = gray.astype(np.float32)
    small = cv2.GaussianBlur(f, (0, 0), max(0.7, 0.055 * side))
    broad = cv2.GaussianBlur(f, (0, 0), max(2.2, 0.58 * side))
    response = np.maximum(0.0, small - broad)

    seed_values = []
    r = max(2, int(round(0.13 * side)))
    h, w = gray.shape[:2]
    for q in trusted:
        c = _center(q["box"])
        x0 = max(0, int(c[0] - r))
        y0 = max(0, int(c[1] - r))
        x1 = min(w, int(c[0] + r + 1))
        y1 = min(h, int(c[1] + r + 1))
        vals = response[y0:y1, x0:x1]
        vals = vals[vals > 0]
        if vals.size:
            seed_values.append(float(np.percentile(vals, 75)))

    if seed_values:
        scale = float(np.median(seed_values))
    else:
        plate = response[_inside_circle_mask(gray.shape, circle)]
        pos = plate[plate > 0]
        scale = float(np.percentile(pos, 75)) if pos.size else 8.0

    threshold = max(cfg.response_floor, cfg.response_fraction * scale)
    mask = (
        (response >= threshold)
        & _inside_circle_mask(gray.shape, circle, margin=0.10 * side)
    ).astype(np.uint8)

    kc = max(3, int(round(cfg.close_fraction * side)))
    if kc % 2 == 0:
        kc += 1
    kc = min(kc, 5)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)),
    )

    # Keep faint trusted centers anchored without bridging a whole colony.
    ar = max(1, int(round(0.06 * side)))
    for q in trusted:
        c = _center(q["box"])
        cv2.circle(
            mask,
            (int(round(c[0])), int(round(c[1]))),
            ar,
            1,
            -1,
        )

    return response, mask, threshold, scale


def _extract_raw_nodes(mask, response, typical, cfg):
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

    kd = max(3, int(round(cfg.peak_kernel_fraction * side)))
    if kd % 2 == 0:
        kd += 1
    kd = min(kd, 31)

    dil = cv2.dilate(dist, np.ones((kd, kd), np.uint8))
    peak = (
        (mask > 0)
        & (dist >= dil - 1e-6)
        & (dist >= cfg.peak_radius_min_fraction * side)
    ).astype(np.uint8)

    nlab, labels, stats, cents = cv2.connectedComponentsWithStats(peak, 8)
    nodes = []
    for lab in range(1, nlab):
        if int(stats[lab, cv2.CC_STAT_AREA]) <= 0:
            continue
        x, y = map(float, cents[lab])
        ix = int(np.clip(round(x), 0, dist.shape[1] - 1))
        iy = int(np.clip(round(y), 0, dist.shape[0] - 1))
        radius = float(dist[iy, ix])
        if radius > cfg.peak_radius_max_fraction * side:
            continue
        nodes.append(
            dict(
                pt=np.asarray([x, y], dtype=float),
                radius=radius,
                response=float(response[iy, ix]),
                raw_valid=True,
                valid_new=False,
                locked=False,
                owners=[],
            )
        )

    # NMS raw maxima once globally.
    nodes.sort(key=lambda n: (n["radius"], n["response"]), reverse=True)
    kept = []
    for node in nodes:
        if all(
            float(np.linalg.norm(node["pt"] - k["pt"]))
            >= cfg.peak_nms_fraction * side
            for k in kept
        ):
            kept.append(node)

    return kept, dist


def _rim_blocked(pt, typical, circle, outer_circle, cfg):
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    cx, cy, r = map(float, circle)
    rim_band = cfg.rim_band_fraction * side
    if outer_circle is not None:
        rim_band = max(
            rim_band,
            0.50 * max(0.0, float(outer_circle[2]) - r),
        )
    return (
        math.hypot(float(pt[0]) - cx, float(pt[1]) - cy)
        > max(0.0, r - rim_band)
    )


def _normal_box(pt, typical, shape):
    h, w = shape[:2]
    tw, th = map(float, typical)
    return _clip_box(
        [
            float(pt[0]) - 0.50 * tw,
            float(pt[1]) - 0.50 * th,
            float(pt[0]) + 0.50 * tw,
            float(pt[1]) + 0.50 * th,
        ],
        w,
        h,
    )



def _specular_highlight_artifact(gray, box):
    """Detect compact near-white clipped glare.

    This is an absolute-brightness gate. It is intentionally narrow enough to
    apply both to Step-4 boxes and to graph-discovered nodes.
    """
    h, w = gray.shape[:2]
    b = np.asarray(box, dtype=float)
    x1 = max(0, int(math.floor(b[0])))
    y1 = max(0, int(math.floor(b[1])))
    x2 = min(w, int(math.ceil(b[2])))
    y2 = min(h, int(math.ceil(b[3])))
    patch = gray[y1:y2, x1:x2]
    if patch.size < 16:
        return dict(is_artifact=False, reason="patch_too_small")

    p95 = float(np.percentile(patch, 95))
    p99 = float(np.percentile(patch, 99))
    vmax = float(patch.max())
    frac245 = float(np.mean(patch >= 245))
    frac250 = float(np.mean(patch >= 250))

    # Real colonies in the current plates are bright but usually not sensor-
    # clipped. Specular reflections contain a compact 245-255 core.
    is_artifact = bool(
        vmax >= 250.0
        and p99 >= 245.0
        and (
            frac245 >= 0.018
            or frac250 >= 0.010
        )
    )
    return dict(
        is_artifact=is_artifact,
        reason=(
            "near_white_specular_highlight"
            if is_artifact
            else "not_specular"
        ),
        max=round(vmax, 2),
        p95=round(p95, 2),
        p99=round(p99, 2),
        fraction_ge_245=round(frac245, 5),
        fraction_ge_250=round(frac250, 5),
    )




def _small_lobe_line_artifact(
    gray,
    node,
    typical,
    median_locked_radius,
    circle,
):
    """Conservative line/rim artifact check for NEW graph nodes only.

    Reject only when a sub-typical lobe is crossed by a long image edge very
    close to the lobe center. This targets dish-rim/scratch structure while
    preserving normal-sized colonies.
    """
    if median_locked_radius is None or median_locked_radius <= 0:
        return dict(is_artifact=False, reason="no_radius_reference")

    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))

    cx, cy, cr = map(float, circle)
    radial_ratio = math.hypot(
        float(node["pt"][0]) - cx,
        float(node["pt"][1]) - cy,
    ) / max(cr, 1e-6)
    if radial_ratio < 0.90:
        return dict(
            is_artifact=False,
            reason="not_near_dish_rim",
            radial_ratio=round(radial_ratio, 4),
        )

    radius = float(node.get("radius", 0.0))
    radius_ratio = radius / median_locked_radius
    if radius_ratio > 0.90:
        return dict(
            is_artifact=False,
            reason="lobe_not_small",
            radius_ratio=round(radius_ratio, 4),
        )

    x, y = map(float, node["pt"])
    h, w = gray.shape[:2]
    R = max(20, int(round(2.2 * side)))
    x1 = max(0, int(math.floor(x - R)))
    y1 = max(0, int(math.floor(y - R)))
    x2 = min(w, int(math.ceil(x + R + 1)))
    y2 = min(h, int(math.ceil(y + R + 1)))
    patch = gray[y1:y2, x1:x2]
    if patch.size < 64:
        return dict(
            is_artifact=False,
            reason="patch_too_small",
            radius_ratio=round(radius_ratio, 4),
        )

    blur = cv2.GaussianBlur(patch, (5, 5), 1.0)
    edges = cv2.Canny(blur, 40, 100)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(12, int(round(0.70 * side))),
        minLineLength=max(15, int(round(1.40 * side))),
        maxLineGap=max(3, int(round(0.15 * side))),
    )

    best_length_units = 0.0
    best_distance_units = 99.0
    if lines is not None:
        px = x - x1
        py = y - y1
        for L in lines[:, 0]:
            xa, ya, xb, yb = map(float, L)
            vx = xb - xa
            vy = yb - ya
            den = vx * vx + vy * vy
            if den <= 1e-9:
                continue
            t = max(
                0.0,
                min(
                    1.0,
                    ((px - xa) * vx + (py - ya) * vy) / den,
                ),
            )
            distance = math.hypot(
                px - (xa + t * vx),
                py - (ya + t * vy),
            )
            length = math.hypot(vx, vy)
            length_units = length / side
            distance_units = distance / side

            if (
                distance_units < best_distance_units
                or (
                    abs(distance_units - best_distance_units) < 1e-9
                    and length_units > best_length_units
                )
            ):
                best_distance_units = distance_units
                best_length_units = length_units

    is_artifact = bool(
        best_length_units >= 1.50
        and best_distance_units <= 0.22
    )
    return dict(
        is_artifact=is_artifact,
        reason=(
            "small_lobe_crossed_by_long_line"
            if is_artifact
            else "not_line_artifact"
        ),
        radius_ratio=round(radius_ratio, 4),
        line_length_units=round(best_length_units, 4),
        line_distance_units=round(best_distance_units, 4),
    )


def _build_box_spatial_index(rows, cell, margin=0.0):
    """Index row boxes into every grid cell touched by box+margin."""
    grid = {}
    cell = max(1.0, float(cell))
    for idx, q in enumerate(rows):
        b = np.asarray(q["box"], dtype=float)
        x1 = int(math.floor((b[0] - margin) / cell))
        y1 = int(math.floor((b[1] - margin) / cell))
        x2 = int(math.floor((b[2] + margin) / cell))
        y2 = int(math.floor((b[3] + margin) / cell))
        for gy in range(y1, y2 + 1):
            for gx in range(x1, x2 + 1):
                grid.setdefault((gx, gy), []).append(idx)
    return grid


def _query_box_spatial_index(grid, pt, cell):
    cell = max(1.0, float(cell))
    gx = int(math.floor(float(pt[0]) / cell))
    gy = int(math.floor(float(pt[1]) / cell))
    return grid.get((gx, gy), [])


def _build_point_spatial_index(points, cell):
    grid = {}
    cell = max(1.0, float(cell))
    for idx, pt in enumerate(points):
        gx = int(math.floor(float(pt[0]) / cell))
        gy = int(math.floor(float(pt[1]) / cell))
        grid.setdefault((gx, gy), []).append(idx)
    return grid


def _query_point_spatial_index(grid, bounds, cell):
    """Return unique point indices inside grid cells touched by bounds."""
    cell = max(1.0, float(cell))
    x1, y1, x2, y2 = map(float, bounds)
    gx1 = int(math.floor(x1 / cell))
    gy1 = int(math.floor(y1 / cell))
    gx2 = int(math.floor(x2 / cell))
    gy2 = int(math.floor(y2 / cell))
    out = set()
    for gy in range(gy1, gy2 + 1):
        for gx in range(gx1, gx2 + 1):
            out.update(grid.get((gx, gy), []))
    return list(out)


def _validate_new_nodes(
    gray,
    nodes,
    ownership_rows,
    typical,
    circle,
    outer_circle,
    cfg,
):
    """Classify lobe nodes independently from Step-4 ownership."""
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    ownership_centers = [_center(q["box"]) for q in ownership_rows]
    ownership_cell = side
    ownership_grid = _build_box_spatial_index(
        ownership_rows,
        ownership_cell,
        margin=1.05 * side,
    )

    for node in nodes:
        box = _normal_box(node["pt"], typical, gray.shape)

        candidate = dict(
            box=box,
            review_needed=True,
            supporting_ids=[],
            tile_edges=[],
        )
        ev = base.evidence(gray, candidate, circle)
        radial = ev.get("radial", {})
        reason = base._hard_artifact_reason(
            gray, candidate, circle, outer_circle, ev
        )

        spec = _specular_highlight_artifact(gray, box)

        node["box"] = box
        node["evidence"] = ev
        node["artifact_reason"] = reason
        node["specular"] = spec
        node["colony_like"] = bool(
            not spec.get("is_artifact", False)
            and reason is None
            and ev.get("support", False)
            and ev.get("signal", 0.0) >= cfg.signal_min
            and radial.get("kind") != "reject"
            and radial.get("filled_fraction", 0.0)
            >= cfg.filled_fraction_min
            and radial.get("core", 0.0) >= cfg.core_min
        )

        # Every Step-4 ROI owns its location, even a review ROI. This prevents
        # completion logic from creating a second box where detection already
        # exists. Review recovery is handled separately below.
        owned = False
        candidate_owner_indices = _query_box_spatial_index(
            ownership_grid,
            node["pt"],
            ownership_cell,
        )
        for owner_index in candidate_owner_indices:
            q = ownership_rows[owner_index]
            c = ownership_centers[owner_index]
            b = np.asarray(q["box"], dtype=float)

            if float(np.linalg.norm(node["pt"] - c)) <= 0.58 * side:
                owned = True
                break

            if (
                b[0] <= node["pt"][0] <= b[2]
                and b[1] <= node["pt"][1] <= b[3]
            ):
                owned = True
                break

            if base._box_ios(box, b) >= 0.28:
                owned = True
                break

        node["owned_by_step4"] = owned
        node["rim_blocked"] = _rim_blocked(
            node["pt"], typical, circle, outer_circle, cfg
        )
        node["valid_new"] = bool(
            node["colony_like"]
            and not owned
            and not node["rim_blocked"]
        )

        if node["specular"].get("is_artifact", False):
            node["reject_reason"] = "near_white_specular_highlight"
        elif node["rim_blocked"]:
            node["reject_reason"] = "rim_owned"
        elif owned:
            node["reject_reason"] = "step4_owned"
        elif not node["colony_like"]:
            node["reject_reason"] = reason or "not_colony_like"
        else:
            node["reject_reason"] = None


def _point_rect_gap(pt, box):
    x, y = map(float, pt)
    x1, y1, x2, y2 = map(float, box)
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _recover_review_rows(
    review_rows,
    nodes,
    accepted_rows,
    typical,
):
    """Recover a Step-4 review ROI when a real lobe exists inside that ROI.

    The original Step-4 geometry is preserved. A nearby lobe already owned by an
    accepted ROI does not recover the review box, avoiding duplicate recovery.
    """
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    accepted_centers = [_center(q["box"]) for q in accepted_rows]

    node_points = [np.asarray(node["pt"], dtype=float) for node in nodes]
    node_cell = side
    node_grid = _build_point_spatial_index(node_points, node_cell)

    accepted_cell = side
    accepted_grid = _build_box_spatial_index(
        accepted_rows,
        accepted_cell,
        margin=0.55 * side,
    )

    recovered = []
    remaining = []
    audit = []

    for q in review_rows:
        b = np.asarray(q["box"], dtype=float)
        qc = _center(b)

        candidates = []
        search_bounds = [
            b[0] - 0.10 * side,
            b[1] - 0.10 * side,
            b[2] + 0.10 * side,
            b[3] + 0.10 * side,
        ]
        candidate_node_indices = _query_point_spatial_index(
            node_grid,
            search_bounds,
            node_cell,
        )
        for idx in candidate_node_indices:
            node = nodes[idx]
            if not node.get("colony_like", False):
                continue
            p = node_points[idx]

            inside = bool(
                b[0] <= p[0] <= b[2]
                and b[1] <= p[1] <= b[3]
            )
            gap = _point_rect_gap(p, b)
            if not inside and gap > 0.10 * side:
                continue

            # Do not recover a review box from an accepted neighbour's lobe.
            owned_by_accepted = False
            candidate_accepted_indices = _query_box_spatial_index(
                accepted_grid,
                p,
                accepted_cell,
            )
            for accepted_index in candidate_accepted_indices:
                aq = accepted_rows[accepted_index]
                ac = accepted_centers[accepted_index]
                ab = np.asarray(aq["box"], dtype=float)
                if float(np.linalg.norm(p - ac)) <= 0.46 * side:
                    owned_by_accepted = True
                    break
                if (
                    ab[0] <= p[0] <= ab[2]
                    and ab[1] <= p[1] <= ab[3]
                ):
                    owned_by_accepted = True
                    break
            if owned_by_accepted:
                continue

            candidates.append(
                (
                    float(np.linalg.norm(p - qc)),
                    -float(node.get("radius", 0.0)),
                    idx,
                )
            )

        if not candidates:
            remaining.append(q)
            continue

        node_id = min(candidates)[2]
        node = nodes[node_id]
        q2 = dict(q)
        q2["_seed_status"] = "review_recovered_by_lobe"
        q2["_recovered_node_id"] = int(node_id)
        recovered.append(q2)
        audit.append(
            dict(
                id=q.get("id"),
                action="review_recovered_by_lobe",
                box=list(map(float, q["box"])),
                node_id=int(node_id),
                node_pt=[
                    round(float(v), 2) for v in node["pt"]
                ],
            )
        )

    return recovered, remaining, audit


def _assign_locked_nodes(nodes, trusted, typical, cfg):
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    locked_ids = set()

    snap_radius = cfg.seed_snap_fraction * side
    node_cell = side
    node_points = [np.asarray(node["pt"], dtype=float) for node in nodes]
    node_grid = _build_point_spatial_index(node_points, node_cell)

    for owner_index, q in enumerate(trusted):
        c = _center(q["box"])
        candidates = []
        search_bounds = [
            c[0] - snap_radius,
            c[1] - snap_radius,
            c[0] + snap_radius,
            c[1] + snap_radius,
        ]
        candidate_node_indices = _query_point_spatial_index(
            node_grid,
            search_bounds,
            node_cell,
        )
        for idx in candidate_node_indices:
            node = nodes[idx]
            d = float(np.linalg.norm(node["pt"] - c))
            if d <= snap_radius:
                candidates.append((d, -node["radius"], idx))

        if candidates:
            idx = min(candidates)[2]
        else:
            # Step-4 detection remains locked even if foreground maxima are weak.
            idx = len(nodes)
            nodes.append(
                dict(
                    pt=c.copy(),
                    radius=max(1.0, 0.18 * side),
                    response=0.0,
                    raw_valid=False,
                    valid_new=False,
                    locked=True,
                    owners=[],
                    box=list(map(float, q["box"])),
                    evidence=q.get("_evidence", {}),
                    artifact_reason=q.get("_artifact_reason"),
                    synthetic_locked=True,
                )
            )
            node_points.append(c.copy())
            gx = int(math.floor(float(c[0]) / node_cell))
            gy = int(math.floor(float(c[1]) / node_cell))
            node_grid.setdefault((gx, gy), []).append(idx)

        node = nodes[idx]
        node["locked"] = True
        node.setdefault("owners", []).append(owner_index)
        locked_ids.add(idx)

    return locked_ids



def _strict_same_lobe_duplicate_suppression(
    gray,
    trusted,
    nodes,
    typical,
    circle,
    outer_circle,
):
    """Resolve multiple detector boxes assigned to the same visual lobe.

    Rules:
      1) Same-lobe + ONE local peak -> duplicate boxes, keep one.
      2) Same-lobe + >=2 local peaks -> do NOT collapse distinct small boxes.
      3) A large parent box spanning multiple peaks is suppressed when smaller
         child boxes already explain those peaks.

    No transitive duplicate chaining is used.
    """
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    typical_area = max(1.0, float(np.prod(typical)))
    suppressed = set()
    audit = []
    peak_cache = {}

    def pair_union_box(i, j):
        a = np.asarray(trusted[i]["box"], dtype=float)
        b = np.asarray(trusted[j]["box"], dtype=float)
        return [
            float(min(a[0], b[0])),
            float(min(a[1], b[1])),
            float(max(a[2], b[2])),
            float(max(a[3], b[3])),
        ]

    def local_peak_count(box):
        # Dense plates revisit the same parent/union boxes many times.
        # Cache by rounded geometry; this changes runtime only, not decisions.
        key = tuple(round(float(v), 2) for v in box)
        cached = peak_cache.get(key)
        if cached is not None:
            return cached
        _centers, diag = base._strong_local_centers(
            gray,
            box,
            typical,
            circle,
            outer_circle,
            expand=0.02,
        )
        result = (int(diag.get("nms_peak_count", 0)), diag)
        peak_cache[key] = result
        return result

    def score(idx, node_pt):
        q = trusted[idx]
        c = _center(q["box"])
        return (
            1 if q.get("_seed_status") == "accepted_step4_locked" else 0,
            1 if not q.get("review_needed", False) else 0,
            float(q.get("confidence", 0.0)),
            len(q.get("supporting_ids", [])),
            -float(np.linalg.norm(c - node_pt)),
        )

    for node_id, node in enumerate(nodes):
        owners = [
            i for i in node.get("owners", [])
            if i not in suppressed
        ]
        if len(owners) < 2:
            continue

        node_pt = np.asarray(node["pt"], dtype=float)

        # --- A. Suppress multi-peak parent boxes first -----------------------
        for i in owners:
            if i in suppressed:
                continue
            bi = np.asarray(trusted[i]["box"], dtype=float)
            wi = np.maximum(1.0, bi[2:] - bi[:2])
            area_ratio_i = float(np.prod(wi) / typical_area)
            if area_ratio_i < 1.45:
                continue

            peak_count_i, peak_diag_i = local_peak_count(bi.tolist())
            if peak_count_i < 2:
                continue

            child_candidates = []
            for j in owners:
                if j == i or j in suppressed:
                    continue
                bj = np.asarray(trusted[j]["box"], dtype=float)
                wj = np.maximum(1.0, bj[2:] - bj[:2])
                area_ratio_j = float(np.prod(wj) / typical_area)
                ios = float(base._box_ios(bi, bj))
                cj = _center(bj)

                # Smaller box lies largely inside parent and is centered on a
                # different part of the multi-peak parent region.
                if (
                    ios >= 0.82
                    and area_ratio_j <= 1.15
                    and float(np.linalg.norm(cj - node_pt)) <= 1.10 * side
                ):
                    child_candidates.append(j)

            if child_candidates:
                suppressed.add(i)
                audit.append(
                    dict(
                        action="multi_peak_parent_suppressed",
                        node_id=int(node_id),
                        parent_id=trusted[i].get("id"),
                        child_ids=[
                            trusted[j].get("id")
                            for j in child_candidates
                        ],
                        parent_area_ratio=round(area_ratio_i, 4),
                        parent_peak_count=peak_count_i,
                        parent_peak_diagnostics=peak_diag_i,
                    )
                )

        # --- B. Pairwise non-transitive duplicate suppression ---------------
        remaining = [
            i for i in owners
            if i not in suppressed
        ]
        # Higher-quality candidates get first chance to own the lobe.
        remaining.sort(
            key=lambda i: score(i, node_pt),
            reverse=True,
        )

        kept = []
        for i in remaining:
            duplicate_of = None
            duplicate_meta = None

            for k in kept:
                bi = np.asarray(trusted[i]["box"], dtype=float)
                bk = np.asarray(trusted[k]["box"], dtype=float)
                ci = _center(bi)
                ck = _center(bk)
                dist = float(np.linalg.norm(ci - ck))
                ios = float(base._box_ios(bi, bk))

                # Same lobe alone is not enough. The pair must also be locally
                # close/overlapping, then the union must contain only ONE peak.
                if not (
                    dist <= 0.60 * side
                    or ios >= 0.65
                ):
                    continue

                # Each detector box also has its own recentered filled-core
                # peak. Two touching colonies can merge into ONE union peak, but
                # their per-box peaks remain spatially distinct. Never collapse
                # such a pair.
                evi = trusted[i].get("_evidence") or base.evidence(
                    gray, trusted[i], circle
                )
                evk = trusted[k].get("_evidence") or base.evidence(
                    gray, trusted[k], circle
                )
                pi = np.asarray(evi.get("peak", ci), dtype=float)
                pk = np.asarray(evk.get("peak", ck), dtype=float)
                peak_dist = float(np.linalg.norm(pi - pk))
                peak_dist_units = peak_dist / side

                distinct_filled_cores = bool(
                    peak_dist_units >= 0.30
                    and evi.get("signal", 0.0) >= 9.0
                    and evk.get("signal", 0.0) >= 9.0
                    and evi.get("radial", {}).get("kind") != "reject"
                    and evk.get("radial", {}).get("kind") != "reject"
                )
                if distinct_filled_cores:
                    # Different recentered cores = different colony hypotheses,
                    # even if the union peak detector merges the touching pair.
                    continue

                union = pair_union_box(i, k)
                peak_count, peak_diag = local_peak_count(union)
                if peak_count > 1:
                    # Two or more union centers: also likely touching colonies.
                    continue

                duplicate_of = k
                duplicate_meta = dict(
                    center_distance=round(dist, 3),
                    center_distance_units=round(dist / side, 4),
                    ios=round(ios, 4),
                    per_box_peak_distance=round(peak_dist, 3),
                    per_box_peak_distance_units=round(peak_dist_units, 4),
                    local_peak_count=peak_count,
                    peak_diagnostics=peak_diag,
                )
                break

            if duplicate_of is None:
                kept.append(i)
            else:
                suppressed.add(i)
                audit.append(
                    dict(
                        action="same_lobe_single_peak_duplicate_suppressed",
                        node_id=int(node_id),
                        node_pt=[
                            round(float(v), 2)
                            for v in node_pt
                        ],
                        keeper_id=trusted[duplicate_of].get("id"),
                        suppressed_ids=[trusted[i].get("id")],
                        pair_check=duplicate_meta,
                    )
                )

    return suppressed, audit


def _line_edge_metrics(mask, dist, a, b, side):
    """Local connection between two lobe maxima.

    gap_fraction: longest background gap along center-to-center line / colony side
    saddle_ratio: minimum distance-transform radius between the lobes divided by
                  the smaller endpoint radius.
    """
    pa = np.asarray(a["pt"], dtype=float)
    pb = np.asarray(b["pt"], dtype=float)
    distance = float(np.linalg.norm(pb - pa))
    samples = max(17, int(math.ceil(distance)) + 1)

    values = []
    foreground = []
    for t in np.linspace(0.0, 1.0, samples):
        p = pa * (1.0 - t) + pb * t
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            foreground.append(bool(mask[y, x] > 0))
            values.append(float(dist[y, x]))
        else:
            foreground.append(False)
            values.append(0.0)

    longest = current = 0
    for fg in foreground:
        if fg:
            current = 0
        else:
            current += 1
            longest = max(longest, current)

    step = distance / max(1, samples - 1)
    gap_fraction = (longest * step) / max(side, 1e-6)

    ra = max(1e-6, float(a.get("radius", 0.0)))
    rb = max(1e-6, float(b.get("radius", 0.0)))
    denom = min(ra, rb)

    middle = values[
        max(1, int(0.18 * len(values))):
        max(2, int(0.82 * len(values)))
    ]
    saddle = min(middle) if middle else 0.0
    saddle_ratio = saddle / denom if denom > 1e-6 else 1.0

    return dict(
        distance=distance,
        distance_fraction=distance / max(side, 1e-6),
        gap_fraction=float(gap_fraction),
        saddle_ratio=float(saddle_ratio),
    )


def _build_graph(nodes, mask, dist, typical, cfg):
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    cell = cfg.edge_distance_max_fraction * side
    grid = {}

    for idx, node in enumerate(nodes):
        gx = int(math.floor(float(node["pt"][0]) / cell))
        gy = int(math.floor(float(node["pt"][1]) / cell))
        grid.setdefault((gx, gy), []).append(idx)

    adjacency = [[] for _ in nodes]
    seen = set()

    for idx, node in enumerate(nodes):
        gx = int(math.floor(float(node["pt"][0]) / cell))
        gy = int(math.floor(float(node["pt"][1]) / cell))

        for yy in range(gy - 1, gy + 2):
            for xx in range(gx - 1, gx + 2):
                for j in grid.get((xx, yy), []):
                    if j == idx:
                        continue
                    key = (min(idx, j), max(idx, j))
                    if key in seen:
                        continue
                    seen.add(key)

                    d = float(
                        np.linalg.norm(
                            nodes[idx]["pt"] - nodes[j]["pt"]
                        )
                    )
                    dfrac = d / side
                    if not (
                        cfg.edge_distance_min_fraction
                        <= dfrac
                        <= cfg.edge_distance_max_fraction
                    ):
                        continue

                    metrics = _line_edge_metrics(
                        mask, dist, nodes[idx], nodes[j], side
                    )

                    # Two lobes may touch continuously (small saddle) or have a
                    # tiny dark gap, but a long background gap is not "connected".
                    if metrics["gap_fraction"] > cfg.edge_gap_max_fraction:
                        continue
                    if (
                        metrics["gap_fraction"] <= 0.02
                        and metrics["saddle_ratio"] > cfg.edge_saddle_max
                    ):
                        # Uniform-width foreground: one elongated lobe, not a neck.
                        continue

                    adjacency[idx].append((j, metrics))
                    adjacency[j].append((idx, metrics))

    return adjacency


def resolve_lobe_graph(
    gray,
    rows,
    circle,
    outer_circle=None,
    config: LobeGraphConfig | None = None,
):
    cfg = config or LobeGraphConfig()
    typical = base.typical_geometry(rows)
    trusted, review_rejected, specular_rejected = _trusted_rows(
        gray, rows, circle, outer_circle
    )

    response, mask, threshold, scale = _foreground(
        gray, trusted, typical, circle, cfg
    )
    nodes, dist = _extract_raw_nodes(
        mask, response, typical, cfg
    )

    # Ownership uses ALL Step-4 boxes, not only trusted ones. Otherwise the
    # completion graph can create a duplicate exactly where a review box already
    # exists.
    _validate_new_nodes(
        gray,
        nodes,
        rows,
        typical,
        circle,
        outer_circle,
        cfg,
    )

    recovered_review, review_rejected, recovery_audit = _recover_review_rows(
        review_rejected,
        nodes,
        trusted,
        typical,
    )
    trusted = trusted + recovered_review

    locked_ids = _assign_locked_nodes(
        nodes, trusted, typical, cfg
    )

    # A blanket rim ban can hide a real colony close to the dish wall. Use
    # per-image lobe thickness to rescue only colony-sized round lobes.
    locked_radii = [
        float(nodes[i].get("radius", 0.0))
        for i in locked_ids
        if not nodes[i].get("synthetic_locked", False)
        and float(nodes[i].get("radius", 0.0)) > 0.0
    ]
    rim_round_rescued = []
    if locked_radii:
        median_locked_radius = float(np.median(locked_radii))
        for node_id, node in enumerate(nodes):
            if (
                node.get("rim_blocked", False)
                and node.get("colony_like", False)
                and not node.get("owned_by_step4", False)
                and not node.get("specular", {}).get("is_artifact", False)
                and float(node.get("evidence", {}).get("radial", {}).get("rim", 99.0)) <= 0.35
                and float(node.get("evidence", {}).get("radial", {}).get("ring_fraction", 1.0)) <= 0.15
                and float(node.get("radius", 0.0))
                >= 0.80 * median_locked_radius
            ):
                node["rim_blocked"] = False
                node["valid_new"] = True
                node["reject_reason"] = None
                node["rim_round_rescued"] = True
                rim_round_rescued.append(node_id)
    else:
        median_locked_radius = None

    # Conservative graph-add artifact guard. Only NEW/unowned nodes are checked;
    # trusted Step-4 detections are never removed by this rule.
    line_artifact_nodes = []
    if median_locked_radius is not None:
        for node_id, node in enumerate(nodes):
            if not node.get("valid_new", False):
                continue
            if node.get("owned_by_step4", False):
                continue
            line_check = _small_lobe_line_artifact(
                gray,
                node,
                typical,
                median_locked_radius,
                circle,
            )
            node["line_artifact"] = line_check
            if line_check.get("is_artifact", False):
                node["valid_new"] = False
                node["reject_reason"] = "small_lobe_crossed_by_long_line"
                line_artifact_nodes.append(node_id)

    # Detector can fire multiple overlapping boxes for the SAME visual lobe.
    # Suppress only strict same-lobe duplicates; different lobe nodes always win.
    duplicate_owner_indices, duplicate_audit = _strict_same_lobe_duplicate_suppression(
        gray,
        trusted,
        nodes,
        typical,
        circle,
        outer_circle,
    )

    # New synthetic locked nodes were appended after validation.
    adjacency = _build_graph(
        nodes, mask, dist, typical, cfg
    )

    represented = set(locked_ids)
    added = set()
    added_reason = {}
    queue = deque(sorted(locked_ids))
    edge_audit = []

    def walk_from_queue(queue_obj, source_label):
        while queue_obj:
            src = queue_obj.popleft()
            for dst, metrics in adjacency[src]:
                if dst in represented:
                    continue

                node = nodes[dst]
                if not node.get("valid_new", False):
                    continue

                # Every Step-4 ROI owns its location. Graph completion can
                # only create a node where Step 4 had no box at all.
                if node.get("owned_by_step4", False):
                    represented.add(dst)
                    continue

                represented.add(dst)
                added.add(dst)
                added_reason.setdefault(
                    dst,
                    "recursive_lobe_graph_missing",
                )
                queue_obj.append(dst)
                edge_audit.append(
                    dict(
                        source=source_label,
                        from_node=int(src),
                        to_node=int(dst),
                        from_pt=[
                            round(float(v), 2) for v in nodes[src]["pt"]
                        ],
                        to_pt=[
                            round(float(v), 2) for v in node["pt"]
                        ],
                        metrics={
                            k: round(float(v), 4)
                            for k, v in metrics.items()
                        },
                    )
                )

    # First pass: complete touching/nearby lobes from detector-owned seeds.
    walk_from_queue(queue, "locked_seed_graph")

    # Second pass: recover truly isolated detector misses. This is deliberately
    # stricter than graph-linked completion.
    orphan_added = []
    orphan_queue = deque()
    if median_locked_radius is not None and median_locked_radius > 0:
        for node_id, node in enumerate(nodes):
            if node_id in represented:
                continue
            if not node.get("valid_new", False):
                continue
            if node.get("owned_by_step4", False):
                continue
            if node.get("rim_blocked", False):
                continue
            if node.get("specular", {}).get("is_artifact", False):
                continue

            ev = node.get("evidence", {})
            radial = ev.get("radial", {})
            radius_ratio = (
                float(node.get("radius", 0.0))
                / median_locked_radius
            )
            component_fraction = float(
                ev.get("component_fraction", 0.0)
            )

            orphan_ok = bool(
                0.72 <= radius_ratio <= 1.35
                and radial.get("kind") == "support"
                and ev.get("signal", 0.0) >= 10.0
                and radial.get("core", 0.0) >= 6.0
                and radial.get("filled_fraction", 0.0) >= 0.75
                and radial.get("aspect", 0.0) >= 0.70
                and radial.get("rim", 99.0) <= 0.35
                and radial.get("ring_fraction", 1.0) <= 0.15
                and 0.20 <= component_fraction <= 1.45
            )
            if not orphan_ok:
                continue

            represented.add(node_id)
            added.add(node_id)
            added_reason[node_id] = "orphan_lobe_detector_miss"
            orphan_added.append(node_id)
            orphan_queue.append(node_id)

    # A recovered orphan can itself reveal a touching neighbour that YOLO also
    # missed, so run the same graph walk once from these new trusted image seeds.
    walk_from_queue(orphan_queue, "orphan_seed_graph")

    final_boxes = []
    for owner_index, q in enumerate(trusted):
        if owner_index in duplicate_owner_indices:
            continue
        final_boxes.append(
            dict(
                id=f'lobe_seed_{q.get("id","candidate")}',
                box=[round(float(v), 2) for v in q["box"]],
                resolution="counted_single",
                resolution_reason=q.get(
                    "_seed_status", "step4_locked"
                ),
                parent_ids=[q.get("id")],
            )
        )

    for node_id in sorted(added):
        node = nodes[node_id]
        final_boxes.append(
            dict(
                id=f'lobe_missing_{node_id:04d}',
                box=[round(float(v), 2) for v in node["box"]],
                resolution="counted_split",
                resolution_reason=added_reason.get(
                    node_id,
                    "recursive_lobe_graph_missing",
                ),
                source_seed=dict(
                    source="distance_lobe_graph",
                    center=[
                        round(float(v), 2) for v in node["pt"]
                    ],
                    radius=round(float(node["radius"]), 3),
                ),
                parent_ids=[],
            )
        )

    debug = dict(
        typical=[round(float(v), 3) for v in typical],
        step4_rows=len(rows),
        trusted=len(trusted),
        same_lobe_duplicates_suppressed=len(duplicate_owner_indices),
        rim_round_rescued=len(rim_round_rescued),
        line_artifacts_rejected=len(line_artifact_nodes),
        median_locked_lobe_radius=(
            None if median_locked_radius is None
            else round(median_locked_radius, 4)
        ),
        rejected_review=len(review_rejected),
        recovered_review=len(recovered_review),
        specular_rejected=len(specular_rejected),
        raw_lobe_nodes=len(nodes),
        locked_nodes=len(locked_ids),
        graph_edges=sum(len(a) for a in adjacency) // 2,
        added_missing=len(added),
        orphan_detector_misses=len(orphan_added),
        foreground_threshold=round(float(threshold), 4),
        response_scale=round(float(scale), 4),
    )

    node_debug = []
    for idx, node in enumerate(nodes):
        node_debug.append(
            dict(
                id=idx,
                pt=[round(float(v), 2) for v in node["pt"]],
                radius=round(float(node.get("radius", 0.0)), 3),
                locked=bool(idx in locked_ids),
                valid_new=bool(node.get("valid_new", False)),
                represented=bool(idx in represented),
                added=bool(idx in added),
                owners=list(node.get("owners", [])),
                reject_reason=node.get("reject_reason"),
                colony_like=bool(node.get("colony_like", False)),
                owned_by_step4=bool(node.get("owned_by_step4", False)),
                rim_blocked=bool(node.get("rim_blocked", False)),
                signal=round(float(node.get("evidence", {}).get("signal", 0.0)), 3),
                component_fraction=round(float(node.get("evidence", {}).get("component_fraction", 0.0)), 4),
                radial_kind=node.get("evidence", {}).get("radial", {}).get("kind"),
                filled_fraction=round(float(node.get("evidence", {}).get("radial", {}).get("filled_fraction", 0.0)), 4),
                core=round(float(node.get("evidence", {}).get("radial", {}).get("core", 0.0)), 3),
            )
        )

    debug["review_recovery_audit"] = recovery_audit
    debug["same_lobe_duplicate_audit"] = duplicate_audit
    debug["same_lobe_duplicate_ids"] = [
        trusted[i].get("id")
        for i in sorted(duplicate_owner_indices)
    ]
    debug["specular_rejected_ids"] = [
        q.get("id") for q in specular_rejected
    ]
    return final_boxes, edge_audit, debug, node_debug, mask, response
