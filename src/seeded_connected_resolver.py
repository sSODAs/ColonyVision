"""Strict seeded-connected colony completion.

Purpose
-------
Freeze the current detector/filter/dedup result and replace ONLY the old
yellow/addition logic.

A new colony may be proposed only when:
  1) foreground pixels are physically connected to an existing final seed,
  2) the connected component extends outside that seed ROI,
  3) the same connected component contains a second colony-like core,
  4) the second core is not already represented by another final box.

No graph jump, gap allowance, dilation, or MORPH_CLOSE is used here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

import lobe_graph_resolver as lobe
import colony_evidence as base


@dataclass
class SeededConnectedConfig:
    patch_expand_fraction: float = 0.95
    patch_expand_step_fraction: float = 0.75
    max_expand_steps: int = 4

    low_to_high_ratio: float = 0.58
    min_low_threshold: float = 1.0
    min_high_threshold: float = 2.0

    # A detector box need not cover the full visible colony. Continuation is
    # therefore measured beyond a normal single-colony footprint around the
    # seed center, not merely outside the rectangle.
    continuation_radius_fraction: float = 0.68
    continuation_area_fraction: float = 0.025

    peak_kernel_fraction: float = 0.34
    peak_radius_min_fraction: float = 0.11
    peak_radius_max_fraction: float = 0.78
    peak_nms_fraction: float = 0.40

    represented_center_fraction: float = 0.62
    candidate_min_center_fraction: float = 0.50
    candidate_max_center_fraction: float = 1.15

    saddle_ratio_max: float = 0.72
    saddle_radius_min_fraction: float = 0.035

    candidate_signal_min: float = 8.0
    candidate_core_min: float = 5.0
    candidate_filled_min: float = 0.50
    candidate_aspect_min: float = 0.56

    # A second DT peak must own a meaningful watershed territory inside the
    # SAME connected component before it can become another colony.
    territory_area_min_fraction: float = 0.28
    territory_area_max_fraction: float = 2.20
    territory_aspect_min: float = 0.42
    territory_extent_min: float = 0.28


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


def _expand_box(box, amount, w, h):
    return _clip_box(
        [
            float(box[0]) - amount,
            float(box[1]) - amount,
            float(box[2]) + amount,
            float(box[3]) + amount,
        ],
        w,
        h,
    )


def _response_image(gray, typical):
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    f = gray.astype(np.float32)
    small = cv2.GaussianBlur(f, (0, 0), max(0.7, 0.055 * side))
    broad = cv2.GaussianBlur(f, (0, 0), max(2.2, 0.58 * side))
    return np.maximum(0.0, small - broad)


def _seed_thresholds(response, seed_box, typical, cfg):
    h, w = response.shape[:2]
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    search = _expand_box(seed_box, 0.45 * side, w, h)
    x1, y1, x2, y2 = map(int, map(round, search))
    patch = response[y1:y2, x1:x2]
    if patch.size == 0:
        return cfg.min_high_threshold, cfg.min_low_threshold

    med = float(np.median(patch))
    mad = float(np.median(np.abs(patch - med))) + 1e-6
    pmax = float(patch.max())

    high = max(
        cfg.min_high_threshold,
        med + 1.55 * mad,
        0.12 * pmax,
    )

    # A seed ROI supplies a second calibration source. This helps faint colonies
    # without lowering the threshold based on unrelated background pixels.
    b = np.asarray(seed_box, dtype=float)
    sx1 = max(0, int(math.floor(b[0])))
    sy1 = max(0, int(math.floor(b[1])))
    sx2 = min(w, int(math.ceil(b[2])))
    sy2 = min(h, int(math.ceil(b[3])))
    seed_vals = response[sy1:sy2, sx1:sx2]
    seed_pos = seed_vals[seed_vals > 0]
    if seed_pos.size:
        high = min(
            high,
            max(cfg.min_high_threshold, 0.72 * float(np.percentile(seed_pos, 65))),
        )

    low = max(cfg.min_low_threshold, cfg.low_to_high_ratio * high)
    return float(high), float(low)


def _dish_mask(shape, circle, margin):
    h, w = shape[:2]
    cx, cy, r = map(float, circle)
    rr = max(0.0, r - margin)
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr * rr



def _has_strict_boundary_continuation(
    response,
    seed_box,
    typical,
    circle,
    cfg,
):
    """Cheap strict test: does THIS seed component cross its ROI boundary?

    Uses a small local window and the same no-close/no-dilate weak foreground.
    If the seed-connected label never leaves the original ROI, there is nothing
    to trace or split.
    """
    h, w = response.shape[:2]
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    high_thr, low_thr = _seed_thresholds(response, seed_box, typical, cfg)

    patch_box = _expand_box(seed_box, 0.22 * side, w, h)
    x1 = max(0, int(math.floor(patch_box[0])))
    y1 = max(0, int(math.floor(patch_box[1])))
    x2 = min(w, int(math.ceil(patch_box[2])))
    y2 = min(h, int(math.ceil(patch_box[3])))
    if x2 <= x1 or y2 <= y1:
        return False, dict(reason="empty_patch")

    dish = _dish_mask(response.shape, circle, margin=0.04 * side)
    local = response[y1:y2, x1:x2]
    weak = (
        (local >= low_thr)
        & dish[y1:y2, x1:x2]
    ).astype(np.uint8)
    strong = (
        (local >= high_thr)
        & dish[y1:y2, x1:x2]
    )

    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
    if nlab <= 1:
        return False, dict(reason="no_foreground_component")

    b = np.asarray(seed_box, dtype=float)
    sx1 = max(0, int(math.floor(b[0] - x1)))
    sy1 = max(0, int(math.floor(b[1] - y1)))
    sx2 = min(weak.shape[1], int(math.ceil(b[2] - x1)))
    sy2 = min(weak.shape[0], int(math.ceil(b[3] - y1)))
    if sx2 <= sx1 or sy2 <= sy1:
        return False, dict(reason="seed_outside_patch")

    bw = max(1, sx2 - sx1)
    bh = max(1, sy2 - sy1)
    mx = max(1, int(round(0.14 * bw)))
    my = max(1, int(round(0.14 * bh)))
    ix1 = min(sx2, sx1 + mx)
    iy1 = min(sy2, sy1 + my)
    ix2 = max(ix1 + 1, sx2 - mx)
    iy2 = max(iy1 + 1, sy2 - my)

    authority = np.zeros_like(weak, dtype=bool)
    authority[iy1:iy2, ix1:ix2] = True

    best_lab = None
    best_score = -1
    for lab in range(1, nlab):
        comp = labels == lab
        strong_inside = int(np.count_nonzero(comp & strong & authority))
        weak_inside = int(np.count_nonzero(comp & authority))
        if weak_inside <= 0:
            continue
        score = strong_inside * 1000000 + weak_inside
        if score > best_score:
            best_score = score
            best_lab = lab

    if best_lab is None:
        return False, dict(reason="no_seed_label")

    selected = labels == best_lab

    yy, xx = np.ogrid[y1:y2, x1:x2]
    seed_center = _center(seed_box)
    distance = np.sqrt(
        (xx - float(seed_center[0])) ** 2
        + (yy - float(seed_center[1])) ** 2
    )
    far = (
        selected
        & (
            distance
            >= cfg.continuation_radius_fraction * side
        )
    )
    far_area = int(np.count_nonzero(far))

    # A normal detector box can be slightly smaller than the visible colony.
    # Only connected pixels that reach beyond a normal single-colony footprint
    # count as a potential attached continuation.
    minimum = max(
        2,
        int(round(
            cfg.continuation_area_fraction
            * float(np.prod(typical))
        )),
    )
    max_distance = (
        float(np.max(distance[selected]))
        if np.any(selected)
        else 0.0
    )
    crosses = bool(
        far_area >= minimum
        and max_distance
        >= cfg.continuation_radius_fraction * side
    )
    return (
        crosses,
        dict(
            reason=(
                "seed_component_extends_beyond_single_colony_footprint"
                if crosses
                else "seed_component_within_single_colony_footprint"
            ),
            far_area=far_area,
            minimum_area=minimum,
            max_distance_fraction=round(
                max_distance / max(side, 1e-6), 4
            ),
            continuation_radius_fraction=cfg.continuation_radius_fraction,
            high_threshold=round(float(high_thr), 4),
            low_threshold=round(float(low_thr), 4),
        ),
    )


def _select_seed_component(
    response,
    seed_box,
    typical,
    circle,
    cfg,
):
    """Follow one weak-foreground component seeded from the existing ROI.

    Strong pixels inside the seed choose the component. Growth is permitted only
    through 8-connected weak foreground pixels. No closing/dilation occurs.
    """
    h, w = response.shape[:2]
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    high_thr, low_thr = _seed_thresholds(response, seed_box, typical, cfg)

    dish = _dish_mask(response.shape, circle, margin=0.04 * side)
    current = _expand_box(seed_box, cfg.patch_expand_fraction * side, w, h)
    final = None

    for step in range(cfg.max_expand_steps):
        x1 = max(0, int(math.floor(current[0])))
        y1 = max(0, int(math.floor(current[1])))
        x2 = min(w, int(math.ceil(current[2])))
        y2 = min(h, int(math.ceil(current[3])))
        if x2 <= x1 or y2 <= y1:
            break

        local_response = response[y1:y2, x1:x2]
        weak = (
            (local_response >= low_thr)
            & dish[y1:y2, x1:x2]
        ).astype(np.uint8)
        strong = (
            (local_response >= high_thr)
            & dish[y1:y2, x1:x2]
        )

        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
        if nlab <= 1:
            break

        b = np.asarray(seed_box, dtype=float)
        sx1 = max(0, int(math.floor(b[0] - x1)))
        sy1 = max(0, int(math.floor(b[1] - y1)))
        sx2 = min(weak.shape[1], int(math.ceil(b[2] - x1)))
        sy2 = min(weak.shape[0], int(math.ceil(b[3] - y1)))
        if sx2 <= sx1 or sy2 <= sy1:
            break

        # Central part of the detector ROI is the only seed authority.
        bw = max(1, sx2 - sx1)
        bh = max(1, sy2 - sy1)
        mx = max(1, int(round(0.14 * bw)))
        my = max(1, int(round(0.14 * bh)))
        ix1 = min(sx2, sx1 + mx)
        iy1 = min(sy2, sy1 + my)
        ix2 = max(ix1 + 1, sx2 - mx)
        iy2 = max(iy1 + 1, sy2 - my)

        seed_authority = np.zeros_like(weak, dtype=bool)
        seed_authority[iy1:iy2, ix1:ix2] = True

        best_lab = None
        best_score = -1
        for lab in range(1, nlab):
            comp = labels == lab
            strong_inside = int(np.count_nonzero(comp & strong & seed_authority))
            weak_inside = int(np.count_nonzero(comp & seed_authority))
            if weak_inside <= 0:
                continue
            # Strong overlap dominates; weak overlap is fallback for faint seeds.
            score = strong_inside * 1000000 + weak_inside
            if score > best_score:
                best_score = score
                best_lab = lab

        if best_lab is None:
            break

        selected = labels == best_lab
        ys, xs = np.where(selected)
        if not len(xs):
            break

        edge_pad = max(1, int(round(0.04 * side)))
        touches = bool(
            xs.min() <= edge_pad
            or ys.min() <= edge_pad
            or xs.max() >= selected.shape[1] - 1 - edge_pad
            or ys.max() >= selected.shape[0] - 1 - edge_pad
        )

        global_mask = np.zeros_like(response, dtype=np.uint8)
        global_mask[y1:y2, x1:x2][selected] = 1

        gx1 = float(x1 + xs.min())
        gy1 = float(y1 + ys.min())
        gx2 = float(x1 + xs.max() + 1)
        gy2 = float(y1 + ys.max() + 1)

        final = dict(
            mask=global_mask,
            component_box=[gx1, gy1, gx2, gy2],
            component_area=int(np.count_nonzero(selected)),
            high_threshold=high_thr,
            low_threshold=low_thr,
            patch=[x1, y1, x2, y2],
            touches_patch_edge=touches,
            step=step,
        )

        if not touches:
            break

        current = _expand_box(
            current,
            cfg.patch_expand_step_fraction * side,
            w,
            h,
        )

    return final


def _component_extension(component_mask, seed_box, typical, cfg):
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    h, w = component_mask.shape[:2]

    yy, xx = np.ogrid[:h, :w]
    c = _center(seed_box)
    distance = np.sqrt(
        (xx - float(c[0])) ** 2
        + (yy - float(c[1])) ** 2
    )
    extension = (
        (component_mask > 0)
        & (
            distance
            >= cfg.continuation_radius_fraction * side
        )
    )
    area = int(np.count_nonzero(extension))
    enough = (
        area
        >= cfg.continuation_area_fraction
        * float(np.prod(typical))
    )
    return extension.astype(np.uint8), area, bool(enough)


def _component_peaks(component_mask, response, typical, cfg):
    """Find DT cores on a cropped component, returning global coordinates."""
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))

    ys, xs = np.where(component_mask > 0)
    if not len(xs):
        return [], dict(
            dist=np.zeros((1, 1), dtype=np.float32),
            mask=np.zeros((1, 1), dtype=np.uint8),
            offset=np.asarray([0.0, 0.0], dtype=float),
        )

    pad = max(2, int(round(0.10 * side)))
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(component_mask.shape[1], int(xs.max()) + 1 + pad)
    y2 = min(component_mask.shape[0], int(ys.max()) + 1 + pad)

    local_mask = component_mask[y1:y2, x1:x2].astype(np.uint8)
    local_response = response[y1:y2, x1:x2]
    dist = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)

    kd = max(3, int(round(cfg.peak_kernel_fraction * side)))
    if kd % 2 == 0:
        kd += 1
    kd = min(kd, 31)

    dil = cv2.dilate(dist, np.ones((kd, kd), np.uint8))
    peak_mask = (
        (local_mask > 0)
        & (dist >= dil - 1e-6)
        & (dist >= cfg.peak_radius_min_fraction * side)
    ).astype(np.uint8)

    nlab, labels, stats, cents = cv2.connectedComponentsWithStats(peak_mask, 8)
    peaks = []
    for lab in range(1, nlab):
        if int(stats[lab, cv2.CC_STAT_AREA]) <= 0:
            continue
        lx, ly = map(float, cents[lab])
        ix = int(np.clip(round(lx), 0, dist.shape[1] - 1))
        iy = int(np.clip(round(ly), 0, dist.shape[0] - 1))
        radius = float(dist[iy, ix])
        if radius > cfg.peak_radius_max_fraction * side:
            continue
        gx = float(lx + x1)
        gy = float(ly + y1)
        peaks.append(
            dict(
                pt=np.asarray([gx, gy], dtype=float),
                local_pt=np.asarray([lx, ly], dtype=float),
                radius=radius,
                response=float(local_response[iy, ix]),
            )
        )

    peaks.sort(key=lambda p: (p["radius"], p["response"]), reverse=True)
    kept = []
    for peak in peaks:
        if all(
            float(np.linalg.norm(peak["pt"] - other["pt"]))
            >= cfg.peak_nms_fraction * side
            for other in kept
        ):
            kept.append(peak)

    return kept, dict(
        dist=dist,
        mask=local_mask,
        offset=np.asarray([float(x1), float(y1)], dtype=float),
    )


def _watershed_peak_regions(dist_pack, peaks, typical):
    """Partition ONE cropped connected component among its DT cores."""
    if not peaks:
        return []

    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    typical_area = max(1.0, float(np.prod(typical)))

    dist = dist_pack["dist"]
    local_mask = dist_pack["mask"]
    offset = dist_pack["offset"]

    surface = cv2.normalize(
        dist, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    surface = 255 - surface

    markers = np.zeros(local_mask.shape, dtype=np.int32)
    markers[local_mask == 0] = 1

    marker_radius = max(1, int(round(0.035 * side)))
    for i, peak in enumerate(peaks):
        local_pt = np.asarray(peak["pt"], dtype=float) - offset
        x = int(np.clip(round(float(local_pt[0])), 0, markers.shape[1]-1))
        y = int(np.clip(round(float(local_pt[1])), 0, markers.shape[0]-1))
        cv2.circle(markers, (x, y), marker_radius, i + 2, -1)

    cv2.watershed(
        cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR),
        markers,
    )

    stats = []
    for i, peak in enumerate(peaks):
        region = (markers == (i + 2)) & (local_mask > 0)
        ys, xs = np.where(region)
        if not len(xs):
            stats.append(
                dict(
                    area=0,
                    area_ratio=0.0,
                    aspect=0.0,
                    extent=0.0,
                    bbox=None,
                )
            )
            continue

        x1 = int(xs.min()); x2 = int(xs.max()) + 1
        y1 = int(ys.min()); y2 = int(ys.max()) + 1
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = int(len(xs))
        stats.append(
            dict(
                area=area,
                area_ratio=float(area / typical_area),
                aspect=float(min(bw, bh) / max(bw, bh)),
                extent=float(area / max(1, bw * bh)),
                bbox=[
                    int(x1 + offset[0]),
                    int(y1 + offset[1]),
                    int(x2 + offset[0]),
                    int(y2 + offset[1]),
                ],
            )
        )
    return stats


def _saddle_metrics(dist_pack, a, b, side):
    dist = dist_pack["dist"]
    offset = dist_pack["offset"]
    pa = np.asarray(a, dtype=float) - offset
    pb = np.asarray(b, dtype=float) - offset

    def sample(p):
        x = int(np.clip(round(float(p[0])), 0, dist.shape[1] - 1))
        y = int(np.clip(round(float(p[1])), 0, dist.shape[0] - 1))
        return float(dist[y, x])

    da = sample(pa)
    db = sample(pb)
    denom = min(da, db)
    if denom <= 1e-6:
        return dict(ratio=1.0, radius=0.0, radius_fraction=0.0)

    vals = []
    for t in np.linspace(0.18, 0.82, 41):
        p = pa * (1.0 - t) + pb * t
        vals.append(sample(p))

    saddle = min(vals) if vals else 0.0
    return dict(
        ratio=float(saddle / denom),
        radius=float(saddle),
        radius_fraction=float(saddle / max(side, 1e-6)),
    )


def _represented(pt, boxes, side, cfg):
    p = np.asarray(pt, dtype=float)
    for q in boxes:
        b = np.asarray(q["box"], dtype=float)
        c = _center(b)
        if float(np.linalg.norm(p - c)) <= cfg.represented_center_fraction * side:
            return True
        if b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]:
            return True
    return False


def _candidate_box(pt, typical, shape):
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


def _validate_candidate(
    gray,
    box,
    circle,
    outer_circle,
    typical,
    base_boxes,
    cfg,
):
    spec = lobe._specular_highlight_artifact(gray, box)
    if spec.get("is_artifact", False):
        return False, dict(reason="specular", specular=spec)

    large_ring = base._large_ring_surface_artifact(
        gray,
        box,
        typical,
        outer_circle,
    )
    if large_ring.get("is_artifact", False):
        return False, dict(
            reason="large_surface_ring",
            specular=spec,
            large_ring=large_ring,
        )

    q = dict(
        box=box,
        review_needed=True,
        supporting_ids=[],
        tile_edges=[],
    )
    ev = base.evidence(gray, q, circle)
    radial = ev.get("radial", {})
    reason = base._hard_artifact_reason(
        gray, q, circle, outer_circle, ev
    )

    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))
    ev_peak = np.asarray(ev.get("peak", _center(box)), dtype=float)

    # evidence() is allowed to recenter. That recentered core must still be
    # genuinely unrepresented; otherwise a yellow candidate can accidentally
    # borrow evidence from a neighbouring green colony.
    recentered_owned = _represented(
        ev_peak,
        base_boxes,
        side,
        cfg,
    )

    ok = bool(
        reason is None
        and not recentered_owned
        and ev.get("signal", 0.0) >= cfg.candidate_signal_min
        and radial.get("kind") != "reject"
        and radial.get("core", 0.0) >= cfg.candidate_core_min
        and radial.get("filled_fraction", 0.0) >= cfg.candidate_filled_min
        and radial.get("aspect", 0.0) >= cfg.candidate_aspect_min
        and radial.get("ring_fraction", 1.0) <= 0.35
    )
    return ok, dict(
        reason=(
            "recentered_peak_owned"
            if recentered_owned
            else reason
        ),
        evidence=ev,
        specular=spec,
        large_ring=large_ring,
        recentered_owned=recentered_owned,
    )


def _strict_seeded_additions(
    gray,
    base_boxes,
    typical,
    circle,
    outer_circle,
    cfg,
):
    response = _response_image(gray, typical)
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))

    proposals = []
    seed_audit = []

    for seed_index, seed in enumerate(base_boxes):
        seed_box = list(map(float, seed["box"]))

        crosses, precheck = _has_strict_boundary_continuation(
            response,
            seed_box,
            typical,
            circle,
            cfg,
        )
        if not crosses:
            seed_audit.append(
                dict(
                    seed_index=seed_index,
                    action="no_connected_pixel_continuation",
                    precheck=precheck,
                )
            )
            continue

        trace = _select_seed_component(
            response,
            seed_box,
            typical,
            circle,
            cfg,
        )
        if trace is None:
            seed_audit.append(
                dict(seed_index=seed_index, action="no_seed_component")
            )
            continue

        extension_mask, extension_area, has_extension = _component_extension(
            trace["mask"],
            seed_box,
            typical,
            cfg,
        )
        if not has_extension:
            seed_audit.append(
                dict(
                    seed_index=seed_index,
                    action="connected_component_inside_seed",
                    extension_area=extension_area,
                )
            )
            continue

        peaks, dist = _component_peaks(
            trace["mask"],
            response,
            typical,
            cfg,
        )
        if len(peaks) < 2:
            seed_audit.append(
                dict(
                    seed_index=seed_index,
                    action="extension_but_single_core",
                    extension_area=extension_area,
                    peak_count=len(peaks),
                )
            )
            continue

        territories = _watershed_peak_regions(
            dist,
            peaks,
            typical,
        )

        seed_center = _center(seed_box)
        primary_index = min(
            range(len(peaks)),
            key=lambda i: float(
                np.linalg.norm(peaks[i]["pt"] - seed_center)
            ),
        )
        primary = peaks[primary_index]

        local_added = []
        for peak_index, peak in enumerate(peaks):
            if peak_index == primary_index:
                continue
            pt = peak["pt"]

            territory = territories[peak_index]
            if not (
                cfg.territory_area_min_fraction
                <= territory["area_ratio"]
                <= cfg.territory_area_max_fraction
                and territory["aspect"] >= cfg.territory_aspect_min
                and territory["extent"] >= cfg.territory_extent_min
            ):
                continue

            # This module exists only for a component extending out of THIS ROI.
            # A second core fully contained well inside the original ROI belongs
            # to normal duplicate/split handling, not connected-extension logic.
            b = np.asarray(seed_box, dtype=float)
            outside_seed = not (
                b[0] <= pt[0] <= b[2]
                and b[1] <= pt[1] <= b[3]
            )
            if not outside_seed:
                continue

            core_distance = float(
                np.linalg.norm(pt - primary["pt"])
            )
            if core_distance < cfg.candidate_min_center_fraction * side:
                continue
            if core_distance > cfg.candidate_max_center_fraction * side:
                # Do not jump across a long connected chain in one step.
                # A farther colony must be reached through an accepted local
                # neighbour in a later hop.
                continue

            if _represented(pt, base_boxes, side, cfg):
                continue

            saddle = _saddle_metrics(
                dist,
                primary["pt"],
                pt,
                side,
            )
            if saddle["ratio"] > cfg.saddle_ratio_max:
                continue
            if saddle["radius_fraction"] < cfg.saddle_radius_min_fraction:
                # A one-pixel/noise bridge is not sufficient physical contact.
                continue

            box = _candidate_box(pt, typical, gray.shape)
            ok, validation = _validate_candidate(
                gray,
                box,
                circle,
                outer_circle,
                typical,
                base_boxes,
                cfg,
            )
            if not ok:
                continue

            proposals.append(
                dict(
                    box=[round(float(v), 2) for v in box],
                    pt=np.asarray(pt, dtype=float),
                    radius=float(peak["radius"]),
                    response=float(peak["response"]),
                    seed_index=seed_index,
                    seed_box=seed_box,
                    saddle=saddle,
                    territory=territory,
                    validation=validation,
                )
            )
            local_added.append(
                [
                    round(float(pt[0]), 2),
                    round(float(pt[1]), 2),
                ]
            )

        seed_audit.append(
            dict(
                seed_index=seed_index,
                action=(
                    "seeded_component_split_candidates"
                    if local_added
                    else "connected_extension_no_valid_second_core"
                ),
                extension_area=extension_area,
                peak_count=len(peaks),
                added_centers=local_added,
                component_box=[
                    round(float(v), 2)
                    for v in trace["component_box"]
                ],
                high_threshold=round(float(trace["high_threshold"]), 4),
                low_threshold=round(float(trace["low_threshold"]), 4),
            )
        )

    # The same attached colony may be reached from two neighbouring seed boxes.
    # Deduplicate additions only after strict physical-connectivity proof.
    proposals.sort(
        key=lambda p: (
            p["radius"],
            p["validation"]["evidence"].get("signal", 0.0),
        ),
        reverse=True,
    )
    kept = []
    for p in proposals:
        if all(
            float(np.linalg.norm(p["pt"] - q["pt"]))
            >= 0.42 * side
            for q in kept
        ):
            kept.append(p)

    return kept, seed_audit, response


def resolve_seeded_connected(
    gray,
    rows,
    circle,
    outer_circle=None,
    config: SeededConnectedConfig | None = None,
    frozen_base_boxes=None,
    frozen_old_debug=None,
):
    """Freeze current detector/filter result; replace additions with strict tracing.

    When frozen_base_boxes is supplied, detector/filter/dedup is NOT recomputed.
    This is the preferred mode after the base result has been frozen.
    """
    cfg = config or SeededConnectedConfig()

    if frozen_base_boxes is None:
        old_final, old_edges, old_debug, old_nodes, old_mask, old_response = (
            lobe.resolve_lobe_graph(
                gray,
                rows,
                circle,
                outer_circle,
            )
        )
        base_boxes = [
            dict(q)
            for q in old_final
            if q.get("resolution") != "counted_split"
        ]
        old_graph_additions = sum(
            1 for q in old_final if q.get("resolution") == "counted_split"
        )
    else:
        base_boxes = [dict(q) for q in frozen_base_boxes]
        old_debug = dict(frozen_old_debug or {})
        old_graph_additions = 0

    typical = base.typical_geometry(
        [dict(box=q["box"]) for q in base_boxes]
        if base_boxes
        else rows
    )

    additions, seed_audit, response = _strict_seeded_additions(
        gray,
        base_boxes,
        typical,
        circle,
        outer_circle,
        cfg,
    )

    final_boxes = [dict(q) for q in base_boxes]
    for i, rec in enumerate(additions):
        final_boxes.append(
            dict(
                id=f"seeded_connected_{i:04d}",
                box=rec["box"],
                resolution="counted_split",
                resolution_reason="seeded_connected_attached_split",
                source_seed=dict(
                    source="strict_pixel_connected_component",
                    seed_index=int(rec["seed_index"]),
                    center=[
                        round(float(rec["pt"][0]), 2),
                        round(float(rec["pt"][1]), 2),
                    ],
                    saddle={
                        k: round(float(v), 4)
                        for k, v in rec["saddle"].items()
                    },
                ),
                parent_ids=[],
            )
        )

    debug = dict(
        base_final=len(base_boxes),
        old_graph_additions=old_graph_additions,
        seeded_connected_additions=len(additions),
        typical=[round(float(v), 3) for v in typical],
        seed_audit=seed_audit,
        old_debug=old_debug,
    )

    return final_boxes, debug, response
