"""Image-evidence helpers used by the final ColonyVision post-processing pipeline."""
from __future__ import annotations

import math

import cv2
import numpy as np

from detection import e3_radial

def _clip_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [x1, y1, x2, y2]

def _center(box):
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=float)

def _wh(box):
    return np.array([box[2] - box[0], box[3] - box[1]], dtype=float)

def _inside_circle(pt, circle, margin=0.0):
    cx, cy, radius = map(float, circle)
    return math.hypot(float(pt[0]) - cx, float(pt[1]) - cy) <= max(0.0, radius - margin)

def _box_ios(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    inter = float(np.maximum(0.0, rb - lt).prod())
    aa = float(np.maximum(0.0, a[2:] - a[:2]).prod())
    bb = float(np.maximum(0.0, b[2:] - b[:2]).prod())
    return inter / max(1.0, min(aa, bb))

def evidence(gray, q, circle):
    """Re-center an existing proposal on its strongest local filled bright core."""
    b = np.array(q["box"], dtype=float)
    wh = np.maximum(1.0, b[2:] - b[:2])
    c = (b[:2] + b[2:]) * 0.5
    d = float(min(wh))
    x0, y0 = np.maximum(0, np.floor(c - wh)).astype(int)
    x1, y1 = np.minimum([gray.shape[1], gray.shape[0]], np.ceil(c + wh)).astype(int)
    patch = gray[y0:y1, x0:x1].astype(np.float32)

    if min(patch.shape) < 5 or d < 5:
        return dict(support=False, peak=c.tolist(), signal=0.0, reason="small_patch")

    smooth = cv2.GaussianBlur(patch, (0, 0), max(0.7, d * 0.065))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr = np.sqrt(((xx - c[0]) / wh[0]) ** 2 + ((yy - c[1]) / wh[1]) ** 2)
    central = rr <= 0.31
    annulus = (rr >= 0.62) & (rr <= 0.92)

    if not central.any() or not annulus.any():
        return dict(support=False, peak=c.tolist(), signal=0.0, reason="invalid_patch")

    search = np.where(central, smooth, -np.inf)
    iy, ix = np.unravel_index(search.argmax(), search.shape)
    peak = np.array([ix + x0, iy + y0], dtype=float)
    bg = float(np.percentile(smooth[annulus], 35))
    signal = float(smooth[iy, ix] - bg)

    recentered = np.r_[peak - wh / 2.0, peak + wh / 2.0]
    radial = e3_radial(gray, recentered, circle)

    core_mask = (smooth > bg + 0.50 * max(signal, 1.0)).astype("uint8")
    _, labels, stats, _ = cv2.connectedComponentsWithStats(core_mask, 8)
    lab = int(labels[iy, ix])
    area = float(stats[lab, cv2.CC_STAT_AREA]) if lab else 0.0
    fraction = area / float(max(1.0, wh.prod()))

    support = (
        radial["kind"] == "support"
        and signal >= 10
        and 0.12 <= fraction <= 1.40
    )
    if not support and radial["kind"] != "reject" and len(q.get("supporting_ids", [])) >= 2:
        support = (
            radial.get("core", 0) >= 6
            and radial.get("filled_fraction", 0) >= 0.65
            and signal >= 9
            and 0.17 <= fraction <= 1.15
            and radial.get("aspect", 0) >= 0.68
            and radial.get("ring_drop", 99) < 10
        )
    if q.get("tile_edges") and radial.get("aspect", 0) < 0.62:
        support = False

    # Old accepted detections are not automatically trusted. They get only a
    # weaker fallback when the local image still looks like a filled object.
    # Hollow rings, scratches and very weak reflections therefore can be rejected.
    if (
        not support
        and radial.get("kind") != "reject"
        and not q.get("review_needed", False)
        and signal >= 5.5
        and 0.08 <= fraction <= 1.60
        and radial.get("aspect", 0) >= 0.50
        and radial.get("filled_fraction", 0) >= 0.45
    ):
        support = True

    if radial.get("kind") == "reject":
        reason = radial.get("reason", "ring_artifact")
    elif support:
        reason = "recentered_filled_core"
    elif radial.get("aspect", 1) < 0.48:
        reason = "elongated_scratch_or_fragment"
    else:
        reason = "insufficient_local_evidence"

    return dict(
        support=bool(support),
        artifact=bool(radial.get("kind") == "reject"),
        peak=peak.tolist(),
        signal=round(signal, 3),
        component_fraction=round(fraction, 4),
        radial=radial,
        reason=reason,
    )

def typical_geometry(rows):
    """Estimate normal colony box geometry from the current image only."""
    good = []
    for q in rows:
        b = np.asarray(q["box"], dtype=float)
        wh = b[2:] - b[:2]
        if min(wh) >= 5 and max(wh) / max(1.0, min(wh)) <= 1.65 and not q.get("review_needed", False):
            good.append(wh)
    if len(good) < 5:
        good = []
        for q in rows:
            b = np.asarray(q["box"], dtype=float)
            wh = b[2:] - b[:2]
            if min(wh) >= 5:
                good.append(wh)
    if not good:
        return np.array([24.0, 24.0], dtype=float)

    arr = np.asarray(good, dtype=float)
    med = np.median(arr, axis=0)
    side = float(np.median(np.sqrt(arr[:, 0] * arr[:, 1])))
    # Do not let one axis become unrealistically thin.
    med = np.maximum(med, side * 0.70)
    return med

def _local_peak_candidates(gray, search_box, typical, circle):
    """Detect visual colony centers with local contrast + distance-transform peaks."""
    h, w = gray.shape[:2]
    tw, th = map(float, typical)
    side = max(6.0, math.sqrt(tw * th))
    margin = 0.45 * side

    x1, y1, x2, y2 = map(float, search_box)
    x1 = int(max(0, math.floor(x1 - margin)))
    y1 = int(max(0, math.floor(y1 - margin)))
    x2 = int(min(w, math.ceil(x2 + margin)))
    y2 = int(min(h, math.ceil(y2 + margin)))
    patch = gray[y1:y2, x1:x2].astype(np.float32)

    if min(patch.shape) < 8:
        return [], dict(reason="small_search_patch")

    sigma_small = max(0.7, side * 0.075)
    sigma_large = max(2.0, side * 0.62)
    small = cv2.GaussianBlur(patch, (0, 0), sigma_small)
    large = cv2.GaussianBlur(patch, (0, 0), sigma_large)
    response = np.maximum(0.0, small - large)

    # Robust local threshold. Because this is only applied around detector proposals,
    # we can use a permissive threshold and let the two independent peak cues agree.
    med = float(np.median(response))
    mad = float(np.median(np.abs(response - med))) + 1e-6
    maxv = float(response.max())
    threshold = max(2.5, med + 2.2 * mad, 0.20 * maxv)

    fg = (response >= threshold).astype(np.uint8)
    kclose = max(3, int(round(side * 0.13)))
    if kclose % 2 == 0:
        kclose += 1
    kclose = min(kclose, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kclose, kclose))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    kd = max(3, int(round(side * 0.48)))
    if kd % 2 == 0:
        kd += 1
    kd = min(kd, 31)
    dil_dist = cv2.dilate(dist, np.ones((kd, kd), np.uint8))
    dist_peaks = (dist >= dil_dist - 1e-6) & (dist >= max(1.4, 0.13 * side))

    kr = max(3, int(round(side * 0.58)))
    if kr % 2 == 0:
        kr += 1
    kr = min(kr, 35)
    dil_resp = cv2.dilate(response, np.ones((kr, kr), np.uint8))
    resp_peaks = (response >= dil_resp - 1e-6) & (response >= threshold)

    ys, xs = np.where(dist_peaks | resp_peaks)
    raw = []
    dmax = float(max(1e-6, dist.max()))
    rmax = float(max(1e-6, response.max()))
    # The expanded patch is context only. A newly discovered center must still
    # belong to the original proposal union (small tolerance allowed).
    own_margin = 0.12 * side
    sx1, sy1, sx2, sy2 = map(float, search_box)
    for py, px in zip(ys.tolist(), xs.tolist()):
        gx, gy = float(px + x1), float(py + y1)
        if not _inside_circle((gx, gy), circle, margin=1.0):
            continue
        if not (sx1 - own_margin <= gx <= sx2 + own_margin and sy1 - own_margin <= gy <= sy2 + own_margin):
            continue
        score = float(response[py, px] / rmax + 0.70 * dist[py, px] / dmax)
        raw.append((score, gx, gy, float(response[py, px]), float(dist[py, px])))

    # Independent local circle cue, inspired by the object-circle separation used
    # in AST-MEASURE. It is only accepted when the Hough center also has local
    # positive contrast, so glass/rim circles alone do not create counts.
    hough_count = 0
    hough_img = np.uint8(np.clip(small, 0, 255))
    circles = cv2.HoughCircles(
        hough_img,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(4.0, 0.45 * side),
        param1=55,
        param2=max(7.0, 0.18 * side),
        minRadius=max(2, int(round(0.16 * side))),
        maxRadius=max(3, int(round(0.68 * side))),
    )
    if circles is not None:
        for hx, hy, hr in circles[0]:
            px = int(np.clip(round(float(hx)), 0, response.shape[1] - 1))
            py = int(np.clip(round(float(hy)), 0, response.shape[0] - 1))
            gx, gy = float(hx + x1), float(hy + y1)
            if not _inside_circle((gx, gy), circle, margin=1.0):
                continue
            if not (sx1 - own_margin <= gx <= sx2 + own_margin and sy1 - own_margin <= gy <= sy2 + own_margin):
                continue
            if float(response[py, px]) < 0.55 * threshold:
                continue
            score = float(0.80 + response[py, px] / rmax + 0.25 * dist[py, px] / dmax)
            raw.append((score, gx, gy, float(response[py, px]), float(dist[py, px])))
            hough_count += 1

    raw.sort(reverse=True)
    kept = []
    min_sep = max(5.0, 0.44 * side)
    for rec in raw:
        pt = np.array(rec[1:3], dtype=float)
        if all(float(np.linalg.norm(pt - np.array(k[1:3]))) >= min_sep for k in kept):
            kept.append(rec)

    diagnostics = dict(
        reason="visual_peaks",
        patch=[x1, y1, x2, y2],
        threshold=round(threshold, 3),
        response_max=round(maxv, 3),
        foreground_fraction=round(float(fg.mean()), 5),
        raw_peak_count=len(raw),
        local_hough_count=hough_count,
        nms_peak_count=len(kept),
    )
    return kept, diagnostics

def _expand_box(box, amount, width, height):
    x1, y1, x2, y2 = map(float, box)
    return [
        max(0.0, x1 - amount),
        max(0.0, y1 - amount),
        min(float(width), x2 + amount),
        min(float(height), y2 + amount),
    ]

def _enclosing_ring_support(gray, box):
    """Measure a larger circular edge surrounding a candidate.

    True colonies have their main boundary close to the candidate box radius.
    Glass/bubble defects can instead show a second, larger ring centered on the
    candidate. We sample radial Sobel maxima from 0.75-1.85 box-sides and look
    for many angular rays agreeing on the same large radius.
    """
    b = np.asarray(box, dtype=float)
    c = _center(b)
    wh = np.maximum(1.0, _wh(b))
    side = float(math.sqrt(float(wh.prod())))
    if side < 8:
        return dict(support=0.0, strong_rays=0, radius_band=None)

    R = int(round(2.10 * side))
    x1 = max(0, int(c[0] - R))
    y1 = max(0, int(c[1] - R))
    x2 = min(gray.shape[1], int(c[0] + R + 1))
    y2 = min(gray.shape[0], int(c[1] + R + 1))
    patch = gray[y1:y2, x1:x2].astype(np.float32)
    if min(patch.shape) < 12:
        return dict(support=0.0, strong_rays=0, radius_band=None)

    smooth = cv2.GaussianBlur(patch, (0, 0), 1.2)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    pc = np.array([c[0] - x1, c[1] - y1], dtype=float)
    angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    radii = np.arange(0.75 * side, 1.85 * side, 2.0)

    chosen_r = []
    chosen_g = []
    for angle in angles:
        xs = np.rint(pc[0] + radii * math.cos(angle)).astype(int)
        ys = np.rint(pc[1] + radii * math.sin(angle)).astype(int)
        valid = (
            (xs >= 0)
            & (xs < grad.shape[1])
            & (ys >= 0)
            & (ys < grad.shape[0])
        )
        if not np.any(valid):
            continue
        vals = grad[ys[valid], xs[valid]]
        rr = radii[valid]
        k = int(np.argmax(vals))
        chosen_r.append(float(rr[k] / side))
        chosen_g.append(float(vals[k]))

    if not chosen_r:
        return dict(support=0.0, strong_rays=0, radius_band=None)

    positive_grad = grad[grad > 1.0]
    if positive_grad.size < 12:
        return dict(
            support=0.0,
            strong_rays=0,
            radius_band=None,
            gradient_threshold=0.0,
        )
    # Do not allow a flat synthetic/background patch to obtain support from
    # zero-gradient samples. Real glass/bubble rings have strong edge magnitude.
    grad_threshold = max(8.0, float(np.percentile(positive_grad, 65)))
    bins = np.arange(0.75, 1.90, 0.12)
    hist = np.zeros(len(bins) - 1, dtype=int)
    strong = 0
    for radius_norm, strength in zip(chosen_r, chosen_g):
        if strength < grad_threshold:
            continue
        strong += 1
        j = int(np.searchsorted(bins, radius_norm, side="right") - 1)
        if 0 <= j < len(hist):
            hist[j] += 1

    best = int(np.argmax(hist)) if len(hist) else 0
    support = float(hist[best] / max(1, len(angles))) if len(hist) else 0.0
    band = [float(bins[best]), float(bins[best + 1])] if len(hist) else None
    return dict(
        support=round(support, 4),
        strong_rays=int(strong),
        radius_band=band,
        gradient_threshold=round(grad_threshold, 3),
    )

def _large_ring_surface_artifact(gray, box, typical, outer_circle=None):
    """Detect a small bright patch that lies on a much larger circular defect.

    This targets bubbles / glass rings where the detector sees a bright arc segment
    as if it were a colony. The candidate itself can look filled, so ordinary
    radial tests fail. We instead search a larger local Hough circle and ask whether
    the candidate center lies on that circle's circumference.
    """
    b = np.asarray(box, dtype=float)
    c = _center(b)
    side = max(6.0, float(math.sqrt(float(np.prod(typical)))))

    R = int(3.0 * side)
    x1 = max(0, int(c[0] - R))
    y1 = max(0, int(c[1] - R))
    x2 = min(gray.shape[1], int(c[0] + R))
    y2 = min(gray.shape[0], int(c[1] + R))
    patch = gray[y1:y2, x1:x2]
    if min(patch.shape) < 24:
        return dict(is_artifact=False, reason="large_ring_patch_small")

    blur = cv2.GaussianBlur(patch, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(10.0, 0.60 * side),
        param1=70,
        param2=max(12.0, 0.25 * side),
        # Search broadly because OpenCV Hough's candidate set changes when
        # minRadius is too restrictive. Decision below still accepts only circles
        # >= 1.60 colony-side units.
        minRadius=max(12, int(0.80 * side)),
        maxRadius=max(16, int(2.60 * side)),
    )
    if circles is None:
        return dict(is_artifact=False, reason="no_large_circle")

    best = None
    best_large = None
    for hx, hy, hr in circles[0]:
        gx = float(hx + x1)
        gy = float(hy + y1)
        rr = float(hr)
        d = float(math.hypot(float(c[0]) - gx, float(c[1]) - gy))
        radial_error = abs(d - rr) / side
        radius_units = rr / side
        rec = dict(
            center=[round(gx, 2), round(gy, 2)],
            radius=round(rr, 2),
            radius_box_units=round(radius_units, 4),
            radial_error=round(radial_error, 4),
        )
        if best is None or radial_error < best["radial_error"]:
            best = rec
        if radius_units >= 1.60:
            if best_large is None or radial_error < best_large["radial_error"]:
                best_large = rec

    rim_fraction = None
    if outer_circle is not None:
        ocx, ocy, oradius = map(float, outer_circle)
        rim_fraction = float(
            math.hypot(float(c[0]) - ocx, float(c[1]) - ocy)
            / max(oradius, 1.0)
        )

    is_artifact = bool(
        best_large is not None
        and best_large["radial_error"] <= 0.10
        and rim_fraction is not None
        and rim_fraction >= 0.84
    )
    return dict(
        is_artifact=is_artifact,
        reason=(
            "candidate_on_large_surface_ring"
            if is_artifact
            else "large_ring_not_aligned"
        ),
        best=best,
        best_large=best_large,
        rim_fraction=(None if rim_fraction is None else round(rim_fraction, 4)),
    )

def _hard_artifact_reason(gray, q, circle, outer_circle, ev):
    """Return a deterministic rejection reason only for visually implausible colonies."""
    box=np.asarray(q["box"],dtype=float)
    wh=np.maximum(1.0,box[2:]-box[:2])
    aspect=float(min(wh)/max(wh))
    radial=ev.get("radial",{})

    if radial.get("kind")=="reject":
        return radial.get("reason","hollow_ring_artifact")

    # Thin scratches / streak fragments are a recurring false positive in gel.
    if aspect < 0.24 and ev.get("signal",0) < 28:
        return "thin_surface_scratch"

    # A region with essentially no filled bright core is not promoted merely
    # because an earlier detector accepted it.
    if (
        ev.get("signal",0) < 8
        and radial.get("core",0) < 4
        and radial.get("filled_fraction",0) < 0.55
    ):
        return "weak_unfilled_surface_mark"

    # Physical rim glare: only apply very close to the AST-fitted OUTER dish rim
    # and require another implausibility cue so real edge colonies are retained.
    if outer_circle is not None:
        c=_center(box);cx,cy,R=map(float,outer_circle)
        rfrac=float(math.hypot(float(c[0])-cx,float(c[1])-cy)/max(R,1.0))
        if rfrac >= 0.965 and (
            aspect < 0.62
            or radial.get("core",0) < 7
            or radial.get("filled_fraction",0) < 0.62
        ):
            return "physical_rim_glare"

    return None

def _strong_local_centers(gray, box, typical, circle, outer_circle, expand=0.30):
    """Return conservative image-derived colony centers around one ROI.

    Every returned center must pass a normalized full-size local colony test.
    This helper is used only for restore/split decisions; detector geometry alone
    cannot create a center.
    """
    tw, th = map(float, typical)
    side = max(6.0, math.sqrt(tw * th))
    search = _expand_box(
        box, expand * side, gray.shape[1], gray.shape[0]
    )
    peaks, diag = _local_peak_candidates(
        gray, search, typical, circle
    )
    records = []
    for score, x, y, response, distance in peaks:
        provisional = _clip_box(
            [
                x - 0.50 * tw,
                y - 0.50 * th,
                x + 0.50 * tw,
                y + 0.50 * th,
            ],
            gray.shape[1],
            gray.shape[0],
        )
        candidate = dict(
            box=provisional,
            review_needed=True,
            supporting_ids=[],
            tile_edges=[],
        )
        ev = evidence(gray, candidate, circle)
        radial = ev.get("radial", {})
        reason = _hard_artifact_reason(
            gray, candidate, circle, outer_circle, ev
        )
        ring = _enclosing_ring_support(gray, provisional)
        ring_band = ring.get("radius_band")
        strong_outer_ring = (
            ring.get("support", 0.0) >= 0.30
            and ring.get("strong_rays", 0) >= 15
            and ring_band is not None
            and float(ring_band[0]) >= 0.90
        )
        solid = (
            reason is None
            and ev.get("support", False)
            and not strong_outer_ring
            and ev.get("signal", 0.0) >= 9.0
            and radial.get("kind") != "reject"
            and radial.get("aspect", 0.0) >= 0.58
            and radial.get("filled_fraction", 0.0) >= 0.50
            and radial.get("core", 0.0) >= 5.0
        )
        if not solid:
            continue
        records.append(
            dict(
                pt=[float(x), float(y)],
                score=float(score),
                provisional_box=[
                    round(float(v), 2) for v in provisional
                ],
                evidence=ev,
                enclosing_ring=ring,
                source="conservative_local_center",
            )
        )

    # One visual colony may yield several nearby maxima.
    records.sort(key=lambda r: r["score"], reverse=True)
    kept = []
    min_sep = 0.50 * side
    for rec in records:
        pt = np.asarray(rec["pt"], dtype=float)
        if all(
            float(
                np.linalg.norm(
                    pt - np.asarray(k["pt"], dtype=float)
                )
            ) >= min_sep
            for k in kept
        ):
            kept.append(rec)
    return kept, diag
