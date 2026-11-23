"""Petri-dish circle estimation using scored Hough candidates.

Adapted for ColonyVision from the dish-crop strategy used in the local AST-MEASURE
project: candidate circles are not accepted by center/radius prior alone. The rim
must have gradient support around the circumference and across all quadrants.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from PIL import Image


def _sample_circle(cx: float, cy: float, r: float, count: int = 360) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((cx + r * np.cos(ang), cy + r * np.sin(ang)))


def _ring_gradient_values(gradient: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """Sample a thin radial band so a 1-2 px Hough radius offset is tolerated."""
    h, w = gradient.shape
    vals = []
    for dr in (-3.0, -1.5, 0.0, 1.5, 3.0):
        pts = _sample_circle(cx, cy, max(2.0, r + dr))
        xs = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        vals.append(gradient[ys, xs])
    return np.max(np.stack(vals, axis=1), axis=1)


def _score_circle(gray: np.ndarray, gradient: np.ndarray, circle, valid_rect):
    cx, cy, r = map(float, circle)
    h, w = gray.shape
    x0, y0, x1, y1 = valid_rect

    pts = _sample_circle(cx, cy, r)
    valid = (
        (pts[:, 0] >= x0)
        & (pts[:, 0] < x1)
        & (pts[:, 1] >= y0)
        & (pts[:, 1] < y1)
    )
    valid_fraction = float(np.mean(valid))
    if valid_fraction < 0.82:
        return -1e9, dict(
            support=0.0,
            quadrant_support=0.0,
            strength=0.0,
            valid_fraction=valid_fraction,
            center_penalty=1.0,
            border_penalty=1.0,
        )

    ring = _ring_gradient_values(gradient, cx, cy, r)
    # Ignore samples in padded canvas.
    ring = ring[valid]
    valid_pts = pts[valid]

    edge_threshold = max(7.0, float(np.percentile(gradient[y0:y1, x0:x1], 80)))
    strong = ring >= edge_threshold
    support = float(np.mean(strong))
    strength = float(np.percentile(ring, 80))

    quadrants = []
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        q = ((valid_pts[:, 0] - cx) * sx >= 0) & ((valid_pts[:, 1] - cy) * sy >= 0)
        quadrants.append(float(np.mean(strong[q])) if np.any(q) else 0.0)
    quadrant_support = min(quadrants)

    content_cx = (x0 + x1) * 0.5
    content_cy = (y0 + y1) * 0.5
    content_min = max(1.0, min(x1 - x0, y1 - y0))
    center_penalty = float(math.hypot(cx - content_cx, cy - content_cy) / content_min)

    # Penalize circles that extend outside the real (unpadded) image, not merely canvas.
    border_margin = min(cx - x0, cy - y0, x1 - 1 - cx, y1 - 1 - cy) - r
    border_penalty = max(0.0, float(-border_margin / content_min))

    # The edge evidence dominates. Center is only a weak prior.
    score = (
        support * 95.0
        + quadrant_support * 60.0
        + strength * 0.20
        + valid_fraction * 15.0
        - center_penalty * 22.0
        - border_penalty * 260.0
    )
    return score, dict(
        support=support,
        quadrant_support=quadrant_support,
        strength=strength,
        valid_fraction=valid_fraction,
        center_penalty=center_penalty,
        border_penalty=border_penalty,
    )


def estimate_plate_hough(image: Image.Image, canvas_size: int = 600, return_debug: bool = False):
    """Return (cx, cy, radius, param2) in EXIF-upright original coordinates.

    When return_debug=True returns (circle_or_none, debug_dict).
    """
    src = image.convert("RGB")
    w, h = src.size
    scale = min(canvas_size / w, canvas_size / h)
    tw = max(1, round(w * scale))
    th = max(1, round(h * scale))

    arr = np.asarray(src.resize((tw, th), Image.Resampling.BILINEAR), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    bg = int(np.median(border))
    canvas = np.full((canvas_size, canvas_size), bg, dtype=np.uint8)
    ox = (canvas_size - tw) // 2
    oy = (canvas_size - th) // 2
    canvas[oy : oy + th, ox : ox + tw] = gray
    valid_rect = (ox, oy, ox + tw, oy + th)

    blur = cv2.GaussianBlur(canvas, (9, 9), 2)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(12, 12)).apply(blur)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    content_min = min(tw, th)
    min_r = max(20, int(content_min * 0.31))
    max_r = min(int(canvas_size * 0.54), int(content_min * 0.54))

    candidates = []
    seen = set()
    for source_name, hough_input in (("blur", blur), ("clahe", clahe)):
        for param2 in (18, 22, 26, 30, 36, 44):
            circles = cv2.HoughCircles(
                hough_input,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(20, int(content_min * 0.42)),
                param1=85,
                param2=param2,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is None:
                continue
            for cx, cy, r in circles[0]:
                key = (int(round(cx / 3)), int(round(cy / 3)), int(round(r / 3)))
                if key in seen:
                    continue
                seen.add(key)
                score, detail = _score_circle(canvas, gradient, (cx, cy, r), valid_rect)
                candidates.append(
                    dict(
                        circle=[float(cx), float(cy), float(r)],
                        score=float(score),
                        source=source_name,
                        param2=int(param2),
                        **detail,
                    )
                )

    candidates.sort(key=lambda z: z["score"], reverse=True)
    chosen = None
    for c in candidates:
        if (
            c["support"] >= 0.30
            and c["quadrant_support"] >= 0.12
            and c["valid_fraction"] >= 0.90
            and c["score"] >= 35.0
        ):
            chosen = c
            break

    debug = dict(
        canvas_size=canvas_size,
        scale=scale,
        offset=[ox, oy],
        valid_rect=list(valid_rect),
        min_radius=min_r,
        max_radius=max_r,
        candidates=candidates[:30],
        selected=chosen,
    )

    if chosen is None:
        return (None, debug) if return_debug else None

    cx, cy, r = chosen["circle"]
    result = (
        (cx - ox) / scale,
        (cy - oy) / scale,
        r / scale,
        chosen["param2"],
    )
    return (result, debug) if return_debug else result
