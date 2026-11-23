from __future__ import annotations

import math

import numpy as np

import detection as pt


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
