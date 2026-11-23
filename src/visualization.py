from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from preprocessing.plate import crop_plate


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

def ui_bold_font(size=24):
    candidates = [
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    return ui_font(size)

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

    # Square-corner black number tag above each green detection box.
    label_size = max(30, min(44, int(min(out.width, out.height) * 0.014)))
    font_label = ui_bold_font(label_size)
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
            pad_x, pad_y = 7, 6
            label_gap = max(4, int(label_size * 0.12))
            tag_w = tw + 2 * pad_x
            tag_h = th + 2 * pad_y

            # Align to the detection's left edge, but leave a visible gap.
            bx1 = max(0, min(out.width - tag_w, x1))
            by1 = y1 - tag_h - label_gap
            if by1 < 0:
                by1 = y2 + label_gap
            by1 = max(0, min(out.height - tag_h, by1))
            bx2, by2 = bx1 + tag_w, by1 + tag_h

            # Sharp corners, black background, centered white digit(s).
            draw.rectangle((bx1, by1, bx2, by2), fill="black")
            draw.text(
                (bx1 + pad_x - tb[0], by1 + pad_y - tb[1]),
                label,
                fill="white",
                font=font_label,
            )

    if title:
        bar_h = 68
        draw.rectangle((0, 0, min(out.width, 1700), bar_h), fill="white")
        draw.text((16, 18), title, fill="black", font=ui_font(26))

    return out

def draw_final_count(im: Image.Image, count: int):
    out = im.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    font_size = max(34, min(52, int(min(out.width, out.height) * 0.016)))
    font = ui_bold_font(font_size)

    suffix = " (TNTC)" if int(count) > 300 else ""
    text = f"Colony Count: {int(count)}{suffix}"

    # Leave room between the badge and image edges, and around its text.
    edge_margin = max(26, int(min(out.width, out.height) * 0.018))
    x, y = edge_margin, edge_margin
    pad_x, pad_y = 22, 14
    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]

    # Square red count badge without a black border.
    draw.rectangle(
        (x, y, x + tw + 2 * pad_x, y + th + 2 * pad_y),
        fill="#c80000",
    )
    draw.text(
        (x + pad_x - tb[0], y + pad_y - tb[1]),
        text,
        font=font,
        fill="white",
    )
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
