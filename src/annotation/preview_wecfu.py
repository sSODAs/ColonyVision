"""Visualize saved WeCFU colony predictions without rerunning inference.
Outputs per-image overlays, original-vs-overlay previews, a contact sheet and HTML gallery.
Usage: python src/annotation/preview_wecfu.py
"""
import argparse
import csv
import html
import json
import re
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT / "annotations" / "pseudo_labels" / "wecfu_ultra" / "wecfu_ultra_export.zip"
DEFAULT_IMAGES = ROOT / "raw" / "wecfu_ready"
DEFAULT_OUT = ROOT / "outputs" / "wecfu_ultra_preview"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def font(size):
    candidates = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()

def annotated(image, marks):
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    thickness = max(3, int(min(result.size) / 650))
    dot = max(5, thickness + 2)
    text_font = font(max(20, int(min(result.size) / 100)))
    for idx, item in enumerate(marks, start=1):
        cx = float(item["cx"])
        cy = float(item["cy"])
        r = max(3.0, float(item["r"]))
        # Yellow circle = WeCFU reported center/radius, NOT verified colony outline.
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline="#fff200", width=thickness + 2)
        draw.ellipse((cx-dot, cy-dot, cx+dot, cy+dot), fill="#ff3030", outline="white", width=2)
        if len(marks) <= 65:
            draw.text((cx + r + 4, cy-r-4), str(idx), font=text_font,
                      fill="white", stroke_width=3, stroke_fill="#242424")
    return result

def thumb(image, width, height):
    preview = image.copy()
    preview.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#24272c")
    canvas.paste(preview, ((width-preview.width)//2,(height-preview.height)//2))
    return canvas

def compare(original, marked, name, count):
    pw, ph = 585, 790
    out = Image.new("RGB", (pw*2+40,ph+115), "#171b22")
    drawer = ImageDraw.Draw(out)
    titlefont = font(23)
    subfont = font(19)
    drawer.text((20,14),name[:75],font=titlefont,fill="#ffffff")
    drawer.text((20,55),f"WeCFU candidate markers: {count}  |  NOT VERIFIED",font=subfont,fill="#ffce74")
    out.paste(thumb(original,pw,ph),(15,100))
    out.paste(thumb(marked,pw,ph),(pw+25,100))
    drawer.text((23,878),"ORIGINAL",font=subfont,fill="#f4f4f4")
    drawer.text((pw+33,878),"WE CFU DETECTIONS",font=subfont,fill="#f4f4f4")
    return out

def sheet(entries,out):
    columns = 4
    cw,ch = 272,340
    rows = (len(entries)+columns-1)//columns
    sheet = Image.new("RGB",(cw*columns,max(rows,1)*ch+58),"#171b22")
    pen = ImageDraw.Draw(sheet)
    pen.text((16,10),"WeCFU ULTRA / PILOT PREVIEW  -  ALL LABELS UNVERIFIED",
             fill="white",font=font(23))
    for k,item in enumerate(entries):
        x=(k%columns)*cw
        y=(k//columns)*ch+58
        with Image.open(out/"overlays"/item["overlay"]) as image:
            panel=thumb(image,250,277)
        sheet.paste(panel,(x+10,y+8))
        name=item["image"]
        pen.text((x+12,y+290),f"{k+1:02d}. {name[:25]}",fill="white",font=font(14))
        pen.text((x+12,y+313),f"Detected: {item['count']} | hint: {item['filename_hint']}",
                 fill="#f6c75b",font=font(14))
    return sheet

def filename_hint(name):
    stem=Path(name).stem
    match=re.search(r"_([0-9]+)$",stem)
    return match.group(1) if match else "?"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,default=DEFAULT_ARCHIVE)
    p.add_argument("--images",type=Path,default=DEFAULT_IMAGES)
    p.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=p.parse_args()
    args.archive=args.archive.resolve()
    args.images=args.images.resolve()
    args.out=args.out.resolve()
    if not args.archive.exists():
        raise SystemExit("Missing WeCFU archive: "+str(args.archive))
    if not args.images.exists():
        raise SystemExit("Missing WeCFU image folder: "+str(args.images))
    output=args.out
    (output/"overlays").mkdir(parents=True,exist_ok=True)
    (output/"comparisons").mkdir(parents=True,exist_ok=True)
    entries=[]
    with zipfile.ZipFile(args.archive) as z:
        records=sorted(name for name in z.namelist() if name.startswith("detections/") and name.endswith(".json"))
        if not records:
            raise SystemExit("No WeCFU detections JSON files in ZIP")
        for n,entry in enumerate(records,1):
            base=Path(entry.removeprefix("detections/").removesuffix(".json")).name
            image_path=args.images/base
            if not image_path.is_file():
                raise FileNotFoundError(f"Cannot match WeCFU saved coordinates to image: {image_path}")
            state=json.loads(z.read(entry))
            with Image.open(image_path) as raw:
                if raw.getexif().get(274,1) not in (1,None):
                    raise ValueError(f"{base} still has EXIF rotation; image not aligned")
                original=raw.convert("RGB")
            saved=state.get("image_size",{})
            if (saved.get("width"), saved.get("height")) != original.size:
                raise ValueError(f"{base}: saved prediction geometry does not match image {original.size} / {saved}")
            marks=[m for m in state.get("detections",[]) if m.get("accepted",True)]
            for mark in marks:
                if not (0<=float(mark["cx"])<=original.width and 0<=float(mark["cy"])<=original.height):
                    raise ValueError(f"Out of bounds marker in {base}")
            overlay=annotated(original,marks)
            overlay_name=Path(base).stem + "_overlay.jpg"
            comparison_name=Path(base).stem + "_compare.jpg"
            overlay.save(output/"overlays"/overlay_name,quality=92,subsampling=0)
            compare(original,overlay,base,len(marks)).save(output/"comparisons"/comparison_name,quality=89)
            entries.append({
                "image":base,"count":len(marks),"filename_hint":filename_hint(base),
                "reviewed":str(bool(state.get("reviewed",False))).lower(),
                "overlay":overlay_name,"comparison":comparison_name,
            })
            print(f"[{n}/{len(records)}] {base}: {len(marks)} candidate markers",flush=True)
    sheet(entries,output).save(output/"overview.jpg",quality=90)
    with (output/"preview_counts.csv").open("w",newline="",encoding="utf-8-sig") as file:
        writer=csv.DictWriter(file,fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    gallery=['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ColonyVision WeCFU Preview</title>',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<style>body{background:#12171d;color:#e9eef4;font-family:Segoe UI,Arial,sans-serif;margin:24px}h1{font-size:24px}',
        'p{color:#c7d0d9} .gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:20px}',
        'article{background:#242d37;border-radius:10px;padding:12px}article img{width:100%;height:320px;object-fit:contain;background:#141619}',
        'a{color:#a7d5ff} strong{color:#ffd46b}small{color:#c7d0d9}</style></head><body>',
        '<h1>ColonyVision — WeCFU Ultra Preview</h1>',
        '<p>YELLOW = estimated center/radius, RED = center. Candidate markers are NOT reviewed bounding boxes.</p>',
        '<p>Hint is an unverified number parsed from filename, NOT ground truth. Click image to see original vs overlay.</p>',
        f'<p>Images: {len(entries)} | Total candidate markers: {sum(e["count"] for e in entries)}</p>',
        '<div class="gallery">']
    for e in entries:
        name=html.escape(e["image"])
        gallery.append('<article><a href="comparisons/'+html.escape(e["comparison"],quote=True)+'"><img loading="lazy" src="overlays/'+html.escape(e["overlay"],quote=True)+'" alt="'+name+'"></a>')
        gallery.append('<div>'+name+'</div><div><strong>Predicted: '+str(e["count"])+'</strong> | Filename hint: '+html.escape(e["filename_hint"])+'</div>')
        gallery.append('<small>Reviewed: '+html.escape(e["reviewed"])+' | <a href="overlays/'+html.escape(e["overlay"],quote=True)+'">Full resolution</a></small></article>')
    gallery.append('</div></body></html>')
    (output/"index.html").write_text("\n".join(gallery),encoding="utf-8")
    print(f"Output: {output}\nOpen: {output/'index.html'}\nOverview: {output/'overview.jpg'}",flush=True)

if __name__=="__main__":
    main()
