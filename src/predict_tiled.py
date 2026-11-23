"""ColonyVision tiled inference and paired E1/E2 experiments.
Legacy infer() and CLI remain available for existing Teacher scripts.
Experiment mode uses one Student checkpoint, common counting region and settings.
Predictions are not independently verified ground truth.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def positions(length, tile, stride):
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def infer(model, image, tile, stride, conf, imgsz, iou, device, *, pil_input=False, audit=None, patch_transform=None):
    """Return original working-image xyxy/conf and number before global NMS.

    Legacy callers preserve their input convention. Both paired experiments
    pass PIL RGB images explicitly instead of ambiguous NumPy channel order.
    """
    import torch
    import torchvision
    w, h = image.size
    candidates = []
    if audit is not None and audit:
        raise ValueError("audit must start empty")
    for top in positions(h, tile, stride):
        for left in positions(w, tile, stride):
            patch = image.crop((left, top, min(w, left + tile), min(h, top + tile)))
            if patch_transform is not None:
                transformed=patch_transform(patch)
                if transformed.size!=patch.size or transformed.mode!='RGB':raise ValueError('Tile transform changed geometry or mode')
                patch=transformed
            source = patch if pil_input else np.asarray(patch)
            result = model.predict(source=source, device=device, conf=conf,
                                   imgsz=imgsz, max_det=2000, verbose=False)[0]
            if result.boxes is None:
                continue
            for b, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x0, y0, x1, y1 = map(float, b)
                x0, x1 = max(0, min(w, x0 + left)), max(0, min(w, x1 + left))
                y0, y1 = max(0, min(h, y0 + top)), max(0, min(h, y1 + top))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                candidates.append([x0, y0, x1, y1, float(score)])
                if audit is not None:
                    audit.append(dict(id=len(candidates)-1, box=candidates[-1][:4],
                                      confidence=float(score),
                                      tile=[left,top,min(w,left+tile),min(h,top+tile)],
                                      global_nms_kept=False))
    if not candidates:
        return [], 0
    boxes = torch.tensor([c[:4] for c in candidates], dtype=torch.float32)
    scores = torch.tensor([c[4] for c in candidates], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, iou).tolist()
    if audit is not None:
        for i in keep:
            audit[i]['global_nms_kept'] = True
    return [candidates[i] for i in keep], len(candidates)


def annotate(image, marks):
    result = image.copy().convert('RGB')
    draw = ImageDraw.Draw(result)
    width = max(2, int(min(image.size) / 900))
    for x0, y0, x1, y1, score in marks:
        draw.rectangle((x0, y0, x1, y1), outline='#15eb72', width=width)
    return result


def legacy_main():
    """Original single-image/folder CLI, kept for existing workflows."""
    import torch
    from ultralytics import YOLO
    from preprocessing.plate_roi import estimate_plate
    p = argparse.ArgumentParser(description='Legacy tiled inference')
    p.add_argument('--model', type=Path, default=ROOT/'outputs/train/adbc_yolo11n/weights/best.pt')
    p.add_argument('--source', type=Path, default=ROOT/'raw/unlabeled/Petri_plates')
    p.add_argument('--name', default='adbc_tiled_preview')
    p.add_argument('--tile', type=int, default=768)
    p.add_argument('--stride', type=int, default=640)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--iou', type=float, default=.60)
    p.add_argument('--limit', type=int, default=12)
    p.add_argument('--plate-inset', type=float, default=.96)
    p.add_argument('--no-plate-filter', action='store_true')
    p.add_argument('--circular-mask', action='store_true')
    p.add_argument('--strict-roi', action='store_true')
    a = p.parse_args()
    if not a.model.is_file(): raise SystemExit('Checkpoint missing: ' + str(a.model))
    if not 0 < a.stride <= a.tile: raise SystemExit('0 < stride <= tile is required')
    if a.no_plate_filter and (a.circular_mask or a.strict_roi):
        raise SystemExit('--no-plate-filter conflicts with --circular-mask or --strict-roi')
    files = [a.source] if a.source.is_file() else sorted(x for x in a.source.iterdir() if x.is_file() and x.suffix.lower() in EXTS)
    if a.limit > 0: files = files[:a.limit]
    if not files: raise SystemExit('No input images')
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(str(a.model))
    out = ROOT/'outputs/predict'/a.name
    (out/'overlays').mkdir(parents=True, exist_ok=True)
    (out/'detections').mkdir(parents=True, exist_ok=True)
    rows = []
    for n, path in enumerate(files, 1):
        with Image.open(path) as f: upright = ImageOps.exif_transpose(f).convert('RGB')
        plate, status = estimate_plate(upright) if not a.no_plate_filter else (None, 'filter_disabled')
        if plate is None and a.strict_roi:
            record = dict(source=path.name, status='roi_not_found_no_prediction', plate_status=status, count=None, boxes_xyxy_conf=[])
            (out/'detections'/(path.name+'.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
            rows.append([path.name, '', ''])
            print(f'[{n}/{len(files)}] {path.name}: ROI not found; SKIPPED', flush=True)
            continue
        if plate is not None:
            cx, cy, r, _, _ = plate
            pad = 1.06*r
            l, t = max(0, int(cx-pad)), max(0, int(cy-pad))
            rr, b = min(upright.width, int(cx+pad)), min(upright.height, int(cy+pad))
            working = upright.crop((l,t,rr,b))
        else: working, l, t = upright, 0, 0
        if a.circular_mask and plate is not None:
            mask = Image.new('L', working.size, 0); mr = r*1.035
            ImageDraw.Draw(mask).ellipse((cx-l-mr, cy-t-mr, cx-l+mr, cy-t+mr), fill=255)
            working = Image.composite(working, Image.new('RGB', working.size), mask)
        boxes, raw = infer(model, working, a.tile, a.stride, a.conf, a.imgsz, a.iou, device)
        detections = []
        for x0, y0, x1, y1, score in boxes:
            mapped = [x0+l,y0+t,x1+l,y1+t,score]
            if plate is not None and not inside(mapped, (cx,cy,r*a.plate_inset)): continue
            detections.append(mapped)
        overlay = annotate(upright, detections)
        if plate is not None:
            ImageDraw.Draw(overlay).ellipse((cx-r*a.plate_inset,cy-r*a.plate_inset,cx+r*a.plate_inset,cy+r*a.plate_inset), outline='#ffb000', width=5)
        overlay.save(out/'overlays'/(path.stem+'_marked.jpg'), quality=92)
        record = dict(source=path.name, orientation='EXIF_transposed_for_display', image_size={'width':upright.width,'height':upright.height}, model=str(a.model), source_pseudo_label=False, plate_status=status, plate_estimate=list(plate[:3]) if plate is not None else None, plate_inset=a.plate_inset if plate is not None else None, circular_mask_applied=bool(a.circular_mask and plate is not None), boxes_xyxy_conf=detections, count=len(detections))
        (out/'detections'/(path.name+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
        rows.append([path.name,len(detections),raw])
        print(f'[{n}/{len(files)}] {path.name}: {len(detections)} colonies ({raw} before NMS)',flush=True)
    with (out/'counts.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.writer(f);wr.writerow(['image','predicted_count','raw_boxes_before_global_nms']);wr.writerows(rows)
    print('Saved:',out,flush=True)


def inside(box, circle):
    cx, cy, r = circle
    return ((box[0]+box[2])/2-cx)**2 + ((box[1]+box[3])/2-cy)**2 <= r*r


def map_boxes(boxes, offset):
    x, y = offset
    return [[b[0]+x,b[1]+y,b[2]+x,b[3]+y,b[4]] for b in boxes]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_json(path, doc):
    with path.open('x',encoding='utf-8') as f:
        json.dump(doc,f,indent=2,ensure_ascii=False)


def balanced_val(rows, limit=6, seed=42):
    groups = defaultdict(list)
    for r in rows:
        if r['split']=='val': groups[r['species_filename_hint']].append(r)
    if not groups: raise ValueError('No validation images')
    rng=random.Random(seed)
    for key in sorted(groups):
        groups[key].sort(key=lambda r:r['image']);rng.shuffle(groups[key])
    total=sum(map(len,groups.values()))
    wanted=total if limit==0 else limit
    if not 0 < wanted <= total: raise ValueError('Invalid validation sample size')
    chosen=[]
    while len(chosen)<wanted:
        for key in sorted(groups):
            if groups[key] and len(chosen)<wanted: chosen.append(groups[key].pop())
    return [dict(r,selection_id=f'V{i:03d}') for i,r in enumerate(chosen,1)]


def draw_boxes(im, boxes, circle, origin=(0,0), max_size=1150):
    result=im.copy(); w,h=im.size;result.thumbnail((max_size,max_size),Image.Resampling.LANCZOS)
    sx,sy=result.width/w,result.height/h;ox,oy=origin
    def rect(b):return ((b[0]-ox)*sx,(b[1]-oy)*sy,(b[2]-ox)*sx,(b[3]-oy)*sy)
    d=ImageDraw.Draw(result);cx,cy,r=circle
    d.ellipse(rect((cx-r,cy-r,cx+r,cy+r)),outline='#ffba37',width=3)
    for b in boxes:
        d.rectangle(rect(b),outline='#28e891' if inside(b,circle) else '#f65060',width=2)
    return result


def comparison_image(original, enhanced, offset, circle, b1, b2, path):
    # Same crop for visual comparison; baseline model actually saw full image.
    x,y=offset; w,h=enhanced.size
    crop=original.crop((x,y,x+w,y+h))
    a=draw_boxes(crop,b1,circle,offset);b=draw_boxes(enhanced,b2,circle,offset)
    width=a.width+b.width+12; top=50
    combined=Image.new('RGB',(width,top+max(a.height,b.height)+28),'#20242a');d=ImageDraw.Draw(combined)
    combined.paste(a,(0,top));combined.paste(b,(a.width+12,top))
    d.text((6,6),f'E1 ORIGINAL | all={len(b1)} | inside={sum(inside(z,circle) for z in b1)}',fill='white')
    d.text((a.width+18,6),f'E2 CROP + CLAHE | inside={sum(inside(z,circle) for z in b2)}',fill='white')
    d.text((6,25),'Same display crop; E1 inferred on the full image',fill='#c9c9c9')
    d.text((a.width+18,25),'Green = counted; red = outside common circle',fill='#c9c9c9')
    d.text((6,combined.height-20),'UNVERIFIED PREDICTIONS | no additional deduplication or quality rejection',fill='#edcb8a')
    combined.save(path,quality=93)


def experiment_main(argv):
    p=argparse.ArgumentParser(description='Paired E1/E2 inference on validation or frozen Test50')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--limit',type=int,default=None,help='Validation defaults to 6; 0=all. Test always uses all 50.')
    p.add_argument('--model',type=Path,default=ROOT/'outputs/train/colonyvision_student_v1/weights/best.pt')
    p.add_argument('--tile',type=int,default=768);p.add_argument('--stride',type=int,default=640)
    p.add_argument('--imgsz',type=int,default=640);p.add_argument('--conf',type=float,default=.25);p.add_argument('--iou',type=float,default=.60)
    p.add_argument('--clahe-clip',type=float,default=2.0);p.add_argument('--clahe-grid',type=int,default=8)
    p.add_argument('--gamma',type=float,default=1.0)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args(argv)
    if a.self_test:
        self_test();return
    if not (0<a.stride<=a.tile and a.imgsz>0 and 0<a.conf<=1 and 0<a.iou<=1 and a.clahe_clip>=0 and a.clahe_grid>0 and a.gamma>0):p.error('Invalid experiment parameters')
    # Import preflight BEFORE creating output directories. Do not bypass OS policy.
    try:
        import cv2
        import torch
        import torchvision
        import ultralytics
        from ultralytics import YOLO
        from prepare_plate_images import locate_plate, crop_plate, clahe_luma
    except (ImportError,OSError) as exc:
        raise SystemExit('Dependency preflight failed; no outputs created. '+str(exc)) from exc
    split_path=ROOT/'annotations/metadata/teacher_split_1200.csv'
    splits=read_csv(split_path);lookup={r['image']:r for r in splits}
    if len(splits)!=1252 or len(lookup)!=1252:raise SystemExit('Expected 1252 unique original images')
    if a.split=='val':
        rows=balanced_val(splits,6 if a.limit is None else a.limit)
        base=ROOT/'outputs/diagnostics/preprocessing'
        manifest_path=split_path
        run_scope='validation_development_only'
    else:
        if a.limit not in (None,0,50):p.error('Test experiment requires all frozen 50 photos')
        base=ROOT/'outputs/experiments/counting_50'
        manifest_path=base/'manifest_50.csv';rows=read_csv(manifest_path)
        if len(rows)!=50 or set(r['selection_id'] for r in rows)!={f'S{i:03d}' for i in range(1,51)}:raise SystemExit('Invalid frozen Test50 manifest')
        run_scope='frozen_test_unverified_reference'
    if len({r['image'] for r in rows})!=len(rows):raise SystemExit('Duplicate selected images')
    for r in rows:
        parent=lookup.get(r['image']);photo=Path(r['source'])
        if parent is None or parent['split']!=a.split or Path(parent['source']).resolve()!=photo.resolve() or photo.name!=r['image'] or not photo.is_file():raise SystemExit('Input mismatch: '+r['image'])
    if not a.model.is_file():raise SystemExit('Checkpoint missing: '+str(a.model))
    dirs={name:base/name for name in ('baseline','preprocessing','comparison')}
    if any(q.exists() for q in dirs.values()) or (base/'experiment_config.json').exists():raise SystemExit('Experiment outputs already exist. Refusing overwrite; archive only this completed experiment if rerunning.')
    device=0 if torch.cuda.is_available() else 'cpu'
    model=YOLO(str(a.model))
    config=dict(scope=run_scope,split=a.split,selection_ids=[r['selection_id'] for r in rows],model=str(a.model.resolve()),model_sha256=sha(a.model),split_manifest_sha256=sha(split_path),selection_manifest_sha256=sha(manifest_path),tile=a.tile,stride=a.stride,imgsz=a.imgsz,confidence=a.conf,global_nms_iou=a.iou,lab_clahe_clip=a.clahe_clip,clahe_grid=a.clahe_grid,gamma=a.gamma,glare_clip=255,roi_ratio=.93,crop_padding=1.06,hard_mask=False,source_type='PIL_RGB_both_methods',torch_version=str(torch.__version__),ultralytics_version=ultralytics.__version__,opencv_version=cv2.__version__,device=str(device),extra_dedup=False,quality_rejection=False,independent_ground_truth=False)
    base.mkdir(parents=True,exist_ok=True)
    write_json(base/'experiment_config.json',config)
    for key,q in dirs.items():
        q.mkdir();(q/('previews' if key=='comparison' else 'detections')).mkdir()
        if key!='comparison':(q/'overlays').mkdir()
    summaries=[];tables={'baseline':[],'preprocessing':[]}
    for index,row in enumerate(rows,1):
        sid=row['selection_id'];photo=Path(row['source'])
        with Image.open(photo) as f: original=ImageOps.exif_transpose(f).convert('RGB')
        if a.split=='val': outer,roi_source=locate_plate(original);inner=None if outer is None else (*outer[:2],outer[2]*.93)
        else:
            outer=tuple(float(row[k]) for k in ('circle_cx','circle_cy','circle_outer_r'))
            inner=tuple(float(row[k]) for k in ('circle_cx','circle_cy','circle_inner_r'))
            if abs(inner[2]-outer[2]*.93)>.01:raise ValueError('Frozen inner circle differs from agreed 0.93r: '+sid)
            roi_source='frozen_manifest_AI_screened_unverified'
        common=dict(selection_id=sid,image=row['image'],species=row['species_filename_hint'],split=a.split,source_sha256=sha(photo),roi_source=roi_source,image_size={'width':original.width,'height':original.height},circle_outer=outer,circle_inner=inner)
        row_summary=dict(selection_id=sid,image=row['image'],species=row['species_filename_hint'],status='ROI_MISSING',e1_raw='',e1_common_roi='',e1_outside='',e2_raw='',e2_common_roi='',e2_outside='',delta_inside='',e1_seconds='',e2_seconds='')
        if outer is None:
            for k in tables:
                write_json(dirs[k]/'detections'/f'{sid}.json',dict(common,status='ROI_MISSING',count=None,boxes_xyxy_conf=[]))
                tables[k].append(dict(selection_id=sid,image=row['image'],status='ROI_MISSING',raw_count='',count_common_roi='',outside_count='',seconds=''))
            summaries.append(row_summary);print(sid,'ROI_MISSING; paired count skipped',flush=True);continue
        if not all(math.isfinite(z) for z in outer+inner) or not (0<=outer[0]<original.width and 0<=outer[1]<original.height and 0<inner[2]<=outer[2]):raise ValueError('Invalid circle '+sid)
        start=time.perf_counter()
        b1,n1=infer(model,original,a.tile,a.stride,a.conf,a.imgsz,a.iou,device,pil_input=True)
        t1=time.perf_counter()-start
        start=time.perf_counter()
        crop,offset=crop_plate(original,outer,padding=1.06)
        enhanced=clahe_luma(crop,a.clahe_clip,a.clahe_grid,a.gamma,255)
        if enhanced.size!=crop.size:raise ValueError('Preprocessing changed image dimensions '+sid)
        b2,n2=infer(model,enhanced,a.tile,a.stride,a.conf,a.imgsz,a.iou,device,pil_input=True);b2=map_boxes(b2,offset)
        t2=time.perf_counter()-start
        counts=[]
        for k,boxes,raw,sec,off in [('baseline',b1,n1,t1,(0,0)),('preprocessing',b2,n2,t2,offset)]:
            keep=[b for b in boxes if inside(b,inner)];outside=[b for b in boxes if not inside(b,inner)]
            accepted=boxes if k=='baseline' else keep
            record=dict(common,status='UNVERIFIED_PREDICTION',method=k,coordinate_frame='EXIF_upright_original_pixels',crop_origin=off,count=len(accepted),boxes_xyxy_conf=accepted,all_boxes_after_nms=boxes,common_roi_boxes_xyxy_conf=keep,outside_boxes_xyxy_conf=outside,raw_boxes_before_global_nms=raw,count_common_roi=len(keep),outside_count=len(outside),seconds=sec)
            write_json(dirs[k]/'detections'/f'{sid}.json',record)
            tables[k].append(dict(selection_id=sid,image=row['image'],status='UNVERIFIED_PREDICTION',raw_count=len(boxes),count_common_roi=len(keep),outside_count=len(outside),seconds=round(sec,3)))
            draw_boxes(original,boxes,inner).save(dirs[k]/'overlays'/f'{sid}.jpg',quality=93)
            counts.append((len(boxes),len(keep),len(outside)))
        comparison_image(original,enhanced,offset,inner,b1,b2,dirs['comparison']/'previews'/f'{sid}.jpg')
        row_summary.update(status='UNVERIFIED_PREDICTION',e1_raw=counts[0][0],e1_common_roi=counts[0][1],e1_outside=counts[0][2],e2_raw=counts[1][0],e2_common_roi=counts[1][1],e2_outside=counts[1][2],delta_inside=counts[1][1]-counts[0][1],e1_seconds=round(t1,3),e2_seconds=round(t2,3))
        summaries.append(row_summary);print(f'[{index}/{len(rows)}] {sid} E1={counts[0][0]} (inside={counts[0][1]}) E2 inside={counts[1][1]}',flush=True)
    for k,table in tables.items():write_csv(dirs[k]/'counts.csv',table)
    write_csv(dirs['comparison']/'per_image_results.csv',summaries)
    write_json(base/'experiment_complete.json',dict(images=len(rows),paired_predictions=sum(r['status']=='UNVERIFIED_PREDICTION' for r in summaries),roi_missing=sum(r['status']=='ROI_MISSING' for r in summaries),note='Count differences are not accuracy or MAE. No independent count reference used.'))
    print('SAVED:',base,flush=True)


def self_test():
    assert positions(1500,768,640)==[0,640,732]
    assert positions(100,768,640)==[0]
    b=[10,20,30,40,.9];assert map_boxes([b],(100,200))==[[110,220,130,240,.9]]
    assert inside(b,(20,30,5)) and not inside(b,(0,0,5))
    rows=[dict(image=f'{s}{i}',split=k,species_filename_hint=s) for s in 'ABC' for k in ('val','test') for i in range(5)]
    got=balanced_val(rows,6);assert len(got)==6 and all(x['split']=='val' for x in got)
    assert [sum(x['species_filename_hint']==s for x in got) for s in 'ABC']==[2,2,2]
    assert got==balanced_val(rows,6)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as t:
        p=Path(t)/'test.jpg';im=Image.new('RGB',(200,200),'#888888');crop=im.crop((20,20,180,180))
        comparison_image(im,crop,(20,20),(100,100,60),[[55,55,85,85,.9]],[[55,55,85,85,.9]],p)
        assert p.is_file()
    print('[PASS] tile positions, coordinate mapping, ROI center membership, deterministic validation selection, comparison preview')



def coverage_records(records, offset, circle, view):
    """Map all per-tile (already model-NMS'ed) records into original pixels."""
    ox,oy=offset
    output=[]
    for rec in records:
        r=dict(rec)
        r['id']=f'{view}:{rec["id"]}'
        r['view']=view
        r['box']=[v+(ox if i%2==0 else oy) for i,v in enumerate(rec['box'])]
        r['tile']=[v+(ox if i%2==0 else oy) for i,v in enumerate(rec['tile'])]
        r['inside_inner_circle']=inside(r['box'],circle)
        output.append(r)
    return output


def coverage_preview(crop, enhanced, offset, circle, records, path):
    # Show NMS survivors separately and together. Raw tile proposals remain in JSON.
    w,h=crop.size; ox,oy=offset
    def panel(image,views):
        im=image.copy();im.thumbnail((1150,1150),Image.Resampling.LANCZOS)
        sx,sy=im.width/w,im.height/h
        def rect(b):return ((b[0]-ox)*sx,(b[1]-oy)*sy,(b[2]-ox)*sx,(b[3]-oy)*sy)
        d=ImageDraw.Draw(im);cx,cy,r=circle
        d.ellipse(rect([cx-r,cy-r,cx+r,cy+r]),outline='#ffba37',width=3)
        for rec in records:
            if rec['view'] not in views or not rec['global_nms_kept'] or not rec['inside_inner_circle']:continue
            d.rectangle(rect(rec['box']),outline='#25e59c' if rec['view']=='original' else '#41cfff',width=2)
        return im
    panels=[panel(crop,{'original'}),panel(enhanced,{'clahe'}),panel(crop,{'original','clahe'})]
    titles=['Original crop | green','CLAHE crop | cyan','Overlay of both | NOT a count']
    width=sum(p.width for p in panels)+20
    canvas=Image.new('RGB',(width,max(p.height for p in panels)+58),'#20242a');d=ImageDraw.Draw(canvas)
    x=0
    for im,title in zip(panels,titles):
        d.text((x+4,6),title,fill='white');canvas.paste(im,(x,26));x+=im.width+10
    d.text((6,canvas.height-23),'Only per-view global NMS survivors inside ROI shown. Raw per-tile proposals saved separately in JSON. UNVERIFIED.',fill='#edcb8a')
    canvas.save(path,quality=94)


def coverage_main(argv):
    p=argparse.ArgumentParser(description='Complementary crop/CLAHE candidate coverage; not a count benchmark.')
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--limit', type=int, default=None, help='Validation: default 6, 0=all. Test: all frozen 50 only.')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args(argv)
    if a.self_test:
        self_test()
        r=[dict(id=0,box=[10,20,30,40],tile=[0,0,100,100],confidence=.7,global_nms_kept=True)]
        q=coverage_records(r,(100,200),(120,230,50),'original')
        assert q[0]['box']==[110,220,130,240] and q[0]['tile']==[100,200,200,300]
        assert q[0]['inside_inner_circle'] and q[0]['id']=='original:0' and r[0]['box']==[10,20,30,40]
        assert not coverage_records(r,(100,200),(0,0,10),'clahe')[0]['inside_inner_circle']
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as t:
            im=Image.new('RGB',(100,100),'#777777');path=Path(t)/'preview.jpg'
            coverage_preview(im,im,(100,200),(120,230,50),q,path)
            assert path.is_file()
        print('[PASS] Candidate provenance, original-coordinate tiles/boxes, circle membership, preview')
        return
    import cv2,torch,torchvision,ultralytics
    from ultralytics import YOLO
    from prepare_plate_images import crop_plate,clahe_luma,locate_plate
    base=ROOT/'outputs/diagnostics/preprocessing'
    config_path=base/'experiment_config.json';cfg=json.loads(config_path.read_text(encoding='utf-8'))
    complete=json.loads((base/'experiment_complete.json').read_text(encoding='utf-8'))
    if cfg['split']!='val' or cfg['scope']!='validation_development_only' or complete['paired_predictions']!=6 or cfg['selection_ids']!=[f'V{i:03d}' for i in range(1,7)]:raise SystemExit('Expected original completed six-image validation pilot')
    model_path=Path(cfg['model']);split_path=ROOT/'annotations/metadata/teacher_split_1200.csv'
    if sha(model_path)!=cfg['model_sha256'] or sha(split_path)!=cfg['split_manifest_sha256']:raise SystemExit('Model or split identity changed')
    splits=read_csv(split_path);lookup={r['image']:r for r in splits}
    if len(splits)!=1252 or len(lookup)!=1252:raise SystemExit('Invalid original split manifest')
    if a.split=='test':
        if a.limit not in (None,0,50):p.error('Test requires all frozen 50 images')
        manifest_path=ROOT/'outputs/experiments/counting_50/manifest_50.csv'
        chosen=read_csv(manifest_path)
        if len(chosen)!=50 or {r['selection_id'] for r in chosen}!={f'S{i:03d}' for i in range(1,51)}:raise SystemExit('Invalid frozen Test50 manifest')
        out=manifest_path.parent/'candidate_coverage';scope='frozen_test_candidate_coverage_descriptive_only'
    else:
        manifest_path=split_path
        wanted=6 if a.limit is None else a.limit
        chosen=balanced_val(splits,wanted)
        out=base/('candidate_coverage' if wanted==6 else 'coverage_review')
        scope='validation_candidate_coverage_only'
    if len({r['image'] for r in chosen})!=len(chosen):raise SystemExit('Duplicate image selection')
    if out.exists():raise SystemExit('Output exists; inspect existing results, no overwrite: '+str(out))
    jobs=[]
    for selected in chosen:
        sid=selected['selection_id'];row=lookup.get(selected['image']);source=Path(selected['source'])
        if row is None or row['split']!=a.split or Path(row['source']).resolve()!=source.resolve() or source.name!=row['image'] or not source.is_file():raise SystemExit('Image/split identity mismatch '+sid)
        source_sha=sha(source)
        # Compare with original Teacher photo identity, not Teacher detection labels.
        teacher_path=ROOT/'outputs/teacher/adbc_1200/detections'/(row['image']+'.json')
        teacher=json.loads(teacher_path.read_text(encoding='utf-8'))
        if teacher['source_sha256']!=source_sha or teacher.get('split')!=a.split:raise SystemExit('Original photo hash or split changed '+sid)
        with Image.open(source) as f:im=ImageOps.exif_transpose(f).convert('RGB')
        size={'width':im.width,'height':im.height}
        if teacher['image_size']!=size:raise ValueError('Image geometry changed '+sid)
        if a.split=='test':
            outer=tuple(float(selected[k]) for k in ('circle_cx','circle_cy','circle_outer_r'))
            inner=tuple(float(selected[k]) for k in ('circle_cx','circle_cy','circle_inner_r'))
            if abs(inner[2]-outer[2]*.93)>.01:raise ValueError('Frozen ROI differs from 0.93r '+sid)
            roi_source='frozen_manifest_AI_screened_unverified'
        elif sid in cfg['selection_ids']:
            old=json.loads((base/'preprocessing/detections'/f'{sid}.json').read_text(encoding='utf-8'))
            if old['image']!=row['image'] or old['source_sha256']!=source_sha or old['split']!='val' or old['image_size']!=size:raise ValueError('Original validation pilot mismatch '+sid)
            outer=tuple(old['circle_outer']) if old['circle_outer'] else None
            inner=tuple(old['circle_inner']) if old['circle_inner'] else None
            roi_source=old['roi_source']
        else:
            outer,roi_source=locate_plate(im)
            inner=None if outer is None else (*outer[:2],outer[2]*.93)
        if outer is not None and (inner is None or not all(math.isfinite(z) for z in (*outer,*inner)) or not (0<=outer[0]<im.width and 0<=outer[1]<im.height and 0<inner[2]<=outer[2]) or inner[:2]!=outer[:2]):raise ValueError('Invalid ROI '+sid)
        jobs.append((sid,source,dict(image=row['image'],species=row['species_filename_hint'],split=a.split,source_sha256=source_sha,image_size=size,circle_outer=outer,circle_inner=inner,roi_source=roi_source)))
        del im
    model=YOLO(str(model_path));device=0 if torch.cuda.is_available() else 'cpu'
    out.mkdir();(out/'previews').mkdir();(out/'detections').mkdir()
    write_json(out/'config.json',dict(scope=scope,split=a.split,selection_manifest_sha256=sha(manifest_path),parent_config_sha256=sha(config_path),model_sha256=cfg['model_sha256'],selection_ids=[j[0] for j in jobs],tile=cfg['tile'],stride=cfg['stride'],imgsz=cfg['imgsz'],confidence=cfg['confidence'],global_nms_iou=cfg['global_nms_iou'],clahe_clip=cfg['lab_clahe_clip'],clahe_grid=cfg['clahe_grid'],gamma=cfg['gamma'],crop_padding=cfg['crop_padding'],view_paths=['original_crop','clahe_crop'],cross_view_dedup=False,raw_record_scope='after internal YOLO tile NMS, before global NMS',independent_ground_truth=False,device=str(device)))
    rows=[]
    for sid,path,old in jobs:
        with Image.open(path) as f:im=ImageOps.exif_transpose(f).convert('RGB')
        if old['image_size']!={'width':im.width,'height':im.height}:raise ValueError('Image geometry changed '+sid)
        if sha(path)!=old['source_sha256']:raise ValueError('Source changed during run '+sid)
        if old['circle_outer'] is None:
            row=dict(selection_id=sid,image=old['image'],original_tile_proposals='',original_inside_after_global_nms='',clahe_tile_proposals='',clahe_inside_after_global_nms='',status='ROI_MISSING')
            write_json(out/'detections'/f'{sid}.json',dict(old,selection_id=sid,candidates=[],status='ROI_MISSING',independent_ground_truth=False))
            im.thumbnail((1150,1150),Image.Resampling.LANCZOS);ImageDraw.Draw(im).text((8,8),'ROI MISSING - NO COUNT',fill='#ffba37');im.save(out/'previews'/f'{sid}.jpg',quality=94)
            rows.append(row);print(row,flush=True);continue
        outer=tuple(old['circle_outer']);inner=tuple(old['circle_inner'])
        crop,offset=crop_plate(im,outer,padding=cfg['crop_padding'])
        enhanced=clahe_luma(crop,cfg['lab_clahe_clip'],cfg['clahe_grid'],cfg['gamma'],cfg['glare_clip'])
        if crop.size!=enhanced.size:raise ValueError('CLAHE geometry changed')
        records=[];row=dict(selection_id=sid,image=old['image'])
        for view,image in [('original',crop),('clahe',enhanced)]:
            raw=[]
            kept,n=infer(model,image,cfg['tile'],cfg['stride'],cfg['confidence'],cfg['imgsz'],cfg['global_nms_iou'],device,pil_input=True,audit=raw)
            if len(raw)!=n or sum(r['global_nms_kept'] for r in raw)!=len(kept):raise RuntimeError('Candidate audit mismatch')
            mapped=coverage_records(raw,offset,inner,view);records+=mapped
            row[view+'_tile_proposals']=n
            row[view+'_inside_after_global_nms']=sum(r['inside_inner_circle'] and r['global_nms_kept'] for r in mapped)
        row['status']='PROPOSALS_NOT_COLONY_COUNT'
        write_json(out/'detections'/f'{sid}.json',dict(old,selection_id=sid,coordinate_frame='EXIF_upright_original_pixels',crop_origin=offset,candidates=records,status=row['status'],independent_ground_truth=False))
        coverage_preview(crop,enhanced,offset,inner,records,out/'previews'/f'{sid}.jpg')
        rows.append(row);print(row,flush=True)
    write_csv(out/'candidate_summary.csv',rows)
    write_json(out/'complete.json',dict(images=len(rows),roi_missing=sum(r['status']=='ROI_MISSING' for r in rows),completed_predictions=sum(r['status']=='PROPOSALS_NOT_COLONY_COUNT' for r in rows),status='COVERAGE_COMPLETE',note='No final count or accuracy inferred from combined proposals. Changes after viewing Test50 make later results exploratory.'))
    print('SAVED:',out,flush=True)


def mark_multiscale_overlap(records, baseline):
    """Diagnostic only: max overlap relative to the smaller box, not matching/GT."""
    old=np.asarray([r['box'] for r in baseline if r['global_nms_kept'] and r['inside_inner_circle']],dtype=float).reshape(-1,4)
    for r in records:
        b=np.asarray(r['box'],dtype=float)
        if len(old):
            inter=np.maximum(0,np.minimum(old[:,2:],b[2:])-np.maximum(old[:,:2],b[:2])).prod(axis=1)
            area=np.prod(b[2:]-b[:2]); areas=(old[:,2:]-old[:,:2]).prod(axis=1)
            ios=float(np.max(inter/np.maximum(1e-9,np.minimum(area,areas))))
        else:ios=0.0
        r['max_ios_against_768_survivors']=ios
        r['low_overlap_768']=bool(r['inside_inner_circle'] and r['global_nms_kept'] and ios<.20)


def multiscale_preview(crop, offset, circle, old, new, sid, path):
    def panel(records, color):
        im=crop.copy(); w,h=im.size;im.thumbnail((1300,1300),Image.Resampling.LANCZOS)
        sx,sy=im.width/w,im.height/h;ox,oy=offset
        def rect(b):return ((b[0]-ox)*sx,(b[1]-oy)*sy,(b[2]-ox)*sx,(b[3]-oy)*sy)
        d=ImageDraw.Draw(im); cx,cy,r=circle
        d.ellipse(rect([cx-r,cy-r,cx+r,cy+r]),outline='#ffc04a',width=3)
        for q in records:
            if q['global_nms_kept'] and q['inside_inner_circle']:
                d.rectangle(rect(q['box']),outline=color,width=2)
        return im
    low=[q for q in new if q['low_overlap_768']]
    panels=[panel(old,'#25e59c'),panel(new,'#41cfff'),panel(low,'#ff7d45')]
    titles=['Tile 768 | original + CLAHE','Tile 512 | original + CLAHE','Tile 512 | low overlap with 768 (review only)']
    canvas=Image.new('RGB',(sum(im.width for im in panels)+20,max(im.height for im in panels)+56),'#20242a');d=ImageDraw.Draw(canvas)
    x=0
    for im,title in zip(panels,titles):
        d.text((x+5,6),sid+' | '+title,fill='white');canvas.paste(im,(x,28));x+=im.width+10
    d.text((5,canvas.height-20),'Orange: max intersection / smaller area < 0.20. Not confirmed new colonies. All raw proposals stored in JSON.',fill='#edcb8a')
    canvas.save(path,quality=94,format='JPEG')


def multiscale_main(argv):
    p=argparse.ArgumentParser(description='E2 validation multiscale recovery: reuse 768 candidates, add 512 tiles; no extra filtering')
    p.add_argument('--ids',nargs='+',help='Optional subset of existing V001-V050; default all 50. Completed cases are reused.')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args(argv)
    if a.self_test:
        self_test()
        old=[dict(box=[10,10,30,30],global_nms_kept=True,inside_inner_circle=True)]
        new=[dict(box=b,global_nms_kept=k,inside_inner_circle=i) for b,k,i in [([12,12,15,15],True,True),([40,40,50,50],True,True),([60,60,70,70],False,True),([80,80,90,90],True,False)]]
        mark_multiscale_overlap(new,old)
        assert [r['low_overlap_768'] for r in new]==[False,True,False,False]
        assert new[0]['max_ios_against_768_survivors']==1
        assert positions(1600,512,384)==[0,384,768,1088]
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as t:
            path=Path(t)/'preview.jpg';multiscale_preview(Image.new('RGB',(100,100),'#777777'),(0,0),(50,50,45),old,new,'VTEST',path)
            with Image.open(path) as im:assert im.width==320
        print('[PASS] multiscale grid, contained fragments, non-overlap, ROI flags, preview')
        return
    import cv2,torch,torchvision,ultralytics
    from ultralytics import YOLO
    from prepare_plate_images import crop_plate,clahe_luma
    base=ROOT/'outputs/diagnostics/preprocessing'; prior=base/'coverage_review'; out=base/'multiscale'
    cfg=json.loads((prior/'config.json').read_text(encoding='utf-8')); done=json.loads((prior/'complete.json').read_text(encoding='utf-8'))
    parent=base/'experiment_config.json'; original_cfg=json.loads(parent.read_text(encoding='utf-8'))
    ids=[f'V{i:03d}' for i in range(1,51)]
    if cfg['scope']!='validation_candidate_coverage_only' or cfg['split']!='val' or cfg['selection_ids']!=ids or cfg['tile']!=768 or cfg['stride']!=640 or done['completed_predictions']!=50 or done['roi_missing']!=0:raise SystemExit('Expected completed 50-image validation coverage (768)')
    if cfg['parent_config_sha256']!=sha(parent):raise SystemExit('Parent experiment config changed')
    selected=a.ids or ids
    if len(set(selected))!=len(selected) or not set(selected)<=set(ids):p.error('Use unique existing V001-V050 IDs')
    split=ROOT/'annotations/metadata/teacher_split_1200.csv'; model_path=Path(original_cfg['model'])
    if sha(split)!=cfg['selection_manifest_sha256'] or sha(model_path)!=cfg['model_sha256']:raise SystemExit('Source manifest or model changed')
    lookup={r['image']:r for r in read_csv(split)}
    settings=dict(scope='validation_multiscale_candidate_recovery_only',prior_config_sha256=sha(prior/'config.json'),parent_config_sha256=sha(parent),model_sha256=cfg['model_sha256'],selection_manifest_sha256=sha(split),selection_ids=ids,base_tile=768,base_stride=640,extra_tile=512,extra_stride=384,imgsz=cfg['imgsz'],confidence=cfg['confidence'],global_nms_iou=cfg['global_nms_iou'],clahe_clip=cfg['clahe_clip'],clahe_grid=cfg['clahe_grid'],gamma=cfg['gamma'],glare_clip=original_cfg['glare_clip'],crop_padding=cfg['crop_padding'],low_overlap_ios_threshold=.20,extra_filtering=False,cross_scale_dedup=False,independent_ground_truth=False,opencv_version=cv2.__version__,torch_version=torch.__version__,ultralytics_version=ultralytics.__version__)
    if out.exists():
        if not (out/'config.json').is_file() or json.loads((out/'config.json').read_text(encoding='utf-8'))!=settings:raise SystemExit('Existing multiscale output has different config; refusing overwrite')
    else:
        out.mkdir();write_json(out/'config.json',settings)
    (out/'previews').mkdir(exist_ok=True);(out/'detections').mkdir(exist_ok=True)
    device=0 if torch.cuda.is_available() else 'cpu'; model=None
    for sid in selected:
        prior_path=prior/'detections'/f'{sid}.json'; old=json.loads(prior_path.read_text(encoding='utf-8'))
        row=lookup[old['image']]; source=Path(row['source']); oldhash=sha(prior_path)
        if old['selection_id']!=sid or old['split']!='val' or row['split']!='val' or old['status']!='PROPOSALS_NOT_COLONY_COUNT' or sha(source)!=old['source_sha256']:raise ValueError('Invalid validation source '+sid)
        dest=out/'detections'/f'{sid}.json'; preview=out/'previews'/f'{sid}.jpg'
        if dest.exists():
            record=json.loads(dest.read_text(encoding='utf-8'))
            if record['prior_detection_sha256']!=oldhash or record['source_sha256']!=old['source_sha256'] or record['selection_id']!=sid:raise ValueError('Saved record identity mismatch '+sid)
            if preview.exists():print('[SKIP]',sid,flush=True);continue
        with Image.open(source) as f:im=ImageOps.exif_transpose(f).convert('RGB')
        if old['image_size']!={'width':im.width,'height':im.height}:raise ValueError('Image geometry changed '+sid)
        outer=tuple(old['circle_outer']); inner=tuple(old['circle_inner'])
        crop,offset=crop_plate(im,outer,padding=cfg['crop_padding'])
        if list(offset)!=old['crop_origin']:raise ValueError('Crop origin mismatch '+sid)
        baseline=[dict(r,id='768:'+r['id'],tile_size=768,stride=640) for r in old['candidates']]
        if dest.exists():
            new=[r for r in record['candidates'] if r['tile_size']==512]
        else:
            if model is None:model=YOLO(str(model_path))
            enhanced=clahe_luma(crop,cfg['clahe_clip'],cfg['clahe_grid'],cfg['gamma'],original_cfg['glare_clip'])
            new=[]
            for view,image in [('original',crop),('clahe',enhanced)]:
                raw=[]; kept,n=infer(model,image,512,384,cfg['confidence'],cfg['imgsz'],cfg['global_nms_iou'],device,pil_input=True,audit=raw)
                if len(raw)!=n or sum(q['global_nms_kept'] for q in raw)!=len(kept):raise RuntimeError('Audit mismatch '+sid)
                new.extend(dict(q,id='512:'+q['id'],tile_size=512,stride=384) for q in coverage_records(raw,offset,inner,view))
            mark_multiscale_overlap(new,baseline)
            def kept(r):return r['global_nms_kept'] and r['inside_inner_circle']
            summary=dict(selection_id=sid,image=old['image'],tile768_raw=len(baseline),tile512_raw=len(new),tile768_inside_proposals=sum(kept(r) for r in baseline),tile512_inside_proposals=sum(kept(r) for r in new),tile512_low_overlap_proposals=sum(r['low_overlap_768'] for r in new),status='PROPOSALS_NOT_COLONY_COUNT')
            record=dict(old,candidates=baseline+new,prior_detection_sha256=oldhash,summary=summary,scope=settings['scope'])
            tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(dest)
        tmp=preview.with_suffix('.jpg.tmp');multiscale_preview(crop,offset,inner,baseline,new,sid,tmp);tmp.replace(preview)
        print(record['summary'],flush=True)
    rows=[]
    for sid in ids:
        dest=out/'detections'/f'{sid}.json'
        if dest.exists() and (out/'previews'/f'{sid}.jpg').exists():
            record=json.loads(dest.read_text(encoding='utf-8'))
            if record['prior_detection_sha256']!=sha(prior/'detections'/f'{sid}.json'):raise ValueError('Prior result changed '+sid)
            rows.append(record['summary'])
    tmp=out/'candidate_summary.csv.tmp'
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(out/'candidate_summary.csv')
    tmp=out/'complete.json.tmp';tmp.write_text(json.dumps(dict(images=len(rows),expected=50,status='COMPLETE' if len(rows)==50 else 'PARTIAL',note='Low overlap is a review cue, not an additional true count. No cross-scale filtering applied.'),indent=2),encoding='utf-8');tmp.replace(out/'complete.json')
    print(f'SAVED {len(rows)}/50: {out}',flush=True)



# E2: image-assisted region discovery. No counting or extra deduplication.

def foreground_regions(crop, offset, circle, records):
    import cv2
    ox, oy = offset
    w, h = crop.size
    cx, cy, radius = map(float, circle)
    yy, xx = np.ogrid[:h, :w]
    roi = (xx + ox - cx)**2 + (yy + oy - cy)**2 <= radius**2
    if not roi.any():
        raise ValueError('Empty counting ROI')
    valid = [r for r in records if r['global_nms_kept'] and r['inside_inner_circle']]
    primary = [r for r in valid if r['tile_size'] == 768]
    secondary = [r for r in valid if r['tile_size'] == 512]
    # Robust size hint only: not a hard size filter. Ignore thin tile fragments for the hint.
    sizes = []
    for rec in primary:
        x0,y0,x1,y1 = rec['box']; a,b = x1-x0,y1-y0
        if min(a,b) >= 3 and max(a,b)/min(a,b) <= 1.8:
            sizes.append(min(a,b))
    d = float(np.clip(np.median(sizes) if sizes else radius*.045, 12, radius*.18))
    # Scale-aware band-pass on original luminance, NOT amplified CLAHE noise.
    rgb = np.asarray(crop.convert('RGB'))
    light = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:,:,0].astype(np.float32)
    smooth = cv2.GaussianBlur(light,(0,0),1.2)
    residual = np.zeros_like(smooth)
    sigmas = [max(1.5,d*f*.13) for f in (.5,1.,1.6)]
    for sigma in sigmas:
        a = cv2.GaussianBlur(smooth,(0,0),sigma)
        small = cv2.resize(a,(max(1,w//2),max(1,h//2)),interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(small,(0,0),sigma*1.25)
        bg = cv2.resize(bg,(w,h),interpolation=cv2.INTER_LINEAR)
        residual = np.maximum(residual,a-bg)
    hp = light-smooth; values = hp[roi]
    noise = float(1.4826*np.median(np.abs(values-np.median(values))))
    threshold = max(3.5,2.8*noise)
    low_threshold = threshold*.58
    low = np.uint8((residual>=low_threshold)&roi)*255
    high = (residual>=threshold)&roi
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    low = cv2.morphologyEx(low,cv2.MORPH_OPEN,k)
    low = cv2.morphologyEx(low,cv2.MORPH_CLOSE,k)
    low[~roi]=0
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(low,connectivity=8,ltype=cv2.CV_32S)
    hits = np.bincount(labels.ravel(),weights=high.ravel(),minlength=n)
    min_area = max(20,.045*d*d)
    min_width = max(4,.12*d)
    min_core = max(5,.008*d*d)
    keep = np.zeros(n,bool)
    rejected = dict(small_or_weak=0,thin_or_rim_like=0,large_sparse=0)
    for i in range(1,n):
        x,y,bw,bh,area = map(int,stats[i]); long=max(bw,bh); short=min(bw,bh)
        rr = math.hypot(centroids[i][0]+ox-cx,centroids[i][1]+oy-cy)
        fill = area/(bw*bh)
        if area<min_area or short<min_width or hits[i]<min_core:
            rejected['small_or_weak']+=1;continue
        if (long>4*short and fill<.55) or (long>5*d and rr>radius-.9*d and fill<.3):
            rejected['thin_or_rim_like']+=1;continue
        if long>radius*.8 and fill<.24:
            rejected['large_sparse']+=1;continue
        keep[i]=True
    mask = np.uint8(keep[labels])*255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask,connectivity=8,ltype=cv2.CV_32S)
    def box_mask(rs):
        m = np.zeros((h,w),np.uint8)
        for rec in rs:
            b=rec['box']; x0=max(0,math.floor(b[0]-ox)); y0=max(0,math.floor(b[1]-oy))
            x1=min(w,math.ceil(b[2]-ox)); y1=min(h,math.ceil(b[3]-oy))
            if x1>x0 and y1>y0:m[y0:y1,x0:x1]=1
        return m
    covered=box_mask(primary); both=covered|box_mask(secondary)
    flat=labels.ravel(); c768=np.bincount(flat,weights=covered.ravel(),minlength=n); c_all=np.bincount(flat,weights=both.ravel(),minlength=n)
    regions=[]; pad=max(12,round(d*.25)); roi_pixels=int(roi.sum())
    for i in range(1,n):
        x,y,bw,bh,area=map(int,stats[i]); frac=float(c768[i]/area); frac_all=float(c_all[i]/area)
        reasons=[]
        if frac<.15:reasons.append('low_yolo_coverage')
        elif frac<.70 and area-c768[i]>=min_area:reasons.append('partial_yolo_coverage')
        if max(bw,bh)>1.65*d or area>1.35*d*d:reasons.append('large_or_cluster_like')
        if area>.06*roi_pixels or max(bw,bh)>radius*.8:reasons.append('large_region_uncertain')
        # Coordinates are in EXIF-upright original pixels; xyxy half-open.
        regions.append(dict(region_id=i,box=[x+ox,y+oy,x+bw+ox,y+bh+oy],
            review_roi=[max(0,x-pad)+ox,max(0,y-pad)+oy,min(w,x+bw+pad)+ox,min(h,y+bh+pad)+oy],
            centroid=[float(centroids[i][0]+ox),float(centroids[i][1]+oy)],foreground_area_px=area,
            coverage_768=round(frac,5),coverage_768_and_512=round(frac_all,5),reasons=reasons,
            needs_review=bool(reasons)))
    info=dict(size_hint_px=d,size_hint_box_count=len(sizes),noise_mad=noise,residual_threshold=threshold,
        dog_sigmas_px=sigmas,low_threshold=low_threshold,min_component_area_px=min_area,
        rejected_small_or_weak=rejected['small_or_weak'],rejected_thin_or_rim_like=rejected['thin_or_rim_like'],
        rejected_large_sparse=rejected['large_sparse'],roi_pixels=roi_pixels,
        foreground_pixels=int(np.count_nonzero(mask)),primary_box_count=len(primary),supplementary_box_count=len(secondary),
        roi_region_count=len(regions),review_region_count=sum(r['needs_review'] for r in regions),
        low_coverage_regions=sum('low_yolo_coverage' in r['reasons'] for r in regions),
        partial_coverage_regions=sum('partial_yolo_coverage' in r['reasons'] for r in regions),
        cluster_like_regions=sum('large_or_cluster_like' in r['reasons'] for r in regions))
    return mask,residual,regions,info


def foreground_preview(crop,offset,circle,records,mask,regions,sid,path):
    w,h=crop.size; ox,oy=offset
    base=crop.copy();base.thumbnail((1150,1150),Image.Resampling.LANCZOS)
    sx,sy=base.width/w,base.height/h
    def rect(b):return ((b[0]-ox)*sx,(b[1]-oy)*sy,(b[2]-ox)*sx,(b[3]-oy)*sy)
    def circle_on(im):
        cx,cy,r=circle;ImageDraw.Draw(im).ellipse(rect([cx-r,cy-r,cx+r,cy+r]),outline='#ffc04a',width=2)
    a=base.copy();circle_on(a);draw=ImageDraw.Draw(a)
    for r in records:
        if r['tile_size']==768 and r['global_nms_kept'] and r['inside_inner_circle']:
            draw.rectangle(rect(r['box']),outline='#25e59c',width=2)
    m=Image.fromarray(mask).resize(base.size,Image.Resampling.NEAREST).convert('RGB');circle_on(m)
    b=base.copy();circle_on(b);draw=ImageDraw.Draw(b)
    for r in regions:
        if not r['needs_review']:continue
        reasons=r['reasons']
        color='#ff658e' if 'large_region_uncertain' in reasons else '#ff933b' if 'low_yolo_coverage' in reasons else '#ffee4d' if 'partial_yolo_coverage' in reasons else '#58caff'
        draw.rectangle(rect(r['review_roi']),outline=color,width=2)
    panels=[a,m,b]; titles=['YOLO 768 | Original + CLAHE','Foreground mask | NOT instances','Review ROIs | NOT detections']
    canvas=Image.new('RGB',(sum(p.width for p in panels)+20,base.height+66),'#20242a');dr=ImageDraw.Draw(canvas);x=0
    for im,title in zip(panels,titles):
        dr.text((x+4,6),sid+' | '+title,fill='white');canvas.paste(im,(x,28));x+=im.width+10
    dr.text((5,canvas.height-27),'Orange: low coverage | yellow: partial | cyan: cluster-like | pink: large/uncertain. No counts added.',fill='#edcb8a')
    canvas.save(path,format='JPEG',quality=94)


def foreground_self_test():
    import cv2
    from tempfile import TemporaryDirectory
    # Bright circles on sloping background: one covered, one missed, two touching.
    h=w=360; yy,xx=np.mgrid[:h,:w]; bg=np.uint8(42+xx*.10+yy*.04)
    im=np.repeat(bg[:,:,None],3,axis=2)
    for x,y,r in [(100,100,19),(240,100,17),(155,230,22),(186,230,22)]:
        cv2.circle(im,(x,y),r,(160,160,160),-1)
    cv2.circle(im,(20,20),14,(220,220,220),-1) # outside ROI
    crop=Image.fromarray(im); offset=(100,200); circle=(280,380,160)
    records=[dict(tile_size=768,global_nms_kept=True,inside_inner_circle=True,box=[179,279,221,321])]
    mask,res,regions,info=foreground_regions(crop,offset,circle,records)
    assert mask[100,100] and mask[100,240] and mask[230,170] and not mask[20,20]
    def near(x,y):return min(regions,key=lambda q:(q['centroid'][0]-x)**2+(q['centroid'][1]-y)**2)
    assert near(200,300)['coverage_768']>.9
    assert 'low_yolo_coverage' in near(340,300)['reasons']
    assert 'large_or_cluster_like' in near(270,430)['reasons']
    assert all(r['review_roi'][0]>=100 and r['review_roi'][1]>=200 for r in regions)
    blank=Image.fromarray(np.repeat(bg[:,:,None],3,axis=2));bm,_,br,_=foreground_regions(blank,offset,circle,[])
    assert not bm[180,180] and not br, 'Smooth gradient produced regions'
    rng=np.random.default_rng(42)
    noisy=np.uint8(np.clip(bg.astype(float)+rng.normal(0,2.5,bg.shape),0,255))
    noise_rgb=np.repeat(noisy[:,:,None],3,axis=2)
    cv2.circle(noise_rgb,(145,145),18,(150,150,150),-1)
    nm,_,nr,_=foreground_regions(Image.fromarray(noise_rgb),offset,circle,records)
    assert nm[145,145] and len(nr)<15, 'Noise suppression or bright core failed'
    with TemporaryDirectory() as t:
        p=Path(t)/'preview.jpg';foreground_preview(crop,offset,circle,records,mask,regions,'VTEST',p)
        assert p.is_file()
    print('[PASS] foreground, smooth background, missed ROI, touching ROI, coordinate mapping, ROI boundary, preview')


def foreground_main(argv):
    p=argparse.ArgumentParser(description='E2 image-assisted foreground ROI discovery; validation only; no detection count or GPU inference')
    p.add_argument('--ids',nargs='+',help='Optional subset V001-V050; defaults to all 50; resume supported')
    p.add_argument('--self-test',action='store_true');a=p.parse_args(argv)
    if a.self_test:foreground_self_test();return
    import cv2
    from prepare_plate_images import crop_plate
    base=ROOT/'outputs/diagnostics/preprocessing';prior=base/'multiscale';out=base/'foreground_roi'
    cfg=json.loads((prior/'config.json').read_text(encoding='utf-8'));done=json.loads((prior/'complete.json').read_text(encoding='utf-8'))
    ids=[f'V{i:03d}' for i in range(1,51)]
    if cfg['scope']!='validation_multiscale_candidate_recovery_only' or cfg['selection_ids']!=ids or done['status']!='COMPLETE' or done['images']!=50:raise SystemExit('Expected complete validation multiscale run')
    chosen=a.ids or ids
    if len(set(chosen))!=len(chosen) or not set(chosen)<=set(ids):p.error('Use unique V001-V050 IDs')
    split=ROOT/'annotations/metadata/teacher_split_1200.csv'
    if sha(split)!=cfg['selection_manifest_sha256']:raise SystemExit('Split manifest changed')
    lookup={r['image']:r for r in read_csv(split)}
    settings=dict(scope='validation_image_assisted_region_discovery',selection_ids=ids,prior_config_sha256=sha(prior/'config.json'),
        split_manifest_sha256=sha(split),model_sha256=cfg['model_sha256'],primary_tile=768,supplementary_tile=512,
        method='lab_l_multiscale_dog_hysteresis_component_filter',coverage_thresholds=[.15,.70],
        component_area_cutoff='max(20,0.045*size_hint^2)',minimum_width='max(4,0.12*size_hint)',strong_core_pixels='max(5,0.008*size_hint^2)',dog_scales=[.5,1.,1.6],hysteresis_high='max(3.5,2.8*noise_mad)',hysteresis_low_factor=.58,opencv_version=cv2.__version__,
        script_sha256=sha(Path(__file__)),new_colony_candidates=False,extra_dedup=False,independent_ground_truth=False)
    if out.exists():
        if not (out/'config.json').is_file() or json.loads((out/'config.json').read_text(encoding='utf-8'))!=settings:raise SystemExit('Foreground output config mismatch; preserve previous diagnostics before rerunning')
    else:
        out.mkdir();write_json(out/'config.json',settings)
    for sub in ('regions','masks','previews'):(out/sub).mkdir(exist_ok=True)
    for sid in chosen:
        inp=prior/'detections'/f'{sid}.json';old=json.loads(inp.read_text(encoding='utf-8'));ih=sha(inp)
        row=lookup[old['image']];source=Path(row['source'])
        if old['selection_id']!=sid or old['split']!='val' or row['split']!='val' or old['status']!='PROPOSALS_NOT_COLONY_COUNT' or sha(source)!=old['source_sha256']:raise ValueError('Invalid validation input '+sid)
        dest=out/'regions'/f'{sid}.json';maskpath=out/'masks'/f'{sid}.png';prev=out/'previews'/f'{sid}.jpg'
        if dest.exists():
            saved=json.loads(dest.read_text(encoding='utf-8'))
            if saved['prior_detection_sha256']!=ih or saved['source_sha256']!=old['source_sha256']:raise ValueError('Prior record changed '+sid)
            if maskpath.exists() and prev.exists() and saved['mask_sha256']==sha(maskpath) and saved['preview_sha256']==sha(prev):print('[SKIP]',sid,flush=True);continue
        with Image.open(source) as f:im=ImageOps.exif_transpose(f).convert('RGB')
        if old['image_size']!={'width':im.width,'height':im.height}:raise ValueError('Original dimension mismatch '+sid)
        crop,offset=crop_plate(im,tuple(old['circle_outer']),padding=cfg['crop_padding'])
        if list(offset)!=old['crop_origin']:raise ValueError('Crop origin mismatch '+sid)
        circle=tuple(old['circle_inner']);mask,residual,regions,info=foreground_regions(crop,offset,circle,old['candidates'])
        tmp=maskpath.with_suffix('.png.tmp');Image.fromarray(mask).save(tmp,format='PNG');tmp.replace(maskpath)
        tmp=prev.with_suffix('.jpg.tmp');foreground_preview(crop,offset,circle,old['candidates'],mask,regions,sid,tmp);tmp.replace(prev)
        summary=dict(selection_id=sid,image=old['image'],**info,status='REVIEW_REGIONS_NOT_COLONY_COUNT')
        result=dict(selection_id=sid,image=old['image'],source_sha256=old['source_sha256'],prior_detection_sha256=ih,
             mask_sha256=sha(maskpath),preview_sha256=sha(prev),image_size=old['image_size'],crop_origin=list(offset),crop_size=list(crop.size),circle_inner=list(circle),
             coordinate_frame='EXIF_upright_original_pixels_xyxy_half_open',mask_coordinate_frame='crop_local',
             region_id_definition='8-connected components in saved binary mask; reconstruct with OpenCV same version',regions=regions,summary=summary,
             status=summary['status'],independent_ground_truth=False)
        tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(dest)
        print(sid,'regions',info['roi_region_count'],'review',info['review_region_count'],'low coverage',info['low_coverage_regions'],'cluster-like',info['cluster_like_regions'],flush=True)
    rows=[]
    for sid in ids:
        dp=out/'regions'/f'{sid}.json';mp=out/'masks'/f'{sid}.png';pp=out/'previews'/f'{sid}.jpg'
        if not (dp.exists() and mp.exists() and pp.exists()):continue
        d=json.loads(dp.read_text(encoding='utf-8'))
        if d['prior_detection_sha256']!=sha(prior/'detections'/f'{sid}.json') or d['mask_sha256']!=sha(mp) or d['preview_sha256']!=sha(pp):raise ValueError('Saved result changed '+sid)
        rows.append(d['summary'])
    tmp=out/'region_summary.csv.tmp'
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(out/'region_summary.csv')
    tmp=out/'complete.json.tmp';tmp.write_text(json.dumps(dict(images=len(rows),expected=50,status='COMPLETE' if len(rows)==50 else 'PARTIAL',note='Filtered foreground components and review ROIs only. No YOLO candidates removed, no final count or confirmed new instances.'),indent=2),encoding='utf-8');tmp.replace(out/'complete.json')
    print(f'SAVED {len(rows)}/50: {out}',flush=True)


# E2: YOLO-only detection recovery. Image processing changes input pixels,
# never creates a detection or removes a prior YOLO candidate.
def enhance_tile(patch):
    import cv2
    rgb=np.asarray(patch.convert('RGB'))
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    l=lab[:,:,0]
    smooth=cv2.bilateralFilter(l,5,15,3)
    # Broad background estimate on a reduced image avoids expensive full-scale blur.
    h,w=l.shape
    small=cv2.resize(smooth,(max(1,w//8),max(1,h//8)),interpolation=cv2.INTER_AREA).astype(np.float32)
    bg=cv2.resize(cv2.GaussianBlur(small,(0,0),max(1,min(small.shape)/12)),(w,h),interpolation=cv2.INTER_LINEAR)
    shift=np.clip(.65*(float(np.median(bg))-bg),-18,18)
    normalized=np.clip(smooth.astype(np.float32)+shift,0,255).astype(np.uint8)
    enhanced=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(normalized)
    lab[:,:,0]=np.uint8(np.round(.55*normalized+.45*enhanced))
    return Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))


def tile_enhancement_flags(records, baseline, enhanced_normal):
    # IoS and center proximity are review cues only. Preserve ALL records.
    def rects(rs):
        return np.asarray([r['box'] for r in rs if r['inside_inner_circle'] and r['global_nms_kept']],dtype=float).reshape(-1,4)
    def compare(b,old):
        if not len(old):return 0.,False
        i=np.maximum(0,np.minimum(old[:,2:],b[2:])-np.maximum(old[:,:2],b[:2])).prod(axis=1)
        a=(old[:,2:]-old[:,:2]).prod(axis=1);s=np.prod(b[2:]-b[:2])
        ios=float(np.max(i/np.maximum(1e-9,np.minimum(a,s))))
        delta=np.linalg.norm((old[:,:2]+old[:,2:])/2-(b[:2]+b[2:])/2,axis=1)
        scale=np.maximum(np.linalg.norm(old[:,2:]-old[:,:2],axis=1),np.linalg.norm(b[2:]-b[:2]))
        return ios,bool(np.any(delta<.35*scale))
    ab=rects(baseline);c=rects(enhanced_normal)
    for r in records:
        b=np.asarray(r['box'],dtype=float);i,near=compare(b,ab);j,nc=compare(b,c)
        active=r['inside_inner_circle'] and r['global_nms_kept']
        r.update(max_ios_ab=i,near_ab_center=near,max_ios_c=j,near_c_center=nc,
                 review_new_location=bool(active and i<.20 and not near and (r['view']=='local_enhance' or (j<.20 and not nc))))


def tile_enhancement_preview(crop,offset,circle,base,c,d,sid,path):
    w,h=crop.size;ox,oy=offset;im=crop.copy();im.thumbnail((1040,1040),Image.Resampling.LANCZOS)
    sx,sy=im.width/w,im.height/h
    def rect(b):return ((b[0]-ox)*sx,(b[1]-oy)*sy,(b[2]-ox)*sx,(b[3]-oy)*sy)
    def panel(groups):
        canvas=im.copy();draw=ImageDraw.Draw(canvas);cx,cy,r=circle
        draw.ellipse(rect([cx-r,cy-r,cx+r,cy+r]),outline='#ffbd48',width=2)
        for rs,color in groups:
            for q in rs:
                if q['inside_inner_circle'] and q['global_nms_kept']:draw.rectangle(rect(q['box']),outline=color,width=2)
        return canvas
    panels=[panel([(base,'#25e59c')]),panel([(c,'#41cfff')]),panel([([q for q in c if q['review_new_location']],'#ff993d')]),panel([([q for q in d if q['review_new_location']],'#e66af1')])]
    titles=['A+B | original + whole crop CLAHE','C | local tile enhancement conf .25','C | new-location review vs A+B','D | conf .15 new-location review vs A+B+C']
    canvas=Image.new('RGB',(im.width*2+10,(im.height+27)*2+30),'#20242a');draw=ImageDraw.Draw(canvas)
    for idx,(p,title) in enumerate(zip(panels,titles)):
        x=(idx%2)*(im.width+10);y=(idx//2)*(im.height+27);draw.text((x+4,y+6),sid+' | '+title,fill='white');canvas.paste(p,(x,y+26))
    draw.text((5,canvas.height-20),'Review cues only; not verified new colonies. All YOLO proposals retained. No foreground-derived boxes.',fill='#edcb8a')
    canvas.save(path,format='JPEG',quality=94)


def tile_input_preview(crop,offset,c,d,sid,path):
    # Show actual input patches, prioritizing review locations; deterministic fallback.
    ox,oy=offset; chosen=[]
    for q in c+d:
        t=tuple(q['tile'])
        if q['review_new_location'] and t not in chosen:chosen.append(t)
        if len(chosen)>=3:break
    if not chosen:
        w,h=crop.size;x=max(0,(w-768)//2);y=max(0,(h-768)//2)
        chosen=[(x+ox,y+oy,min(w,x+768)+ox,min(h,y+768)+oy)]
    side=460;canvas=Image.new('RGB',(2*side+10,len(chosen)*(side+28)),'#20242a');draw=ImageDraw.Draw(canvas)
    for i,t in enumerate(chosen):
        x0,y0,x1,y1=t;p=crop.crop((x0-ox,y0-oy,x1-ox,y1-oy));q=enhance_tile(p)
        for j,(a,label) in enumerate([(p,'original tile'),(q,'enhanced tile')]):
            a.thumbnail((side,side),Image.Resampling.LANCZOS);x=j*(side+10);y=i*(side+28);draw.text((x+4,y+5),f'{sid} {label} {t}',fill='white');canvas.paste(a,(x,y+27))
    canvas.save(path,format='JPEG',quality=93)


def tile_enhancement_self_test():
    import cv2
    self_test()
    from tempfile import TemporaryDirectory
    rng=np.random.default_rng(9);yy,xx=np.mgrid[:256,:256]
    l=np.uint8(np.clip(70+xx*.15+rng.normal(0,2,(256,256)),0,255));rgb=np.repeat(l[:,:,None],3,axis=2)
    cv2.circle(rgb,(120,110),14,(145,145,145),-1)
    im=Image.fromarray(rgb); before=np.asarray(im).copy();enh=enhance_tile(im)
    assert enh.size==im.size and enh.mode=='RGB' and np.array_equal(np.asarray(im),before)
    assert np.asarray(enh).dtype==np.uint8 and np.array_equal(np.asarray(enh),np.asarray(enhance_tile(im)))
    def rec(b,view):return dict(box=b,tile=[0,0,256,256],global_nms_kept=True,inside_inner_circle=True,view=view)
    base=[rec([100,100,140,140],'original')];c=[rec([102,102,110,110],'local_enhance'),rec([180,180,200,200],'local_enhance')];d=[rec([181,181,201,201],'local_enhance_low'),rec([20,20,40,40],'local_enhance_low')]
    tile_enhancement_flags(c,base,[]);tile_enhancement_flags(d,base,c)
    assert [r['review_new_location'] for r in c]==[False,True] and [r['review_new_location'] for r in d]==[False,True]
    with TemporaryDirectory() as t:
        p=Path(t)/'preview.jpg';tile_enhancement_preview(im,(0,0),(128,128,120),base,c,d,'VTEST',p);assert p.is_file()
        p=Path(t)/'tiles.jpg';tile_input_preview(im,(0,0),c,d,'VTEST',p);assert p.is_file()
    print('[PASS] local enhancement, geometry, deterministic pixels, review cue, preview, input tiles')


def tile_enhancement_main(argv):
    p=argparse.ArgumentParser(description='E2 YOLO-only tile enhancement pilot, reuse A/B; C and D are separate inference passes')
    p.add_argument('--ids',nargs='+',help='Subset of validation V001-V050; default all; resume supported')
    p.add_argument('--self-test',action='store_true');a=p.parse_args(argv)
    if a.self_test:tile_enhancement_self_test();return
    import cv2,torch,ultralytics
    from ultralytics import YOLO
    from prepare_plate_images import crop_plate
    base=ROOT/'outputs/diagnostics/preprocessing';prior=base/'coverage_review';out=base/'local_tile_enhancement'
    cfg=json.loads((prior/'config.json').read_text(encoding='utf-8'));done=json.loads((prior/'complete.json').read_text(encoding='utf-8'))
    parent=base/'experiment_config.json';parent_cfg=json.loads(parent.read_text(encoding='utf-8'));ids=[f'V{i:03d}' for i in range(1,51)]
    if cfg['scope']!='validation_candidate_coverage_only' or cfg['split']!='val' or cfg['selection_ids']!=ids or cfg['tile']!=768 or cfg['stride']!=640 or cfg['confidence']!=.25 or done['completed_predictions']!=50 or done['roi_missing']!=0:raise SystemExit('Expected complete matching validation coverage')
    split=ROOT/'annotations/metadata/teacher_split_1200.csv';model_path=Path(parent_cfg['model'])
    if sha(split)!=cfg['selection_manifest_sha256'] or sha(model_path)!=cfg['model_sha256'] or sha(parent)!=cfg['parent_config_sha256']:raise SystemExit('Input model, manifest or config changed')
    chosen=a.ids or ids
    if len(set(chosen))!=len(chosen) or not set(chosen)<=set(ids):p.error('Use unique V001-V050 IDs')
    lookup={r['image']:r for r in read_csv(split)}
    settings=dict(scope='validation_local_tile_enhancement_only',selection_ids=ids,prior_config_sha256=sha(prior/'config.json'),model_sha256=cfg['model_sha256'],split_manifest_sha256=sha(split),tile=768,stride=640,imgsz=cfg['imgsz'],iou=cfg['global_nms_iou'],confidence_c=.25,confidence_d=.15,enhancement='LAB L bilateral 5/15/3; bounded background correction .65 +/-18; CLAHE 1.5 8x8 blend .45',low_overlap_ios=.20,near_center_diagonal=.35,foreground_used=False,independent_ground_truth=False,opencv=cv2.__version__,torch=torch.__version__,ultralytics=ultralytics.__version__,script_sha256=sha(Path(__file__)))
    if out.exists():
        if not (out/'config.json').is_file() or json.loads((out/'config.json').read_text(encoding='utf-8'))!=settings:raise SystemExit('Output config mismatch; preserve existing results before changing method')
    else:out.mkdir();write_json(out/'config.json',settings)
    for sub in ('detections','previews','tiles'):(out/sub).mkdir(exist_ok=True)
    device=0 if torch.cuda.is_available() else 'cpu';model=None
    for sid in chosen:
        inp=prior/'detections'/f'{sid}.json';old=json.loads(inp.read_text(encoding='utf-8'));oldhash=sha(inp);row=lookup[old['image']];source=Path(row['source'])
        if old['selection_id']!=sid or old['split']!='val' or row['split']!='val' or old['status']!='PROPOSALS_NOT_COLONY_COUNT' or sha(source)!=old['source_sha256']:raise ValueError('Invalid validation source '+sid)
        dp=out/'detections'/f'{sid}.json';pp=out/'previews'/f'{sid}.jpg';tp=out/'tiles'/f'{sid}.jpg'
        if dp.exists():
            saved=json.loads(dp.read_text(encoding='utf-8'))
            if saved['prior_detection_sha256']!=oldhash or saved['source_sha256']!=old['source_sha256']:raise ValueError('Previous result changed '+sid)
            if pp.exists() and tp.exists() and saved['preview_sha256']==sha(pp) and saved['tiles_sha256']==sha(tp):print('[SKIP]',sid,flush=True);continue
        with Image.open(source) as f:im=ImageOps.exif_transpose(f).convert('RGB')
        if old['image_size']!={'width':im.width,'height':im.height}:raise ValueError('Original dimensions changed '+sid)
        crop,offset=crop_plate(im,tuple(old['circle_outer']),padding=cfg['crop_padding']);circle=tuple(old['circle_inner'])
        if list(offset)!=old['crop_origin']:raise ValueError('Crop origin mismatch '+sid)
        ab=[dict(q,tile_size=768,stride=640) for q in old['candidates']]
        if model is None:model=YOLO(str(model_path))
        extra=[]
        for view,conf in [('local_enhance',.25),('local_enhance_low',.15)]:
            raw=[];kept,n=infer(model,crop,768,640,conf,cfg['imgsz'],cfg['global_nms_iou'],device,pil_input=True,audit=raw,patch_transform=enhance_tile)
            if len(raw)!=n or sum(r['global_nms_kept'] for r in raw)!=len(kept):raise RuntimeError('Audit mismatch '+sid)
            extra.append([dict(q,tile_size=768,stride=640,confidence_threshold=conf) for q in coverage_records(raw,offset,circle,view)])
        c,d=extra;tile_enhancement_flags(c,ab,[]);tile_enhancement_flags(d,ab,c)
        def active(q):return q['inside_inner_circle'] and q['global_nms_kept']
        summary=dict(selection_id=sid,image=old['image'],a_inside=sum(active(q) and q['view']=='original' for q in ab),b_inside=sum(active(q) and q['view']=='clahe' for q in ab),c_inside=sum(active(q) for q in c),d_inside=sum(active(q) for q in d),c_review_locations=sum(q['review_new_location'] for q in c),d_review_locations=sum(q['review_new_location'] for q in d),status='PROPOSALS_NOT_COLONY_COUNT')
        tmp=pp.with_suffix('.jpg.tmp');tile_enhancement_preview(crop,offset,circle,ab,c,d,sid,tmp);tmp.replace(pp)
        tmp=tp.with_suffix('.jpg.tmp');tile_input_preview(crop,offset,c,d,sid,tmp);tmp.replace(tp)
        result=dict(old,candidates=ab+c+d,prior_detection_sha256=oldhash,preview_sha256=sha(pp),tiles_sha256=sha(tp),summary=summary,scope=settings['scope'])
        tmp=dp.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(dp)
        print(summary,flush=True)
    rows=[]
    for sid in ids:
        dp=out/'detections'/f'{sid}.json';pp=out/'previews'/f'{sid}.jpg';tp=out/'tiles'/f'{sid}.jpg'
        if not (dp.exists() and pp.exists() and tp.exists()):continue
        saved=json.loads(dp.read_text(encoding='utf-8'))
        if saved['prior_detection_sha256']!=sha(prior/'detections'/f'{sid}.json') or saved['preview_sha256']!=sha(pp) or saved['tiles_sha256']!=sha(tp):raise ValueError('Output changed '+sid)
        rows.append(saved['summary'])
    tmp=out/'candidate_summary.csv.tmp'
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    tmp.replace(out/'candidate_summary.csv')
    tmp=out/'complete.json.tmp';tmp.write_text(json.dumps(dict(images=len(rows),expected=50,status='COMPLETE' if len(rows)==50 else 'PARTIAL',note='YOLO proposals only; no foreground-generated detections, no verified count or recall.'),indent=2),encoding='utf-8');tmp.replace(out/'complete.json')
    print(f'SAVED {len(rows)}/50: {out}',flush=True)


# E3: conservative duplicate consolidation. No new detection or image-wide mask.
E3_VIEWS = ('original', 'clahe', 'local_enhance')
E3_RULES = dict(full_iou=.72, full_min_area_ratio=.70, full_center_min_diag=.16,
                tile_edge_tolerance_px=3., fragment_max_area_ratio=.80,
                fragment_min_containment=.90, clipped_full_min_iou=.75,
                parent_child_containment=.82, parent_child_max_ratio=.83,
                parent_min_child_ratio=.20, parent_pair_max_iou=.34,
                parent_pair_min_separation=.55, parent_min_union_coverage=.58,
                parent_min_envelope_coverage=.72, parent_min_child_area_similarity=.42,
                parent_min_peak_contrast=3., parent_min_saddle_drop=1.5,
                parent_max_children=6, residual_fragment_rule='normalized_clipped_footprint_unique_target',
                residual_min_containment=.78, residual_max_normalized_edge_error=.16, residual_orthogonal_edge_error=.22,
                parent_distance_neck_thresholds=[.32,.46,.60],parent_distance_neck_min_drop=.055,
                parent_distance_neck_min_separation=.38,parent_distance_neck_min_votes=2)


def refinement_box_metrics(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    wh=a[2:]-a[:2]; vh=b[2:]-b[:2]
    aa=float(np.prod(wh));bb=float(np.prod(vh))
    inter=float(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod())
    if min(aa,bb)<=0:return 0.,0.,0.,float('inf')
    return (inter/(aa+bb-inter),inter/min(aa,bb),min(aa,bb)/max(aa,bb),
            float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/min(np.linalg.norm(wh),np.linalg.norm(vh))))


def refinement_edges(rec, bounds):
    b=rec['box'];t=rec['tile'];l,u,r,v=bounds;tol=E3_RULES['tile_edge_tolerance_px']
    return [i for i,internal,near in [(0,t[0]>l,b[0]-t[0]<=tol),(1,t[1]>u,b[1]-t[1]<=tol),
            (2,t[2]<r,t[2]-b[2]<=tol),(3,t[3]<v,t[3]-b[3]<=tol)] if internal and near]


def refinement_fragment(a,b,edges_a,edges_b):
    # A small box touching an internal tile edge must agree with the *clipped*
    # projection of the intact detection in that same tile. Containment alone
    # is insufficient (could be an adjacent small colony).
    if not edges_a or edges_b or a['tile']==b['tile']:return False
    x=np.asarray(a['box'],float); y=np.asarray(b['box'],float); t=np.asarray(a['tile'],float)
    aa=float(np.prod(x[2:]-x[:2]));bb=float(np.prod(y[2:]-y[:2]))
    if not 0<aa<=E3_RULES['fragment_max_area_ratio']*bb:return False
    _,ios,_,_=refinement_box_metrics(x,y)
    if ios<E3_RULES['fragment_min_containment']:return False
    if not any(y[i]<t[i]-3 if i<2 else y[i]>t[i]+3 for i in edges_a):return False
    clipped=np.r_[np.maximum(y[:2],t[:2]),np.minimum(y[2:],t[2:])]
    if np.any(clipped[2:]<=clipped[:2]):return False
    return refinement_box_metrics(x,clipped)[0]>=E3_RULES['clipped_full_min_iou']


def refine_candidates(records,bounds):
    eligible=[q for q in records if q['view'] in E3_VIEWS and q['inside_inner_circle'] and q['global_nms_kept']]
    ids=[q['id'] for q in eligible]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate candidate IDs')
    for q in eligible:
        b=np.asarray(q['box'],float);t=np.asarray(q['tile'],float)
        if b.shape!=(4,) or t.shape!=(4,) or not np.isfinite(b).all() or not np.isfinite(t).all() or not np.isfinite(q['confidence']) or np.any(b[2:]<=b[:2]) or np.any(b[:2]<t[:2]-1e-3) or np.any(b[2:]>t[2:]+1e-3):raise ValueError('Invalid proposal '+q['id'])
    edges={q['id']:refinement_edges(q,bounds) for q in eligible}
    # Prefer intact boxes, then confidence; no box averaging or coordinate changes.
    rs=sorted(eligible,key=lambda q:(bool(edges[q['id']]),-q['confidence'],E3_VIEWS.index(q['view']),q['id']))
    n=len(rs);boxes=np.asarray([q['box'] for q in rs],float).reshape(-1,4);wh=boxes[:,2:]-boxes[:,:2];areas=wh.prod(1);diag=np.linalg.norm(wh,axis=1);cent=(boxes[:,:2]+boxes[:,2:])/2
    # Build tight duplicate neighborhoods. Complete-link clustering avoids A-B-C chains.
    close=[]
    for i in range(n):
        inter=np.maximum(0,np.minimum(boxes[:,2:],boxes[i,2:])-np.maximum(boxes[:,:2],boxes[i,:2])).prod(1)
        iou=inter/(areas+areas[i]-inter); ratio=np.minimum(areas,areas[i])/np.maximum(areas,areas[i]); delta=np.linalg.norm(cent-cent[i],axis=1)/np.minimum(diag,diag[i])
        close.append(set(np.flatnonzero((iou>=E3_RULES['full_iou'])&(ratio>=E3_RULES['full_min_area_ratio'])&(delta<=E3_RULES['full_center_min_diag'])).tolist()))
    groups=[];owner={};ambiguous=set()
    for i in range(n):
        neighbors={owner[j] for j in close[i] if j in owner}
        matches=[g for g in sorted(neighbors) if all(j in close[i] for j in groups[g])]
        if len(matches)==1 and len(neighbors)==1:
            g=matches[0];groups[g].append(i);owner[i]=g
        else:
            if neighbors:ambiguous.add(i);ambiguous.update(groups[g][0] for g in neighbors)
            owner[i]=len(groups);groups.append([i])
    # Resolve explicit tile clipping only against one intact group. Preserve
    # ambiguous or conflicting groups. Never chain fragment -> fragment.
    full=[g for g,mem in enumerate(groups) if not edges[rs[mem[0]]['id']]]
    parents={};fragment_ambiguous=set()
    for g,mem in enumerate(groups):
        if g in full:continue
        a=rs[mem[0]];ab=boxes[mem[0]];aa=areas[mem[0]]
        # Cheap containment prefilter reduces pair checks for crowded plates.
        inter=np.maximum(0,np.minimum(boxes[:,2:],ab[2:])-np.maximum(boxes[:,:2],ab[:2])).prod(1)
        possible=set(np.flatnonzero((areas>=aa/E3_RULES['fragment_max_area_ratio'])&(inter>=aa*E3_RULES['fragment_min_containment'])).tolist())
        matches=[]
        for f in full:
            if groups[f][0] not in possible:continue
            b=rs[groups[f][0]]
            if all(refinement_fragment(rs[j],b,edges[rs[j]['id']],edges[b['id']]) for j in mem):matches.append(f)
        if len(matches)==1 and not any(j in ambiguous for j in mem) and groups[matches[0]][0] not in ambiguous:parents[g]=matches[0]
        elif matches:fragment_ambiguous.add(g)
    result=[];suppressed=[];kept=[]
    for g,mem in enumerate(groups):
        root=parents.get(g,g); rep=rs[groups[root][0]];keeper_id=rep['id']
        if g==root:
            kept.append(dict(rep, supporting_ids=[rs[j]['id'] for j in mem],tile_edges=edges[rep['id']],
                review_needed=bool(any(j in ambiguous for j in mem) or g in fragment_ambiguous or edges[rep['id']]),
                review_reasons=(['ambiguous_overlap'] if any(j in ambiguous for j in mem) else [])+(['ambiguous_tile_fragment'] if g in fragment_ambiguous else [])+(['unmatched_tile_edge'] if edges[rep['id']] else [])))
        for j in mem:
            q=rs[j];is_keeper=q['id']==keeper_id
            reason='retained' if is_keeper else 'tile_edge_fragment' if g!=root else 'near_identical_box'
            row=dict(q,refinement_kept=is_keeper,keeper_id=keeper_id,refinement_reason=reason,tile_edges=edges[q['id']])
            result.append(row)
            if not is_keeper:suppressed.append(row)
    # Populate all support IDs including the absorbed fragments.
    supports=defaultdict(list)
    for q in result:supports[q['keeper_id']].append(q['id'])
    for q in kept:q['supporting_ids']=supports[q['id']]
    if len(result)!=n or len(kept)+len(suppressed)!=n or len({q['id'] for q in kept})!=len(kept) or any(q['keeper_id'] not in supports for q in result):raise RuntimeError('Refinement accounting failed')
    return kept,suppressed,result



def _rectangle_union_area(rects):
    """Exact union of a small list of axis-aligned rectangles."""
    xs=sorted({float(x) for b in rects for x in (b[0],b[2])});area=0.
    for l,r in zip(xs,xs[1:]):
        ys=sorted((float(b[1]),float(b[3])) for b in rects if b[0]<r and b[2]>l)
        if not ys:continue
        lo,hi=ys[0];length=0.
        for a,b in ys[1:]:
            if a>hi:length+=hi-lo;lo,hi=a,b
            else:hi=max(hi,b)
        area+=(r-l)*(length+hi-lo)
    return area


def local_lobes(im,off,parent,children):
    """Distance-transform neck evidence inside a candidate parent ROI only."""
    import cv2
    b=np.array(parent['box'],float);c=np.array([a['box'] for a in children]);size=float(np.median(np.min(c[:,2:]-c[:,:2],axis=1)));o=np.array(off);l,t=np.maximum(0,np.floor(b[:2]-o-.4*size)).astype(int);r,v=np.minimum([im.width,im.height],np.ceil(b[2:]-o+.4*size)).astype(int)
    if r-l<5 or v-t<5:return False,dict(reason='small_patch')
    patch=np.array(im.crop((l,t,r,v)).convert('L'),np.float32);sm=cv2.GaussianBlur(patch,(0,0),max(.65,.035*size));xy=np.array([l,t])+o;bg=float(np.percentile(sm,20));high=float(np.percentile(sm,94))
    if high-bg<8:return False,dict(reason='low_contrast')
    yy,xx=np.ogrid[:sm.shape[0],:sm.shape[1]];results=[]
    for f in [.32,.46,.60]:
        mask=np.uint8(sm>bg+f*(high-bg));dist=cv2.distanceTransform(mask,cv2.DIST_L2,5);peaks=[]
        for a in c:
            center=(a[:2]+a[2:])/2-xy;rad=np.maximum(2,(a[2:]-a[:2])*.27);core=((xx-center[0])/rad[0])**2+((yy-center[1])/rad[1])**2<=1;masked=np.where(core,dist,-1);y,x=np.unravel_index(np.argmax(masked),dist.shape);peaks.append((int(x),int(y),float(dist[y,x])))
        good=True;drops=[]
        for i,(x,y,z) in enumerate(peaks):
            if z<.15*size:good=False
            for X,Y,Z in peaks[i+1:]:
                sep=np.hypot(X-x,Y-y);u=np.linspace(0,1,max(9,int(sep)));line=dist[np.rint(y+u*(Y-y)).astype(int),np.rint(x+u*(X-x)).astype(int)];drop=1-float(line.min())/max(.01,min(z,Z));drops.append(round(drop,3))
                if sep<.38*size or drop<.055:good=False
        results.append(dict(threshold=f,passed=bool(good),peaks=peaks,drops=drops))
    ok=sum(a['passed'] for a in results)>=2
    return ok,dict(reason='distance_neck_supported' if ok else 'no_stable_neck',thresholds=results)


def residual_fragment(a,b):
    """Clipping-consistent thin fragments (not a general overlap threshold)."""
    x=np.array(a['box']);y=np.array(b['box']);t=np.array(a['tile']);w=y[2:]-y[:2];aw=x[2:]-x[:2];aa=np.prod(aw);bb=np.prod(w)
    if not a['tile_edges'] or a['tile']==b['tile'] or aa>.80*bb:return False
    clip=np.r_[np.maximum(y[:2],t[:2]),np.minimum(y[2:],t[2:])]
    if np.any(clip[2:]<=clip[:2]):return False
    inter=np.maximum(0,np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2])).prod()
    if inter/aa<.78:return False
    if not any(y[i]<t[i]-.08*w[i%2] if i<2 else y[i]>t[i]+.08*w[i%2] for i in a['tile_edges']):return False
    err=abs(x-clip)/np.r_[w,w]
    if max(err)>(.22 if b['tile_edges'] and min(w)/max(w)>=.72 and aa/bb<=.25 else .16):return False
    if not any(aw[k]<.76*w[k] and aw[1-k]>.63*w[1-k] for k in (0,1)):return False
    return True


def residual_mapping(kept):
    out={}
    for a in kept:
        if not a['tile_edges']:continue
        matches=[b for b in kept if residual_fragment(a,b)]
        if len(matches)==1:out[a['id']]=matches[0]['id']
    return {a:b for a,b in out.items() if b not in out}


def refine_residual_fragments(kept,suppressed,decisions):
    mapping=residual_mapping(kept)
    result=[]
    for a in decisions:
        row=dict(a);old=row['keeper_id']
        if old in mapping:
            row['prior_keeper_id']=old;row['keeper_id']=mapping[old];row['refinement_kept']=False
            row['fragment_rule']='normalized_clipped_footprint'
            if row['id']==old:row['refinement_reason']='tile_edge_fragment'
        result.append(row)
    supports=defaultdict(list)
    for row in result:supports[row['keeper_id']].append(row['id'])
    final=[dict(q,supporting_ids=supports[q['id']]) for q in kept if q['id'] not in mapping]
    discarded=[q for q in result if not q['refinement_kept']]
    ids={q['id'] for q in final}
    if len(final)+len(discarded)!=len(result) or ids!={q['id'] for q in result if q['refinement_kept']} or any(q['keeper_id'] not in ids for q in result):raise RuntimeError('Residual fragment accounting failed')
    return final,discarded,result,mapping


def nested_image_evidence(crop,offset,parent,children):
    """Conservative two-lobe check on original image, not a new detector.
    Ambiguous or flat profiles must not authorize removal.
    """
    from PIL import ImageFilter
    ox,oy=offset; b=np.asarray(parent['box'],float); c=np.asarray([q['box'] for q in children],float)
    sizes=c[:,2:]-c[:,:2];d=float(np.median(np.min(sizes,axis=1)))
    pad=max(4.,.35*d);l=max(0,int(math.floor(b[0]-ox-pad)));t=max(0,int(math.floor(b[1]-oy-pad)))
    r=min(crop.width,int(math.ceil(b[2]-ox+pad)));v=min(crop.height,int(math.ceil(b[3]-oy+pad)))
    if r-l<5 or v-t<5:return False,dict(reason='small_patch')
    gray=crop.crop((l,t,r,v)).convert('L').filter(ImageFilter.GaussianBlur(radius=max(.7,.08*d)))
    arr=np.asarray(gray,dtype=float);bg=float(np.percentile(arr,25));peaks=[]
    for box,wh in zip(c,sizes):
        center=(box[:2]+box[2:])/2-np.array([ox+l,oy+t]);rx,ry=np.maximum(1,wh*.20)
        x0=max(0,int(math.floor(center[0]-rx)));x1=min(arr.shape[1],int(math.ceil(center[0]+rx))+1)
        y0=max(0,int(math.floor(center[1]-ry)));y1=min(arr.shape[0],int(math.ceil(center[1]+ry))+1)
        if x0>=x1 or y0>=y1:return False,dict(reason='child_outside_patch')
        sub=arr[y0:y1,x0:x1];yy,xx=np.ogrid[y0:y1,x0:x1];core=((xx-center[0])/rx)**2+((yy-center[1])/ry)**2<=1
        if not core.any():return False,dict(reason='empty_child_core')
        vals=np.where(core,sub,-np.inf);px=float(np.max(vals)); coords=np.argwhere(vals>=px-1e-6);iy,ix=coords[np.argmin(np.sum((coords-np.array([center[1]-y0,center[0]-x0]))**2,axis=1))]
        peaks.append((float(ix+x0),float(iy+y0),px))
    drops=[];contrast=[p[2]-bg for p in peaks]
    for i,a in enumerate(peaks):
        for j in range(i+1,len(peaks)):
            z=peaks[j];dist=math.hypot(a[0]-z[0],a[1]-z[1]);
            if dist<.40*min(np.min(sizes[i]),np.min(sizes[j])):return False,dict(reason='same_peak')
            u=np.linspace(.23,.77,max(7,int(dist)))
            xs=np.clip(np.rint(a[0]+u*(z[0]-a[0])).astype(int),0,arr.shape[1]-1)
            ys=np.clip(np.rint(a[1]+u*(z[1]-a[1])).astype(int),0,arr.shape[0]-1)
            drop=float(min(a[2],z[2])-np.percentile(arr[ys,xs],20));drops.append(drop)
            if drop<max(E3_RULES['parent_min_saddle_drop'],.08*min(contrast[i],contrast[j])):
                return local_lobes(crop,offset,parent,children)
    ok=min(contrast)>=E3_RULES['parent_min_peak_contrast']
    return ok,dict(reason='distinct_lobes' if ok else 'low_contrast',min_peak_contrast=round(min(contrast),2),min_saddle_drop=round(min(drops),2))


def refine_nested_parents(kept,suppressed,decisions,crop,offset):
    """Suppress only one-level parent groups with intact, distinct children and
    local image evidence. No transitive merge, no child coordinate modifications.
    """
    q=[dict(x,review_reasons=list(x['review_reasons'])) for x in kept];n=len(q)
    if not n:return q,suppressed,decisions,[]
    boxes=np.asarray([x['box'] for x in q],float);wh=boxes[:,2:]-boxes[:,:2];areas=wh.prod(1);cent=(boxes[:,:2]+boxes[:,2:])/2
    suspects={};checks=[];rules=E3_RULES
    for i,b in enumerate(boxes):
        inter=np.maximum(0,np.minimum(boxes[:,2:],b[2:])-np.maximum(boxes[:,:2],b[:2])).prod(1)
        js=np.flatnonzero((inter/areas>=rules['parent_child_containment'])&(areas<=areas[i]*rules['parent_child_max_ratio'])&(areas>=areas[i]*rules['parent_min_child_ratio'])&np.all(cent>=b[:2],axis=1)&np.all(cent<=b[2:],axis=1)).tolist()
        if len(js)>=2:suspects[i]=js
    accepted={}
    # Parent-child chains and overlapping parent groups are left for review.
    for i,js in sorted(suspects.items()):
        b=boxes[i];row=dict(parent_id=q[i]['id'],parent_box=q[i]['box'],child_ids=[q[j]['id'] for j in js],child_boxes=[q[j]['box'] for j in js],action='review',reason='')
        if len(js)>rules['parent_max_children']:reason='too_many_children'
        elif any(j in suspects for j in js):reason='nested_chain'
        elif any(q[j]['tile_edges'] for j in js):reason='child_tile_edge'
        else:
            reason='';clipped=[]
            for j in js:clipped.append(np.r_[np.maximum(boxes[j,:2],b[:2]),np.minimum(boxes[j,2:],b[2:])])
            union=_rectangle_union_area(clipped)/areas[i];envelope=(np.max(np.asarray(clipped)[:,2:],axis=0)-np.min(np.asarray(clipped)[:,:2],axis=0)).prod()/areas[i]
            row.update(union_coverage=round(float(union),4),envelope_coverage=round(float(envelope),4))
            if min(areas[js])/max(areas[js])<rules['parent_min_child_area_similarity']:reason='child_size_mismatch'
            elif union<rules['parent_min_union_coverage'] or envelope<rules['parent_min_envelope_coverage']:reason='children_do_not_explain_parent'
            else:
                for ix,j in enumerate(js):
                    for k in js[ix+1:]:
                        distance=np.linalg.norm(cent[j]-cent[k])/max(np.min(wh[j]),np.min(wh[k]))
                        if refinement_box_metrics(boxes[j],boxes[k])[0]>rules['parent_pair_max_iou'] or distance<rules['parent_pair_min_separation']:reason='children_not_distinct'
                if not reason:
                    ok,evidence=nested_image_evidence(crop,offset,q[i],[q[j] for j in js]);row['image_evidence']=evidence
                    if ok:accepted[i]=js
                    else:reason=evidence['reason']
        row['reason']=reason or 'distinct_children_supported';checks.append(row)
    # Parent-child chains are blocked above. Shared stable children are allowed
    # if neither parent is a child.
    removed=set(accepted);retained_ids={q[i]['id'] for i in range(n) if i not in removed}
    byid={a['id']:i for i,a in enumerate(q)}
    for row in checks:
        i=byid[row['parent_id']]
        if i in removed:row['action']='suppress_parent'
        else:
            q[i]['review_needed']=True
            if 'nested_parent_uncertain' not in q[i]['review_reasons']:q[i]['review_reasons'].append('nested_parent_uncertain')
    mapping={q[i]['id']:[q[j]['id'] for j in js] for i,js in accepted.items()}
    if any(not set(v)<=retained_ids for v in mapping.values()):raise RuntimeError('Nested replacement chain')
    new=[]
    for a in decisions:
        a=dict(a);old=a['keeper_id']
        if old in mapping:
            a['prior_keeper_id']=old;a['keeper_id']=None;a['replacement_ids']=mapping[old];a['refinement_kept']=False
            if a['id']==old:a['refinement_reason']='nested_parent'
        new.append(a)
    final=[a for a in q if a['id'] in retained_ids];discard=[a for a in new if not a['refinement_kept']]
    if len(final)+len(discard)!=len(new) or {a['id'] for a in new if a['refinement_kept']}!=retained_ids:raise RuntimeError('Nested accounting failed')
    return final,discard,new,checks


def nested_check_preview(crop,offset,checks,sid,path):
    # Per-image close-ups: raw image and candidate overlays. No invented boxes.
    checks=sorted(checks,key=lambda a:(a['action']!='suppress_parent',a['parent_id']));shown=checks[:30];cols=4;cw=240;ch=152
    canvas=Image.new('RGB',(cols*cw,max(80,33+math.ceil(len(shown)/cols)*ch)),'#20242a');dr=ImageDraw.Draw(canvas)
    dr.text((5,5),f'{sid} | parent checks {len(checks)} | showing {len(shown)} | orange parent, green children',fill='white')
    if not shown:dr.text((5,32),'No nested parent suspects',fill='white')
    for i,a in enumerate(shown):
        b=a['parent_box'];ox,oy=offset;pad=.18*max(b[2]-b[0],b[3]-b[1]);l=max(0,int(b[0]-ox-pad));t=max(0,int(b[1]-oy-pad));r=min(crop.width,int(math.ceil(b[2]-ox+pad)));v=min(crop.height,int(math.ceil(b[3]-oy+pad)))
        if l>=r or t>=v:continue
        raw=crop.crop((l,t,r,v));scale=min(115/raw.width,115/raw.height);raw=raw.resize((max(1,round(raw.width*scale)),max(1,round(raw.height*scale))),Image.Resampling.LANCZOS);over=raw.copy();d=ImageDraw.Draw(over);sx=raw.width/(r-l);sy=raw.height/(v-t)
        def rect(z):return ((z[0]-ox-l)*sx,(z[1]-oy-t)*sy,(z[2]-ox-l)*sx,(z[3]-oy-t)*sy)
        d.rectangle(rect(b),outline='#ff963e',width=2)
        for z in a['child_boxes']:d.rectangle(rect(z),outline='#35e69a',width=2)
        x=(i%cols)*cw;y=30+(i//cols)*ch;canvas.paste(raw,(x,y));canvas.paste(over,(x+118,y));dr.text((x+2,y+117),a['action']+' '+a['parent_id'],fill='white');dr.text((x+2,y+130),a['reason'][:35],fill='#ffcb48')
    canvas.save(path,format='JPEG',quality=93)


def refinement_preview(crop,offset,circle,records,kept,suppressed,sid,path):
    w,h=crop.size;ox,oy=offset;im=crop.copy();im.thumbnail((1200,1200),Image.Resampling.LANCZOS);sx,sy=im.width/w,im.height/h
    def rect(b):return ((b[0]-ox)*sx,(b[1]-oy)*sy,(b[2]-ox)*sx,(b[3]-oy)*sy)
    def panel(groups):
        p=im.copy();d=ImageDraw.Draw(p);cx,cy,r=circle;d.ellipse(rect([cx-r,cy-r,cx+r,cy+r]),outline='#ffbd48',width=2)
        for rows,color in groups:
            for q in rows:d.rectangle(rect(q['box']),outline=color,width=2)
        return p
    before=[q for q in records if q['view'] in E3_VIEWS and q['global_nms_kept'] and q['inside_inner_circle']]
    panels=[panel([(before,'#35e69a')]),panel([(kept,'#35e69a'),([q for q in kept if q['review_needed']],'#ffcb48')]),panel([([q for q in suppressed if q['refinement_reason']=='near_identical_box'],'#f15c72'),([q for q in suppressed if q['refinement_reason']=='tile_edge_fragment'],'#43cfff'),([q for q in suppressed if q['refinement_reason']=='nested_parent'],'#ff963e'),([q for q in suppressed if q['refinement_reason']=='hollow_ring_artifact'],'#d977e8'),([q for q in suppressed if q['refinement_reason']=='local_single_blob_duplicate'],'#f15c72')])]
    titles=[f'A+B+C proposals: {len(before)}',f'Supported: {sum(not q["review_needed"] for q in kept)} | unresolved: {sum(q["review_needed"] for q in kept)}',f'Suppressed proposals: {len(suppressed)} | red: duplicate, cyan: fragment, orange: parent']
    canvas=Image.new('RGB',(3*im.width+20,im.height+59),'#20242a');d=ImageDraw.Draw(canvas)
    for i,(p,title) in enumerate(zip(panels,titles)):
        x=i*(im.width+10);d.text((x+4,7),sid+' | '+title,fill='white');canvas.paste(p,(x,28))
    d.text((4,canvas.height-21),'E3 local candidate verification; green supported, yellow unresolved; no verified true count.',fill='#edcb8a');canvas.save(path,format='JPEG',quality=94)


def refinement_final_preview(image,circle,kept,sid,path):
    """Full resolution EXIF-upright original image, retained boxes only."""
    im=image.convert('RGB').copy();w,h=im.size
    line=max(2,round(min(w,h)/850));d=ImageDraw.Draw(im)
    cx,cy,r=map(float,circle)
    d.ellipse((cx-r,cy-r,cx+r,cy+r),outline='#ffbd48',width=line)
    for q in kept:
        b=np.asarray(q['box'],float)
        if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[2:]<=b[:2]) or np.any(b[:2]<-1e-3) or b[2]>w+1e-3 or b[3]>h+1e-3:raise ValueError('Invalid retained box '+str(q['id']))
        d.rectangle(tuple(b),outline='#ffcb48' if q.get('review_needed',False) else '#35e69a',width=line)
    from PIL import ImageFont
    size=max(18,round(min(w,h)/65))
    try:font=ImageFont.truetype('arial.ttf',size)
    except OSError:
        try:font=ImageFont.load_default(size=size)
        except TypeError:font=ImageFont.load_default()
    unresolved=sum(bool(q.get('review_needed')) for q in kept)
    header=[f'{sid} | Supported: {len(kept)-unresolved} | Unresolved: {unresolved}',
            'Green: supported proposal | Yellow: unresolved, not counted | No verified count']
    bar=2*(size+9)+14;canvas=Image.new('RGB',(w,h+bar),'#20242a');canvas.paste(im,(0,bar));dr=ImageDraw.Draw(canvas)
    for i,t in enumerate(header):dr.text((12,6+i*(size+9)),t,fill='white' if i==0 else '#ffcb48',font=font)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.jpg.tmp');canvas.save(tmp,format='JPEG',quality=96,subsampling=0);tmp.replace(path)


def render_saved_refinement_previews(out,prior,lookup,chosen,explicit=False):
    """Render existing decisions only. Never mutate input, JSON or old previews."""
    pending=[]
    for sid in chosen:
        dp=out/'detections'/f'{sid}.json'
        if not dp.is_file():
            if explicit:raise FileNotFoundError(dp)
            continue
        saved=json.loads(dp.read_text(encoding='utf-8'));inp=prior/'detections'/f'{sid}.json';row=lookup[saved['image']];source=Path(row['source'])
        if saved['selection_id']!=sid or row['split']!='val' or saved['input_sha256']!=sha(inp) or saved['source_sha256']!=sha(source):raise ValueError('Saved input mismatch '+sid)
        kept=saved['kept'];decisions=saved['decisions'];ids=[q['id'] for q in kept]
        if len(ids)!=len(set(ids)) or set(ids)!={q['id'] for q in decisions if q['refinement_kept']} or len(kept)!=saved['summary']['retained_proposals']:raise ValueError('Retained accounting mismatch '+sid)
        pending.append((sid,saved,source))
    if not pending:raise SystemExit('No saved refinement detections. Run refine-detections first.')
    for sid,saved,source in pending:
        with Image.open(source) as f:im=ImageOps.exif_transpose(f).convert('RGB')
        if saved['image_size']!={'width':im.width,'height':im.height}:raise ValueError('Image dimensions changed '+sid)
        refinement_final_preview(im,saved['circle_inner'],saved['kept'],sid,out/'final'/f'{sid}.jpg')
        print('[FINAL PREVIEW]',sid,len(saved['kept']),flush=True)
    print(f'Rendered {len(pending)} saved results: {out / "final"}',flush=True)


def refinement_self_test():
    def q(i,b,t,conf=.8,view='original'):
        return dict(id=i,box=b,tile=t,confidence=conf,view=view,inside_inner_circle=True,global_nms_kept=True)
    bounds=(0,0,400,400);t=[0,0,200,200];u=[120,0,320,200]
    a=q('full',[90,95,150,145],t);b=q('same',[91,96,151,146],t,.9,'clahe');c=q('fragment',[120,95,150,145],u,.95);e=q('adjacent',[142,96,193,145],u);f=q('small',[108,108,120,120],t);z=q('other',[210,20,230,40],u,.99,'local_enhance_low')
    k,s,r=refine_candidates([a,b,c,e,f,z],bounds)
    assert {x['id'] for x in k}=={'same','adjacent','small'},(k,s)
    assert {x['id']:x['refinement_reason'] for x in s}=={'full':'near_identical_box','fragment':'tile_edge_fragment'}
    # No merging an overlap chain into one object.
    a=q('a',[50,50,100,100],t,.91);b=q('b',[56,50,106,100],t,.9,'clahe');c=q('c',[62,50,112,100],t,.89,'local_enhance')
    k,s,r=refine_candidates([a,b,c],bounds);assert len(k)==2
    # True crop border must not count as a clipped interior tile edge.
    assert not refinement_edges(q('border',[0,30,20,50],t),bounds)
    assert refine_candidates([],bounds)==([],[],[])
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        p=Path(d)/'p.jpg';refinement_preview(Image.new('RGB',(400,400)),(0,0),(200,200,190),[a,b,c],k,s,'TEST',p);assert p.is_file()
    # Two distinct touching bright lobes: keep children, not their union box.
    def run_nested(rows,im):
        k,s,r=refine_candidates(rows,bounds);return refine_nested_parents(k,s,r,im,(0,0))
    yy,xx=np.mgrid[:400,:400];lobes=np.maximum(np.exp(-((xx-60)**2+(yy-75)**2)/250),np.exp(-((xx-98)**2+(yy-75)**2)/250))
    im=Image.fromarray(np.uint8(55+120*lobes)).convert('RGB')
    parent=q('parent',[38,53,120,97],t,.95);left=q('left',[39,54,81,96],t,.87);right=q('right',[77,54,119,96],t,.86,'clahe')
    k,s,r,checks=run_nested([parent,left,right],im)
    assert {a['id'] for a in k}=={'left','right'},checks
    assert s[0]['replacement_ids']==['left','right'] and s[0]['keeper_id'] is None
    # Same geometry but one broad uniform object: no proof of two colonies.
    one=Image.new('RGB',(400,400),(55,55,55));ImageDraw.Draw(one).ellipse((38,53,120,97),fill=(175,175,175))
    k,s,r,checks=run_nested([parent,left,right],one);assert len(k)==3 and checks[0]['action']=='review',checks
    # One child is insufficient; two overlapping copies are not distinct.
    k,s,r,checks=run_nested([parent,left],im);assert len(k)==2 and not checks
    dup=q('left_copy',[42,55,83,97],t,.8,'clahe');k,s,r,checks=run_nested([parent,left,dup],im);assert 'parent' in {a['id'] for a in k}
    # An actual tile-clipped child must not authorize parent removal.
    edge=q('right_edge',[78,54,119,96],[78,0,278,200],.86,'clahe');k,s,r,checks=run_nested([parent,left,edge],im);assert 'parent' in {a['id'] for a in k}
    with TemporaryDirectory() as d:
        p=Path(d)/'checks.jpg';nested_check_preview(im,(0,0),checks,'TEST',p);assert p.is_file()
        p=Path(d)/'final.jpg';refinement_final_preview(im,(200,200,190),k,'TEST',p);assert p.is_file()
        with Image.open(p) as rendered:assert rendered.width==im.width and rendered.height>im.height
    # Regression fixtures copied from real V003/V010 JSON (rounded subpixel coordinates).
    def fixture(id,box,tile,edges):return dict(id=id,box=box,tile=tile,tile_edges=edges)
    a=fixture('V003_strip',[1900.0854,2134.9839,1969.3407,2144.0],[1768,1376,2536,2144],[3])
    b=fixture('V003_full',[1903.7834,2134.2254,1966.9001,2194.9514],[1768,2016,2536,2784],[])
    c=fixture('V010_strip',[1768.4,1524.7,1885.,1560.],[1117,792,1885,1560],[2,3])
    d=fixture('V010_full',[1766.6,1527.8,1892.2,1629.5],[1757,1432,2525,2200],[])
    assert residual_mapping([a,b])=={a['id']:b['id']}
    assert residual_mapping([c,d])=={c['id']:d['id']}
    assert residual_mapping([a,dict(b,id='other'),b])=={}
    assert residual_mapping([dict(a,tile_edges=[]),b])=={}
    kk,ss,dd,mp=refine_residual_fragments([a,b],[],[dict(x,keeper_id=x['id'],refinement_kept=True,refinement_reason='retained') for x in [a,b]])
    assert len(kk)==1 and len(ss)==1 and len(dd)==2 and kk[0]['supporting_ids']==[a['id'],b['id']]
    print('[PASS] V003/V010 clipping geometry; competing keeper guard; accounting')
    print('[PASS] final-only full-resolution preview')
    print('[PASS] original duplicate/fragment tests; nested two lobes; single blob retained; one child retained; overlapping child copies; tile-edge guard; preview')


def e3_radial(gray,box,circle):
    """Local filled-core / hollow-ring evidence. No full-plate mask."""
    import cv2
    b=np.array(box,float);w=b[2:]-b[:2];c=(b[2:]+b[:2])/2;shape=float(min(w)/max(w));cx,cy,R=circle;rim=float((R-np.hypot(c[0]-cx,c[1]-cy))/max(w))
    rr=np.arange(.05,1.61,.06);ang=np.arange(48)*2*np.pi/48;l,t=np.maximum(0,np.floor(c-w)).astype(int);u,v=np.minimum([gray.shape[1],gray.shape[0]],np.ceil(c+w)).astype(int)
    if min(u-l,v-t)<5:return dict(kind='uncertain',reason='small_roi')
    p=cv2.GaussianBlur(gray[t:v,l:u].astype(np.float32),(0,0),max(.6,min(w)*.018))
    X=(c[0]-l+np.cos(ang)[:,None]*rr*w[0]/2).astype(np.float32);Y=(c[1]-t+np.sin(ang)[:,None]*rr*w[1]/2).astype(np.float32)
    z=cv2.remap(p,X,Y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    core=np.median(z[:,rr<.37],1);ring=np.max(z[:,(rr>.45)&(rr<1.15)],1);bg=np.median(z[:,rr>1.3],1)
    C=float(np.median(core-bg));D=float(np.median(ring-core));P=float(np.mean((ring-core>6)&(ring-bg>12)));S=float(np.median(ring-bg));F=float(np.mean(core-bg>5))
    kind='uncertain';reason='weak_or_irregular_local_evidence'
    if shape>=.65 and rim>.65 and D>=12 and P>=.80 and S>=18 and C<=.65*S:kind='reject';reason='hollow_ring_artifact'
    elif shape>=.48 and C>=9 and F>=.62 and (D<12 or C>.70*S):kind='support';reason='filled_core'
    return dict(kind=kind,reason=reason,core=round(C,2),ring_drop=round(D,2),ring_fraction=round(P,3),ring_signal=round(S,2),filled_fraction=round(F,3),aspect=round(shape,3),rim=round(rim,3))


def e3_single_pair(gray,a,b):
    """Conservative same-object check at two local segmentation thresholds."""
    import cv2
    x=np.array(a['box'],float);y=np.array(b['box'],float);wh=x[2:]-x[:2];vh=y[2:]-y[:2];iou,ios,ratio,delta=refinement_box_metrics(x,y)
    if ios<.77 or ratio<.45 or delta>.22:return False
    if min(wh)/max(wh)<.52 or min(vh)/max(vh)<.52:return False
    size=min(min(wh),min(vh));l,t=np.maximum(0,np.floor(np.minimum(x[:2],y[:2])-.45*size)).astype(int);r,v=np.minimum([gray.shape[1],gray.shape[0]],np.ceil(np.maximum(x[2:],y[2:])+.45*size)).astype(int)
    patch=gray[t:v,l:r].astype(np.float32)
    if min(patch.shape)<5:return False
    sm=cv2.GaussianBlur(patch,(0,0),max(.65,.035*size));bg=float(np.percentile(sm,20));hi=float(np.percentile(sm,92))
    if hi-bg<12:return False
    cs=[(z[:2]+z[2:])/2-np.array([l,t]) for z in (x,y)]
    for f in (.38,.55):
        mask=np.uint8(sm>bg+f*(hi-bg));n,lab,st,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        ids=[int(lab[int(np.clip(round(c[1]),0,lab.shape[0]-1)),int(np.clip(round(c[0]),0,lab.shape[1]-1))]) for c in cs]
        if ids[0]==0 or ids[0]!=ids[1]:return False
        comp=np.uint8(lab==ids[0]);cont,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cnt=max(cont,key=cv2.contourArea);area=cv2.contourArea(cnt);hull=cv2.contourArea(cv2.convexHull(cnt));per=cv2.arcLength(cnt,True)
        if hull<=0 or per<=0 or area/hull<.95 or 4*np.pi*area/per**2<.72:return False
        _,_,bw,bh=cv2.boundingRect(cnt)
        if min(bw,bh)/max(bw,bh)<.67 or max(bw,bh)>1.18*max(max(wh),max(vh)):return False
        dist=cv2.distanceTransform(comp,cv2.DIST_L2,5);mx=float(dist.max())
        if mx<.18*size:return False
        yy,xx=np.where(dist>=.94*mx)
        if np.hypot(np.ptp(xx),np.ptp(yy))>.42*size:return False
        for z in (x,y):
            l0,t0=np.maximum(0,np.floor(z[:2]-[l,t])).astype(int);r0,v0=np.minimum([comp.shape[1],comp.shape[0]],np.ceil(z[2:]-[l,t])).astype(int)
            if np.count_nonzero(comp[t0:v0,l0:r0])<.76*np.count_nonzero(comp):return False
    return True


def e3_decide(gray,kept,circle):
    q=[dict(a) for a in kept];features={a['id']:e3_radial(gray,a['box'],circle) for a in q};ids=[a['id'] for a in q];ar=np.array([a['box'] for a in q],float).reshape(-1,4);wh=ar[:,2:]-ar[:,:2];areas=wh.prod(1);cent=(ar[:,:2]+ar[:,2:])/2
    removed={a['id']:dict(reason='hollow_ring_artifact',replacement_ids=[]) for a in q if features[a['id']]['kind']=='reject'}
    # Only isolated, mutually supported duplicate pairs. No transitive merges.
    edges=[set() for a in q]
    for i,a in enumerate(q):
        if a['id'] in removed:continue
        inter=np.maximum(0,np.minimum(ar[:,2:],ar[i,2:])-np.maximum(ar[:,:2],ar[i,:2])).prod(1);ios=inter/np.minimum(areas,areas[i]);rat=np.minimum(areas,areas[i])/np.maximum(areas,areas[i]);distance=np.linalg.norm(cent-cent[i],axis=1)/np.minimum(np.linalg.norm(wh,axis=1),np.linalg.norm(wh[i]))
        for j in np.flatnonzero((ios>=.77)&(rat>=.45)&(distance<=.22)&(np.arange(len(q))>i)):
            j=int(j)
            if q[j]['id'] not in removed and e3_single_pair(gray,a,q[j]):edges[i].add(j);edges[j].add(i)
    for i,a in enumerate(q):
        if a['id'] in removed or len(edges[i])!=1:continue
        j=next(iter(edges[i]))
        if j<=i or edges[j]!={i}:continue
        def rank(k):return (features[ids[k]]['kind']=='support',not q[k]['tile_edges'],q[k]['confidence'],areas[k],ids[k])
        keep,drop=sorted([i,j],key=rank,reverse=True);removed[ids[drop]]=dict(reason='local_single_blob_duplicate',replacement_ids=[ids[keep]])
    survivors=[i for i,a in enumerate(q) if a['id'] not in removed];decided=[]
    for i in survivors:
        a=q[i];f=features[a['id']];other=ar[survivors];oa=areas[survivors];oc=cent[survivors];inter=np.maximum(0,np.minimum(other[:,2:],ar[i,2:])-np.maximum(other[:,:2],ar[i,:2])).prod(1);ios=inter/np.minimum(oa,areas[i]);rat=np.minimum(oa,areas[i])/np.maximum(oa,areas[i]);dist=np.linalg.norm(oc-cent[i],axis=1)/np.minimum(np.linalg.norm(wh[survivors],axis=1),np.linalg.norm(wh[i]))
        conflict=[ids[survivors[j]] for j in np.flatnonzero((ios>=.67)&((rat<.78)|(dist<.23))&(np.array(survivors)!=i))]
        reasons=[]
        if conflict:reasons.append('unresolved_overlap')
        if f['kind']!='support':reasons.append(f['reason'])
        if a['tile_edges'] and (f.get('aspect',0)<.62 or f['kind']!='support'):reasons.append('unresolved_tile_edge')
        if 'nested_parent_uncertain' in a.get('review_reasons',[]) and conflict:reasons.append('unresolved_parent')
        a.update(local_evidence=f,review_needed=bool(reasons),review_reasons=reasons,conflict_ids=conflict,final_decision='unresolved' if reasons else 'accepted')
        decided.append(a)
    return decided,removed,features


def e3_finalize(image,circle,kept,decisions):
    """Preserve every original proposal and all stage decisions in the audit."""
    final,removed,features=e3_decide(np.asarray(image.convert('L')),kept,circle)
    lookup={a['id']:a for a in final};old_ids={a['id'] for a in kept}
    assert set(lookup)|set(removed)==old_ids and not (set(lookup)&set(removed))
    def resolve(i):
        if i in lookup:return [i]
        return removed.get(i,{}).get('replacement_ids',[])
    audit=[]
    for a in decisions:
        z=dict(a);original_keeper=z.get('keeper_id');repl=z.get('replacement_ids',[])
        targets=sorted(set(j for i in ([original_keeper] if original_keeper else repl) for j in resolve(i)))
        if original_keeper in removed:
            z['local_prior_keeper_id']=original_keeper;z['keeper_id']=targets[0] if len(targets)==1 else None;z['replacement_ids']=targets
            z['refinement_kept']=False
            if z['id']==original_keeper:z['refinement_reason']=removed[original_keeper]['reason']
        elif original_keeper is None and repl:z['replacement_ids']=targets
        z['final_decision']=lookup[z['id']]['final_decision'] if z['id'] in lookup else 'suppressed'
        z['local_evidence']=features.get(z['id'])
        z['final_replacement_ids']=targets
        audit.append(z)
    for a in final:
        a['supporting_ids']=[z['id'] for z in audit if z['keeper_id']==a['id']]
    discarded=[z for z in audit if not z['refinement_kept']]
    assert len(final)+len(discarded)==len(audit) and {z['id'] for z in audit if z['refinement_kept']}==set(lookup)
    assert all(set(z['final_replacement_ids'])<=set(lookup) for z in audit)
    return final,discarded,audit,removed


def e3_countability(kept):
    """Heuristic crowding warning, never a true TNTC classification."""
    n=len(kept);unresolved=sum(a['review_needed'] for a in kept);crowded=0
    if n:
        b=np.array([a['box'] for a in kept],float);w=b[:,2:]-b[:,:2];c=(b[:,:2]+b[:,2:])/2;d=np.sqrt(w.prod(1))
        for i in range(n):
            dist=np.linalg.norm(c-c[i],axis=1);close=(dist<.95*(d+d[i])/2)&(np.arange(n)!=i)
            if close.sum()>=2:crowded+=1
    fraction=crowded/max(1,n)
    status='DENSE_REVIEW_REQUIRED' if fraction>=.15 and crowded>=10 else 'UNRESOLVED_REVIEW_REQUIRED' if unresolved else 'PROVISIONAL_ESTIMATE_ONLY'
    return dict(accepted_proposals=n-unresolved,unresolved_proposals=unresolved,crowded_proposals=crowded,crowded_fraction=round(fraction,4),countability_status=status,provisional_count=n if status=='PROVISIONAL_ESTIMATE_ONLY' else None)


def e3_final_self_test():
    import cv2
    from PIL import ImageDraw
    im=Image.new('L',(360,240),65);dr=ImageDraw.Draw(im);dr.ellipse((35,35,95,95),fill=155);dr.ellipse((160,35,220,95),outline=190,width=7);dr.ellipse((270,35,330,95),fill=71)
    def q(i,b,edge=()):return dict(id=i,box=b,confidence=.8,tile_edges=list(edge),review_needed=True,review_reasons=['old_review'],supporting_ids=[i])
    a=q('full',[32,32,98,98]);b=q('ring',[157,32,223,98]);c=q('faint',[267,32,333,98]);circle=(180,120,300)
    k,s,f=e3_decide(np.array(im),[a,b,c],circle)
    assert s['ring']['reason']=='hollow_ring_artifact',(s,f)
    assert {z['id']:z['final_decision'] for z in k}=={'full':'accepted','faint':'unresolved'},(k,f)
    # Near-duplicate with local evidence, and preserve distinct touching disks.
    duplicate=q('offset',[39,34,100,100]);k,s,f=e3_decide(np.array(im),[a,duplicate],circle);assert len(k)==1 and len(s)==1,(k,s)
    two=Image.new('L',(360,240),65);dr=ImageDraw.Draw(two);dr.ellipse((35,35,95,95),fill=155);dr.ellipse((89,35,149,95),fill=155)
    k,s,f=e3_decide(np.array(two),[a,q('neighbor',[86,32,152,98])],circle);assert len(k)==2 and not s
    ds=[dict(z,keeper_id=z['id'],refinement_kept=True,refinement_reason='retained') for z in [a,b,c]]
    k,s,ds,rm=e3_finalize(im,circle,[a,b,c],ds);assert len(k)==2 and len(s)==1 and len(ds)==3
    assert e3_countability(k)['provisional_count'] is None
    print('[PASS] E3 local filled-core, hollow-ring, weak-evidence, duplicate, adjacent-colony, audit and countability checks')


def refinement_inner_rim(image, outer, candidates):
    """Estimate the circular inner agar/rim boundary around a fitted dish center."""
    import cv2
    gray=np.asarray(image.convert('L'),dtype=np.uint8)
    smooth=cv2.GaussianBlur(gray,(0,0),2.0)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
    gradient=cv2.magnitude(gx,gy)
    cx,cy,r=map(float,outer)
    angles=np.linspace(0,2*np.pi,720,endpoint=False)

    profile=[]
    for frac in np.linspace(.82,.965,59):
        rr=r*float(frac)
        xs=np.clip(np.rint(cx+rr*np.cos(angles)).astype(int),0,gray.shape[1]-1)
        ys=np.clip(np.rint(cy+rr*np.sin(angles)).astype(int),0,gray.shape[0]-1)
        vals=gradient[ys,xs]
        quadrant=[float(np.percentile(vals[k*180:(k+1)*180],65)) for k in range(4)]
        mean_q=float(np.mean(quadrant));min_q=float(min(quadrant))
        score=mean_q+1.5*min_q
        profile.append(dict(frac=float(frac),score=score,mean_q65=mean_q,min_q65=min_q,p75=float(np.percentile(vals,75))))

    best=max(profile,key=lambda z:z['score'])
    credible=best['score']>=20 and best['min_q65']>=5
    ring_frac=float(best['frac']) if credible else .93

    sizes=[]
    for q in candidates:
        if not q.get('global_nms_kept') or q.get('view') not in E3_VIEWS:continue
        b=np.asarray(q['box'],float);wh=b[2:]-b[:2]
        if np.all(wh>0):sizes.append(float(np.sqrt(wh.prod())))
    median_side=float(np.median(sizes)) if sizes else .025*r
    # Put the counting center boundary slightly inside the physical ring. Colony
    # centers can still be close to the wall, but rim glare centered on the ring
    # itself is excluded.
    margin_px=float(np.clip(.10*median_side,5.,12.))
    inner_frac=float(np.clip(ring_frac-margin_px/max(r,1.),.84,.95))
    debug=dict(selected=best,credible=bool(credible),ring_fraction=ring_frac,inner_fraction=inner_frac,
        margin_px=margin_px,median_candidate_side=median_side,top_profile=sorted(profile,key=lambda z:z['score'],reverse=True)[:12])
    return (cx,cy,r*inner_frac),debug


def refinement_refit_roi(image, candidates, old_outer, old_inner):
    """Refit the full dish rim, then independently fit the inner counting ring."""
    from preprocessing.plate_hough import estimate_plate_hough
    fit,debug=estimate_plate_hough(image,return_debug=True)
    outer=tuple(float(x) for x in old_outer)
    source='prior_roi_fallback'
    if fit is not None:
        proposal=tuple(float(x) for x in fit[:3])
        shift=math.hypot(proposal[0]-outer[0],proposal[1]-outer[1])
        ratio=proposal[2]/max(1.,outer[2])
        selected=debug.get('selected') or {}
        if shift<=.18*outer[2] and .82<=ratio<=1.20 and selected.get('quadrant_support',0)>=.12:
            outer=proposal
            source='scored_hough_ast_style'

    inner,inner_debug=refinement_inner_rim(image,outer,candidates)

    # The gradient-derived inner ring can be too conservative on photographs where
    # the plastic wall creates a strong secondary circle. Use detector coverage only
    # as a sanity expansion (never as the circle fit itself): almost all plausible
    # globally-kept candidate centers should remain countable, while the physical rim
    # still stays outside the counting circle.
    cx,cy,r=outer
    radial=[]
    for q in candidates:
        if not q.get('global_nms_kept') or q.get('view') not in E3_VIEWS:continue
        if float(q.get('confidence',0))<.25:continue
        b=np.asarray(q['box'],float);c=(b[:2]+b[2:])/2
        frac=float(math.hypot(float(c[0])-cx,float(c[1])-cy)/max(r,1.))
        if frac<=1.02:radial.append(frac)
    coverage_frac=float(np.percentile(radial,99)) if radial else .0
    coverage_target=float(np.clip(coverage_frac+.012,.91,.965)) if radial else .91
    current_frac=float(inner[2]/max(r,1.))
    final_frac=float(max(current_frac,coverage_target))
    inner=(cx,cy,r*final_frac)
    inner_debug=dict(inner_debug,candidate_center_p99=coverage_frac,
        candidate_coverage_target=coverage_target,pre_coverage_inner_fraction=current_frac,
        final_inner_fraction=final_frac,candidate_centers_used=len(radial))

    return outer,inner,dict(
        source=source,
        old_outer=list(map(float,old_outer)),
        old_inner=list(map(float,old_inner)),
        hough_selected=debug.get('selected'),
        hough_candidates=debug.get('candidates',[])[:12],
        inner_rim=inner_debug,
        inner_radius=float(inner[2]),
    )

def refinement_main(argv):
    p=argparse.ArgumentParser(description='E3 duplicate/artifact refinement with image-based ROI refit')
    p.add_argument('--ids',nargs='+',help='Subset V001-V050; default all')
    p.add_argument('--final-previews',action='store_true',help='Render final-only images from saved JSON; no filtering or inference')
    p.add_argument('--force',action='store_true',help='Recompute selected E3 outputs in-place with current ROI/refinement logic')
    p.add_argument('--self-test',action='store_true');a=p.parse_args(argv)
    if a.self_test:refinement_self_test();e3_final_self_test();return
    base=ROOT/'outputs/diagnostics/preprocessing';prior=base/'local_tile_enhancement';cov=base/'coverage_review';out=ROOT/'outputs/diagnostics/refinement'
    cfg=json.loads((prior/'config.json').read_text(encoding='utf-8'));done=json.loads((prior/'complete.json').read_text(encoding='utf-8'));cc=json.loads((cov/'config.json').read_text(encoding='utf-8'))
    ids=[f'V{i:03d}' for i in range(1,51)];chosen=a.ids or ids
    if cfg['scope']!='validation_local_tile_enhancement_only' or cfg['selection_ids']!=ids or done['status']!='COMPLETE' or done['images']!=50 or cfg['prior_config_sha256']!=sha(cov/'config.json') or cfg['foreground_used'] or cfg['tile']!=768 or cfg['stride']!=640 or cfg['confidence_c']!=.25:raise SystemExit('Expected complete A/B/C validation results')
    if len(set(chosen))!=len(chosen) or not set(chosen)<=set(ids):p.error('Use unique V001-V050 IDs')
    split=ROOT/'annotations/metadata/teacher_split_1200.csv'
    if sha(split)!=cfg['split_manifest_sha256']:raise SystemExit('Manifest changed')
    lookup={r['image']:r for r in read_csv(split)}
    if a.final_previews:
        render_saved_refinement_previews(out,prior,lookup,chosen,explicit=bool(a.ids));return
    from prepare_plate_images import crop_plate
    plate_hough_path=ROOT/'src/preprocessing/plate_hough.py'
    settings=dict(scope='validation_e3_local_candidate_verification',selection_ids=ids,views=list(E3_VIEWS),rules=E3_RULES,prior_config_sha256=sha(prior/'config.json'),model_sha256=cfg['model_sha256'],split_manifest_sha256=sha(split),script_sha256=sha(Path(__file__)),plate_hough_sha256=sha(plate_hough_path),roi_refit='AST-style scored Hough outer rim + per-image gradient inner counting ring',foreground_used=False,low_confidence_view_used=False,independent_ground_truth=False,final_count=False)
    if out.exists():
        if not (out/'config.json').is_file():raise SystemExit('Missing refinement config')
        existing=json.loads((out/'config.json').read_text(encoding='utf-8'))
        if existing!=settings:
            if not a.force:raise SystemExit('Refinement code/ROI changed; rerun with --force to replace E3 outputs in-place')
            tmp=out/'config.json.tmp'
            tmp.write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
            tmp.replace(out/'config.json')
    else:out.mkdir(parents=True);write_json(out/'config.json',settings)
    for sub in ('detections','previews','parent_checks'):(out/sub).mkdir(exist_ok=True)
    for sid in chosen:
        inp=prior/'detections'/f'{sid}.json';old=json.loads(inp.read_text(encoding='utf-8'));ih=sha(inp);row=lookup[old['image']];source=Path(row['source'])
        if old['selection_id']!=sid or old['split']!='val' or row['split']!='val' or old['scope']!=cfg['scope'] or old['status']!='PROPOSALS_NOT_COLONY_COUNT' or sha(source)!=old['source_sha256'] or old['prior_detection_sha256']!=sha(cov/'detections'/f'{sid}.json'):raise ValueError('Input mismatch '+sid)
        dp=out/'detections'/f'{sid}.json';pp=out/'previews'/f'{sid}.jpg';np_path=out/'parent_checks'/f'{sid}.jpg'
        if dp.exists() and pp.exists() and np_path.exists() and not a.force:
            saved=json.loads(dp.read_text(encoding='utf-8'))
            if saved['input_sha256']!=ih or saved['preview_sha256']!=sha(pp) or saved['parent_checks_sha256']!=sha(np_path):raise ValueError('Output changed '+sid)
            print('[SKIP]',sid,flush=True);continue
        with Image.open(source) as f:im=ImageOps.exif_transpose(f).convert('RGB')
        if old['image_size']!={'width':im.width,'height':im.height}:raise ValueError('Image dimensions changed '+sid)
        outer,circle,roi_refinement=refinement_refit_roi(im,old['candidates'],tuple(old['circle_outer']),tuple(old['circle_inner']))
        # Re-evaluate circle membership in original coordinates. This recovers candidates
        # that were detected by tiled inference but excluded only by the old frozen ROI.
        recandidates=[]
        newly_inside=0
        for q in old['candidates']:
            z=dict(q);b=np.asarray(z['box'],float);c=(b[:2]+b[2:])/2
            inside=math.hypot(float(c[0])-circle[0],float(c[1])-circle[1])<=circle[2]
            if inside and not z.get('inside_inner_circle',False):newly_inside+=1
            z['inside_inner_circle']=bool(inside);recandidates.append(z)
        # Tight crop is only for local image processing/preview. Candidate coordinates
        # remain in the original image frame.
        crop,off=crop_plate(im,outer,padding=1.02)
        bounds=(*off,off[0]+crop.width,off[1]+crop.height)
        kept,suppressed,records=refine_candidates(recandidates,bounds)
        kept,suppressed,records,residual=refine_residual_fragments(kept,suppressed,records)
        before_nested=len(kept)
        kept,suppressed,records,checks=refine_nested_parents(kept,suppressed,records,crop,off)
        before_local=len(kept)
        kept,suppressed,records,local_removed=e3_finalize(im,circle,kept,records)
        countability=e3_countability(kept)
        summary=dict(selection_id=sid,image=old['image'],input_proposals=len(records),roi_source=roi_refinement['source'],roi_newly_inside_candidates=newly_inside,retained_before_nested=before_nested,retained_before_local=before_local,local_artifacts_suppressed=sum(x['reason']=='hollow_ring_artifact' for x in local_removed.values()),local_duplicates_suppressed=sum(x['reason']=='local_single_blob_duplicate' for x in local_removed.values()),**countability,residual_tile_fragments_suppressed=len(residual),retained_proposals=len(kept),nested_parent_suppressed=sum(q['refinement_reason']=='nested_parent' for q in suppressed),nested_parent_review=sum(q['action']=='review' for q in checks),near_identical_suppressed=sum(q['refinement_reason']=='near_identical_box' for q in suppressed),tile_fragments_suppressed=sum(q['refinement_reason']=='tile_edge_fragment' for q in suppressed),review_needed=sum(q['review_needed'] for q in kept),status='PROVISIONAL_REFINEMENT_NOT_VERIFIED_COUNT')
        tmp=pp.with_suffix('.jpg.tmp');refinement_preview(crop,off,circle,recandidates,kept,suppressed,sid,tmp);tmp.replace(pp)
        tmp=np_path.with_suffix('.jpg.tmp');nested_check_preview(crop,off,checks,sid,tmp);tmp.replace(np_path)
        result=dict(local_verification_removed=local_removed,residual_fragment_mapping=residual,parent_checks_sha256=sha(np_path),nested_parent_checks=checks,selection_id=sid,image=old['image'],source_sha256=old['source_sha256'],input_sha256=ih,preview_sha256=sha(pp),circle_outer=list(outer),circle_inner=list(circle),roi_refinement=roi_refinement,crop_origin=list(off),crop_bounds=list(bounds),image_size=old['image_size'],coordinate_frame=old['coordinate_frame'],kept=kept,suppressed=suppressed,decisions=records,summary=summary,independent_ground_truth=False,status=summary['status'])
        tmp=dp.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(dp);print(summary,flush=True)
    rows=[]
    for sid in ids:
        dp=out/'detections'/f'{sid}.json';pp=out/'previews'/f'{sid}.jpg';np_path=out/'parent_checks'/f'{sid}.jpg'
        if not (dp.exists() and pp.exists() and np_path.exists()):continue
        saved=json.loads(dp.read_text(encoding='utf-8'))
        if saved['input_sha256']!=sha(prior/'detections'/f'{sid}.json') or saved['preview_sha256']!=sha(pp) or saved['parent_checks_sha256']!=sha(np_path):raise ValueError('Saved result changed '+sid)
        rows.append(saved['summary'])
    tmp=out/'candidate_summary.csv.tmp'
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields:fields.append(key)
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
    tmp.replace(out/'candidate_summary.csv');tmp=out/'complete.json.tmp';tmp.write_text(json.dumps(dict(images=len(rows),expected=50,status='COMPLETE' if len(rows)==50 else 'PARTIAL',note='E3 candidate verification completed computationally; accuracy and countability not independently verified.'),indent=2),encoding='utf-8');tmp.replace(out/'complete.json')
    print(f'SAVED {len(rows)}/50: {out}',flush=True)
    render_saved_refinement_previews(out,prior,lookup,chosen,explicit=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='experiment':experiment_main(sys.argv[2:])
    elif len(sys.argv)>1 and sys.argv[1]=='candidate-coverage':coverage_main(sys.argv[2:])
    elif len(sys.argv)>1 and sys.argv[1]=='multiscale':multiscale_main(sys.argv[2:])
    elif len(sys.argv)>1 and sys.argv[1]=='foreground-roi':foreground_main(sys.argv[2:])
    elif len(sys.argv)>1 and sys.argv[1]=='tile-enhancement':tile_enhancement_main(sys.argv[2:])
    elif len(sys.argv)>1 and sys.argv[1]=='refine-detections':refinement_main(sys.argv[2:])
    else:legacy_main()

if __name__=='__main__':main()
