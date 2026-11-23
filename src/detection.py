from __future__ import annotations

"""Detection and refinement utilities used by ColonyVision."""

import math
from collections import defaultdict

import numpy as np
from PIL import Image

def positions(length, tile, stride):
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs

def infer(model, image, tile, stride, conf, imgsz, iou, device, *, pil_input=False, audit=None, patch_transform=None):
    """Return working-image xyxy/conf detections and count before global NMS."""
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

def inside(box, circle):
    cx, cy, r = circle
    return ((box[0]+box[2])/2-cx)**2 + ((box[1]+box[3])/2-cy)**2 <= r*r

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
    from preprocessing.plate import estimate_plate_hough
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
