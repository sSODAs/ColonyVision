"""Prepare ADBC public, expert-annotated one-class tiled YOLO dataset.
Source: https://doi.org/10.6084/m9.figshare.22022540.v3 (CC BY 4.0).
"""
import argparse, csv, hashlib, json, random, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/"raw"/"external"/"ADBC"
API="https://api.figshare.com/v2/articles/22022540"

def md5(path):
    h=hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def download(entry,folder):
    folder.mkdir(parents=True,exist_ok=True)
    dest=folder/entry["name"]
    digest=entry.get("supplied_md5") or entry.get("computed_md5")
    if dest.is_file() and dest.stat().st_size==entry["size"] and (not digest or md5(dest)==digest):
        return entry["name"]
    tmp=dest.with_suffix(dest.suffix+".part")
    for attempt in range(3):
        try:
            req=urllib.request.Request(entry["download_url"],headers={"User-Agent":"ColonyVision/1.0"})
            with urllib.request.urlopen(req,timeout=90) as r, tmp.open("wb") as f:
                while block:=r.read(1<<20):f.write(block)
            if tmp.stat().st_size!=entry["size"] or (digest and md5(tmp)!=digest):
                raise RuntimeError("size or checksum mismatch")
            tmp.replace(dest)
            return entry["name"]
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt==2:raise
            time.sleep(attempt+1)

def positions(length,tile,stride):
    if length<=tile:return [0]
    arr=list(range(0,length-tile+1,stride))
    if arr[-1]!=length-tile:arr.append(length-tile)
    return arr

def get_boxes(z,name,w,h):
    key=Path(name).with_suffix(".txt").name
    boxes=[]
    for ln in z.read(key).decode("utf-8-sig").splitlines():
        if not ln.strip():continue
        c,x,y,bw,bh=map(float,ln.split())
        if not(0<=c<=23 and 0<=x<=1 and 0<=y<=1 and 0<=bw<=1 and 0<=bh<=1):
            raise ValueError(f"bad source annotation {name}")
        if bw==0 or bh==0:
            # Official ADBC labels contain 3 degenerate zero-height boxes.
            # They cannot be used to train YOLO; skip and preserve the raw archive.
            print(f"WARNING zero-area source label skipped in {name}",flush=True)
            continue
        boxes.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
    return boxes

def make_tiles(path,z,split,out,tile,stride):
    with Image.open(path) as raw:
        # The official YOLO coordinates refer to the EXIF-corrected visual orientation.
        # Verified on sp14/sp19/sp21 examples with orientation=6.
        image=ImageOps.exif_transpose(raw).convert("RGB")
    w,h=image.size
    boxes=get_boxes(z,path.name,w,h)
    n=0
    for top in positions(h,tile,stride):
        for left in positions(w,tile,stride):
            right=min(w,left+tile); bottom=min(h,top+tile)
            width=right-left;height=bottom-top
            if min(width,height)<100:continue
            labels=[]
            for x0,y0,x1,y1 in boxes:
                cx=(x0+x1)/2;cy=(y0+y1)/2
                if not(left<=cx<right and top<=cy<bottom):continue
                u0=max(left,x0);v0=max(top,y0);u1=min(right,x1);v1=min(bottom,y1)
                if u1<=u0 or v1<=v0:continue
                if (u1-u0)*(v1-v0)/max(1,(x1-x0)*(y1-y0))<0.5:continue
                labels.append("0 %.8f %.8f %.8f %.8f"%(
                    ((u0+u1)/2-left)/width,((v0+v1)/2-top)/height,
                    (u1-u0)/width,(v1-v0)/height))
            key=f"{path.stem}_x{left}_y{top}"
            photo=out/"images"/split/(key+".jpg")
            label=out/"labels"/split/(key+".txt")
            photo.parent.mkdir(parents=True,exist_ok=True)
            label.parent.mkdir(parents=True,exist_ok=True)
            if photo.exists()!=label.exists():
                raise RuntimeError("Incomplete tile: "+key)
            if not photo.exists():
                image.crop((left,top,right,bottom)).save(photo,quality=90)
                label.write_text("\n".join(labels)+("\n" if labels else ""),encoding="utf-8")
            n+=1
    return len(boxes),n

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--name",default="adbc_pilot")
    p.add_argument("--max-per-species",type=int,default=1,help="1=24-image smoke test; 0=all 369")
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--tile",type=int,default=768)
    p.add_argument("--stride",type=int,default=640)
    args=p.parse_args()
    if args.max_per_species<0 or not(0<args.stride<=args.tile):raise SystemExit("Bad arguments")
    with urllib.request.urlopen(API,timeout=45) as resp: info=json.load(resp)
    all_images=sorted([f for f in info["files"] if f["name"].lower().endswith(".jpg")],key=lambda x:x["name"])
    chosen=[];per={}
    for f in all_images:
        group=f["name"].split("_")[0]
        if args.max_per_species==0 or per.get(group,0)<args.max_per_species:
            chosen.append(f);per[group]=per.get(group,0)+1
    archive=next(x for x in info["files"] if x["name"]=="annot_YOLO.zip")
    print("Images:",len(chosen),"of",len(all_images),"groups:",len(per),flush=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    (CACHE/"source_metadata.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    download(archive,CACHE)
    with ThreadPoolExecutor(max_workers=min(12,max(1,args.workers))) as pool:
        pending=[pool.submit(download,f,CACHE/"images") for f in chosen]
        for i,future in enumerate(as_completed(pending),1):
            item=future.result()
            if i%10==0 or i==len(pending):
                print(f"Downloaded {i}/{len(pending)}: {item}",flush=True)
    names=sorted(per)
    random.Random(42).shuffle(names)
    splits={v:("test" if i<2 else "val" if i<5 else "train") for i,v in enumerate(names)}
    target=ROOT/"dataset"/args.name
    manifest=[]
    with zipfile.ZipFile(CACHE/"annot_YOLO.zip") as z:
        for i,file in enumerate(chosen,1):
            name=file["name"];group=name.split("_")[0]
            split=splits[group]
            boxes,tiles=make_tiles(CACHE/"images"/name,z,split,target,args.tile,args.stride)
            manifest.append((name,group,split,boxes,tiles))
            if i%5==0 or i==len(chosen):
                print(f"Prepared {i}/{len(chosen)} originals -> {sum(x[4] for x in manifest)} tiles",flush=True)
    with (target/"source_manifest.csv").open("w",newline="",encoding="utf-8") as out:
        writer=csv.writer(out);writer.writerow(["original","source_species","split","original_boxes","tiles"])
        writer.writerows(manifest)
    (target/"data.yaml").write_text(
        f"path: {target.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: colony\n",encoding="utf-8")
    (target/"SOURCE_AND_LICENSE.txt").write_text(
        "ADBC Dataset, CC BY 4.0; https://doi.org/10.6084/m9.figshare.22022540.v3\n"
        "Makrai et al., Scientific Data (2023) https://doi.org/10.1038/s41597-023-02404-8\n"
        f"Derived: species classes collapsed to colony, tiled {args.tile}px stride {args.stride}px.\n"
        "Group held out by species; tiles of same original never cross splits.\n"
        "No proof of accuracy on ColonyVision user photos.\n",encoding="utf-8")
    for split in ("train","val","test"):
        print(split,"originals",sum(x[2]==split for x in manifest),
              "tiles",sum(x[4] for x in manifest if x[2]==split),flush=True)
    print("READY:",target/"data.yaml",flush=True)

if __name__=="__main__":main()
