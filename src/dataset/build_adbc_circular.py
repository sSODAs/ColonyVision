"""Experimental ADBC train-only Circular ROI masking.

Never modifies the existing ADBC baseline training images/labels.
Validation and test use EXACTLY the same unmasked data as the baseline.
Hough uses only pixel data; official GT used only to reject unsafe masking of
an individual *training* photo (never to estimate a circle).
"""
import argparse,csv,math,sys,zipfile
from pathlib import Path
from PIL import Image,ImageOps
from prepare_adbc import make_tiles,get_boxes,ROOT,CACHE
sys.path.insert(0,str(ROOT/"src"))
from preprocessing.plate_hough import estimate_plate_hough

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",default="adbc_full")
    ap.add_argument("--name",default="adbc_circular")
    ap.add_argument("--tile",type=int,default=768)
    ap.add_argument("--stride",type=int,default=640)
    ap.add_argument("--limit",type=int,default=0,help="0=all 260 training originals")
    ap.add_argument("--safety-factor",type=float,default=1.035)
    args=ap.parse_args()
    if not 1.0<=args.safety_factor<=1.08:
        raise SystemExit("safety-factor must be 1.0..1.08")
    base=ROOT/"dataset"/args.baseline
    manifest=base/"source_manifest.csv"
    if not manifest.exists():raise SystemExit("Missing original split manifest")
    with manifest.open(encoding="utf-8-sig",newline="") as f:
        source=[r for r in csv.DictReader(f) if r["split"]=="train"]
    if args.limit>0:source=source[:args.limit]
    out=ROOT/"dataset"/args.name
    if out.resolve()==base.resolve():raise SystemExit("Refusing to modify baseline dataset")
    if (out/"images"/"train").exists() or (out/"labels"/"train").exists():
        raise SystemExit("Output exists; delete or choose another --name to avoid stale mixed tiles")
    outcomes=[]
    with zipfile.ZipFile(CACHE/"annot_YOLO.zip") as z:
        for index,row in enumerate(source,1):
            name=row["original"]
            source_path=CACHE/"images"/name
            with Image.open(source_path) as original:
                image=ImageOps.exif_transpose(original).convert("RGB")
            geometry=estimate_plate_hough(image)
            gt=get_boxes(z,name,*image.size)
            action="masked"
            circle=None
            if geometry is None:
                action="unmasked_roi_not_found"
            else:
                cx,cy,r,_=geometry
                radius=r*args.safety_factor
                # A box is safe only if all its corners fit inside the mask.
                # GT is NOT used to fit the circle, change validation, or test.
                outside=0
                for x0,y0,x1,y1 in gt:
                    if any(math.hypot(x-cx,y-cy)>radius
                           for x,y in ((x0,y0),(x0,y1),(x1,y0),(x1,y1))):
                        outside+=1
                if outside:
                    action=f"unmasked_{outside}_gt_boxes_near_border"
                else:
                    circle=(cx,cy,radius)
            _,tiles=make_tiles(source_path,z,"train",out,args.tile,args.stride,roi_circle=circle)
            outcomes.append([name,action,len(gt),tiles])
            if index%10==0 or index==len(source):
                print(f"Prepared {index}/{len(source)} originals; masked {sum(v[1]=='masked' for v in outcomes)}",flush=True)
    with (out/"roi_audit.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f);writer.writerow(["original","roi_action","official_boxes","tiles"]);writer.writerows(outcomes)
    # Fixed original ADBC validation/test means direct comparisons remain possible.
    (out/"data.yaml").write_text(
        "path: "+out.as_posix()+"\n"
        "train: images/train\n"
        "val: "+(base/"images"/"val").as_posix()+"\n"
        "test: "+(base/"images"/"test").as_posix()+"\n"
        "names:\n  0: colony\n",encoding="utf-8")
    (out/"SOURCE_AND_LICENSE.txt").write_text(
        "ADBC CC BY 4.0 (Figshare) https://doi.org/10.6084/m9.figshare.22022540.v3\n"
        "Experimental image-only Hough circular mask on safe training images.\n"
        f"Outside circle set to black, radius includes safety factor {args.safety_factor}.\n"
        "Training images with uncertain ROI or official boxes crossing edge remain unmasked.\n"
        "Validation and test are unchanged from ADBC baseline for fair comparisons.\n"
        "Do not infer performance on user's own images without evaluation.\n",
        encoding="utf-8")
    print("Result:",out/"data.yaml","masked:",sum(v[1]=='masked' for v in outcomes),
          "unmasked:",sum(v[1]!='masked' for v in outcomes),flush=True)

if __name__=="__main__":main()
