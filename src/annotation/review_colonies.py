"""Review WeCFU provisional colony markers as YOLO boxes; never auto-approve."""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "raw" / "wecfu_ready"
ARCHIVE = ROOT / "annotations" / "pseudo_labels" / "wecfu_ultra" / "wecfu_ultra_export.zip"
DRAFT = ROOT / "annotations" / "review_drafts"
REVIEWED = ROOT / "annotations" / "reviewed" / "colony"

def import_boxes(name):
    if not ARCHIVE.is_file():
        return []
    with zipfile.ZipFile(ARCHIVE) as z:
        key = "detections/" + name + ".json"
        if key not in z.namelist():
            return []
        marks = json.loads(z.read(key)).get("detections", [])
    result = []
    for m in marks:
        if m.get("accepted", True):
            x, y, r = [float(m[k]) for k in ("cx", "cy", "r")]
            if r > 0:
                result.append([x-r, y-r, x+r, y+r])
    return result

def norm_box(b, w, h):
    a,c = sorted((max(0,min(w,float(b[0]))), max(0,min(w,float(b[2])))))
    t,z = sorted((max(0,min(h,float(b[1]))), max(0,min(h,float(b[3])))))
    return [a,t,c,z] if c-a >= 2 and z-t >= 2 else None

def yolo_line(b, w, h):
    x0,y0,x1,y1=b
    return "0 %.8f %.8f %.8f %.8f" % ((x0+x1)/(2*w),(y0+y1)/(2*h),(x1-x0)/w,(y1-y0)/h)

def save_record(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)

class Reviewer:
    def __init__(self, images):
        self.files = images
        self.index = 0
        self.ui = tk.Tk()
        self.ui.title("ColonyVision - Human Annotation Review")
        self.ui.geometry("1300x940")
        self.mode = tk.StringVar(value="add")
        self.radius = tk.IntVar(value=25)
        self.info = tk.StringVar()
        self.status = tk.StringVar()
        self.zoom = 1.0
        self.dirty = False
        self.reviewed = False
        self.drag = None
        self.build()
        self.load()

    def build(self):
        bar=ttk.Frame(self.ui)
        bar.pack(fill="x",padx=5,pady=5)
        ttk.Button(bar,text="Previous",command=lambda:self.move_image(-1)).pack(side="left")
        ttk.Button(bar,text="Next",command=lambda:self.move_image(1)).pack(side="left")
        for m in ("add","move","resize","delete"):
            ttk.Radiobutton(bar,text=m.title(),variable=self.mode,value=m).pack(side="left",padx=3)
        ttk.Label(bar,text="New box radius").pack(side="left",padx=5)
        ttk.Spinbox(bar,from_=3,to=200,textvariable=self.radius,width=5).pack(side="left")
        ttk.Button(bar,text="Zoom -",command=lambda:self.set_zoom(.8)).pack(side="left",padx=6)
        ttk.Button(bar,text="Zoom +",command=lambda:self.set_zoom(1.25)).pack(side="left")
        ttk.Button(bar,text="Approve reviewed",command=self.approve).pack(side="right",padx=3)
        ttk.Button(bar,text="Save draft",command=self.save_draft).pack(side="right",padx=3)
        ttk.Label(self.ui,textvariable=self.info).pack(fill="x",padx=7)
        area=ttk.Frame(self.ui)
        area.pack(fill="both",expand=True)
        self.canvas=tk.Canvas(area,background="#171717",cursor="crosshair")
        sx=ttk.Scrollbar(area,orient="horizontal",command=self.canvas.xview)
        sy=ttk.Scrollbar(area,orient="vertical",command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=sx.set,yscrollcommand=sy.set)
        self.canvas.grid(row=0,column=0,sticky="nsew")
        sx.grid(row=1,column=0,sticky="ew")
        sy.grid(row=0,column=1,sticky="ns")
        area.columnconfigure(0,weight=1)
        area.rowconfigure(0,weight=1)
        ttk.Label(self.ui,textvariable=self.status).pack(fill="x",padx=7)
        ttk.Label(self.ui,text="Add: click or drag box | Move: drag | Resize: drag nearest corner | Right-click: delete | Ctrl+S: save draft | Ctrl+Enter: approve | Alt+arrows: navigate").pack(fill="x",padx=7,pady=5)
        self.canvas.bind("<ButtonPress-1>",self.press)
        self.canvas.bind("<B1-Motion>",self.motion)
        self.canvas.bind("<ButtonRelease-1>",self.release)
        self.canvas.bind("<ButtonPress-3>",self.delete)
        self.canvas.bind("<MouseWheel>",lambda e:self.set_zoom(1.25 if e.delta>0 else .8))
        self.ui.bind("<Control-s>",lambda e:self.save_draft())
        self.ui.bind("<Control-Return>",lambda e:self.approve())
        self.ui.bind("<Alt-Left>",lambda e:self.move_image(-1))
        self.ui.bind("<Alt-Right>",lambda e:self.move_image(1))
        self.ui.protocol("WM_DELETE_WINDOW",self.close)

    def path(self):
        return self.files[self.index]

    def record_path(self,folder):
        return folder / (self.path().name + ".json")

    def load(self):
        with Image.open(self.path()) as img:
            self.original=img.convert("RGB")
        self.w,self.h=self.original.size
        r=self.record_path(REVIEWED)
        d=self.record_path(DRAFT)
        self.reviewed=False
        self.origin="WeCFU provisional markers"
        if d.exists() and (not r.exists() or d.stat().st_mtime >= r.stat().st_mtime):
            self.boxes=json.loads(d.read_text(encoding="utf-8"))["boxes"]
            self.origin="Saved draft"
        elif r.exists():
            data=json.loads(r.read_text(encoding="utf-8"))
            self.boxes=data["boxes"]
            self.reviewed=data.get("status")=="reviewed"
            self.origin="Approved review"
        else:
            self.boxes=import_boxes(self.path().name)
        self.boxes=[n for b in self.boxes if (n:=norm_box(b,self.w,self.h)) is not None]
        self.zoom=1.0
        self.base=min(1150/self.w,790/self.h,1)
        self.dirty=False
        self.draw()

    def draw(self):
        self.scale=self.base*self.zoom
        sw,sh=max(1,round(self.w*self.scale)),max(1,round(self.h*self.scale))
        image=self.original.resize((sw,sh),Image.Resampling.LANCZOS)
        self.tkimage=ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0,0,anchor="nw",image=self.tkimage)
        self.canvas.configure(scrollregion=(0,0,sw,sh))
        self.draw_boxes()

    def draw_boxes(self):
        self.canvas.delete("annotation")
        color="#6bea79" if self.reviewed and not self.dirty else "#ffd45a"
        for b in self.boxes:
            self.canvas.create_rectangle(*[v*self.scale for v in b],outline=color,width=2,tags="annotation")
        label="APPROVED" if self.reviewed and not self.dirty else "UNVERIFIED"
        self.info.set("[%d/%d] %s | %dx%d | %d boxes | %s | %s" % (
            self.index+1,len(self.files),self.path().name,self.w,self.h,len(self.boxes),label,self.origin))
        self.status.set("Yellow = unverified boxes; only approve after checking missed/extra colonies and box sizes.")

    def pos(self,event):
        return (max(0,min(self.w,self.canvas.canvasx(event.x)/self.scale)),
                max(0,min(self.h,self.canvas.canvasy(event.y)/self.scale)))

    def find(self,x,y):
        hits=[(i,(b[2]-b[0])*(b[3]-b[1])) for i,b in enumerate(self.boxes)
              if b[0]<=x<=b[2] and b[1]<=y<=b[3]]
        return min(hits,key=lambda item:item[1])[0] if hits else None

    def press(self,event):
        mode=self.mode.get()
        if mode=="delete":
            self.delete(event)
            return
        x,y=self.pos(event)
        index=self.find(x,y)
        if mode!="add" and index is None:return
        old=self.boxes[index][:] if index is not None else None
        corner=None
        if mode=="resize":
            corners=[(old[0],old[1]),(old[2],old[1]),(old[2],old[3]),(old[0],old[3])]
            corner=min(range(4),key=lambda j:(x-corners[j][0])**2+(y-corners[j][1])**2)
        self.drag=(mode,index,x,y,old,corner)

    def motion(self,event):
        if self.drag is None:return
        mode,index,startx,starty,old,corner=self.drag
        x,y=self.pos(event)
        if mode=="move":
            dx,dy=x-startx,y-starty
            box=norm_box([old[0]+dx,old[1]+dy,old[2]+dx,old[3]+dy],self.w,self.h)
        elif mode=="resize":
            x0,y0,x1,y1=old
            if corner in (0,3):x0=x
            else:x1=x
            if corner in (0,1):y0=y
            else:y1=y
            box=norm_box([x0,y0,x1,y1],self.w,self.h)
        else:return
        if box is not None:
            self.boxes[index]=box
            self.dirty=True
            self.reviewed=False
            self.draw_boxes()

    def release(self,event):
        if self.drag is None:return
        mode,index,x,y,old,corner=self.drag
        self.drag=None
        if mode!="add":return
        x1,y1=self.pos(event)
        if max(abs(x-x1),abs(y-y1))<4:
            r=max(3,self.radius.get())
            box=[x-r,y-r,x+r,y+r]
        else:box=[x,y,x1,y1]
        b=norm_box(box,self.w,self.h)
        if b:
            self.boxes.append(b)
            self.dirty=True
            self.reviewed=False
            self.draw_boxes()

    def delete(self,event):
        x,y=self.pos(event)
        index=self.find(x,y)
        if index is not None:
            self.boxes.pop(index)
            self.dirty=True
            self.reviewed=False
            self.draw_boxes()

    def set_zoom(self,factor):
        self.zoom=max(.5,min(6,self.zoom*factor))
        self.draw()

    def data(self,status):
        file=self.path()
        return {"status":status,"image":file.name,"source_path":file.relative_to(ROOT).as_posix(),
                "sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                "width":self.w,"height":self.h,"boxes":self.boxes,"box_count":len(self.boxes),
                "class_name":"colony","saved_at_utc":datetime.now(timezone.utc).isoformat(),
                "note":"Human-reviewed boxes" if status=="reviewed" else "Draft only: WeCFU geometry not verified"}

    def save_draft(self):
        save_record(self.record_path(DRAFT),self.data("draft"))
        self.dirty=False
        self.reviewed=False
        self.draw_boxes()
        self.status.set("Draft saved; not eligible for training.")

    def approve(self):
        if not messagebox.askyesno("Approve labels",
           "Have you reviewed all colony locations and adjusted the rectangles?\n"
           + self.path().name + ": " + str(len(self.boxes)) + " colonies.\n"
           "This makes the file eligible for training."):
            return
        save_record(self.record_path(REVIEWED),self.data("reviewed"))
        REVIEWED.mkdir(parents=True,exist_ok=True)
        target=REVIEWED/(self.path().name+".txt")
        lines=[yolo_line(b,self.w,self.h) for b in self.boxes]
        target.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")
        self.record_path(DRAFT).unlink(missing_ok=True)
        self.dirty=False
        self.reviewed=True
        self.draw_boxes()
        self.status.set("Verified label saved: "+str(target))

    def confirm(self):
        if not self.dirty:return True
        answer=messagebox.askyesnocancel("Unsaved changes","Save unverified draft? Yes=save, No=discard, Cancel=stay")
        if answer is None:return False
        if answer:self.save_draft()
        return True

    def move_image(self,direction):
        if self.confirm():
            self.index=(self.index+direction)%len(self.files)
            self.load()

    def close(self):
        if self.confirm():self.ui.destroy()

    def run(self):
        self.ui.mainloop()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images",type=Path,default=IMAGES)
    ap.add_argument("--audit",action="store_true",help="Inspect imported candidates without GUI")
    args=ap.parse_args()
    files=sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png"))
    if not files:raise SystemExit("No images: "+str(args.images))
    if args.audit:
        counts=[len(import_boxes(p.name)) for p in files]
        print("Images:",len(files),"Seed boxes:",sum(counts),"Zero-box images:",counts.count(0))
        return
    Reviewer(files).run()

if __name__=="__main__":
    main()
