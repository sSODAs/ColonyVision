"""Image-only Petri dish outer-circle hypothesis using OpenCV HoughCircles.

This is a geometric *estimate*; caller must verify it before deleting labels
or excluding colonies at the rim. Pixels and boxes use EXIF-upright coordinates.
"""
import cv2
import numpy as np
from PIL import Image

def estimate_plate_hough(image, canvas_size=400):
    src=image.convert("RGB")
    w,h=src.size
    scale=min(canvas_size/w,canvas_size/h)
    tw=max(1,round(w*scale));th=max(1,round(h*scale))
    arr=np.asarray(src.resize((tw,th),Image.Resampling.BILINEAR),dtype=np.uint8)
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    # Pad to square *without stretching* non-square camera photographs.
    bg=int(np.median(np.concatenate((gray[0],gray[-1],gray[:,0],gray[:,-1]))))
    canvas=np.full((canvas_size,canvas_size),bg,dtype=np.uint8)
    ox=(canvas_size-tw)//2;oy=(canvas_size-th)//2
    canvas[oy:oy+th,ox:ox+tw]=gray
    blurred=cv2.GaussianBlur(canvas,(9,9),2)
    proposals=[]
    for threshold in (65,45,32):
        circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,
               dp=1.2,minDist=canvas_size*0.38,param1=110,
               param2=threshold,minRadius=round(canvas_size*0.27),
               maxRadius=round(canvas_size*0.58))
        if circles is None:continue
        for x,y,r in circles[0]:
            # Prior: dish near picture center, bounded and approximately circular.
            if abs(x-canvas_size/2)>canvas_size*.24 or abs(y-canvas_size/2)>canvas_size*.24:
                continue
            if min(x-r,y-r)<-canvas_size*.07 or max(x+r,y+r)>canvas_size*1.07:
                continue
            score=(abs(x-canvas_size/2)+abs(y-canvas_size/2))/canvas_size + abs(r-canvas_size*.44)/canvas_size
            proposals.append((score,float(x),float(y),float(r),threshold))
        if proposals:break
    if not proposals:return None
    _,cx,cy,r,threshold=min(proposals)
    center_x=(cx-ox)/scale
    center_y=(cy-oy)/scale
    radius=r/scale
    return center_x,center_y,radius,threshold
