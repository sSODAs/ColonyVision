"""Synthetic checks; real photographs remain unreviewed."""
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"/"annotation"))
import review_colonies as rc
import build_reviewed_dataset as bd

class TestAnnotation(unittest.TestCase):
    def test_geometry(self):
        self.assertEqual(rc.norm_box([-2,-3,20,25],100,100),[0,0,20,25])
        self.assertEqual(rc.yolo_line([10,10,30,30],100,100),
                         "0 0.20000000 0.20000000 0.20000000 0.20000000")

    def test_reviewed_only_and_plate_groups(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            images=base/"source"; images.mkdir()
            reviews=base/"reviewed"; reviews.mkdir()
            plan=base/"plan.csv"
            destination=base/"dataset"
            drafts=base/"drafts"; drafts.mkdir()
            old=(bd.REVIEWED,bd.DRAFT,bd.IMAGES,bd.PLAN,bd.DEST)
            bd.REVIEWED,bd.DRAFT,bd.IMAGES,bd.PLAN,bd.DEST=reviews,drafts,images,plan,destination
            try:
                rows=[]
                for name,split,group in (("a.jpg","train","dish_a"),
                                         ("b.jpg","val","dish_b"),
                                         ("c.jpg","test","dish_c")):
                    image=images/name
                    Image.new("RGB",(100,100),"white").save(image)
                    boxes=[[10,15,30,35]]
                    data={"status":"reviewed","image":name,
                          "sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
                          "width":100,"height":100,"boxes":boxes,"box_count":1}
                    (reviews/(name+".json")).write_text(json.dumps(data),encoding="utf-8")
                    (reviews/(name+".txt")).write_text(rc.yolo_line(boxes[0],100,100)+"\n",encoding="utf-8")
                    rows.append([name,split,group,1,""])
                (reviews/"unapproved.jpg.json").write_text(json.dumps({"status":"draft"}),encoding="utf-8")
                items=bd.reviewed_items()
                self.assertEqual(len(items),3)
                def write_plan():
                    with plan.open("w",newline="",encoding="utf-8") as f:
                        writer=csv.writer(f)
                        writer.writerow(bd.FIELDS)
                        writer.writerows(rows)
                write_plan()
                bd.build(items)
                self.assertTrue((destination/"images"/"train"/"a.jpg").is_file())
                self.assertTrue((destination/"labels"/"val"/"b.txt").is_file())
                self.assertFalse((destination/"images"/"train"/"unapproved.jpg").exists())
                rows[1][2]="dish_a"
                write_plan()
                with self.assertRaisesRegex(ValueError,"Group leakage"):
                    bd.build(items)
                (drafts/"a.jpg.json").write_text('{"status":"draft"}',encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"newer unsanctioned draft"):
                    bd.reviewed_items()
            finally:
                bd.REVIEWED,bd.DRAFT,bd.IMAGES,bd.PLAN,bd.DEST=old

if __name__=="__main__":
    unittest.main()
