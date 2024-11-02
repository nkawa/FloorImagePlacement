
import cv2

from tkinter import *
from PIL import Image, ImageTk, ImageOps  # 画像データ用
import sys
import matplotlib.pyplot as plt 
import tkinter as tk
from tkinter import filedialog
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import json
import random
import pathlib
import os.path as path
import pickle
from datetime import datetime
from glob import glob
from os import makedirs
from mask_editor import MaskEditor
from projection_editor import ProjectionEditor

from util.dscamera import DSCamera
from util.dsutil import mask_dir, empty_image_dir, pj_file, calibImage, _crop


# 元データが、3990x2312 画像で、縮小動画は 1280x742ピクセル
# 元のトラッキングの位置から、 3885.36, 812.703 シフトさせると、元の動画の位置になる。これを 1280x742 に変換
# 1280/3990 = 0.321303

# パレットの正解データも対応できるように！


BASE_X = 3885.36
BASE_Y = 812.703
SCALE = 0.321303

cscale = 1.2
cw = int(1280*cscale)
ch = int(742*cscale)

# 拡大表示のサイズ
kw = 300
kh = 300





cams = "A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,B1,B2,B3,B4,B5,B6,B7,B8".split(",") # 20 cameras




MASK_REG_EXP = lambda n: n + ".png"


def generate_random_color():
    r = [random.uniform(0, 1) for _ in range(3)]
    return (int(r[0]*255),int(r[1]*255),int(r[2]*255))



setq =-1
def get_seq_color():
    global setq
    cmap = plt.get_cmap("tab20") # ココがポイント(tableau20 の色を使う)
    setq = setq + 1
    print(cmap(setq%20))
    r,g,b,_ = cmap(setq%20) 
    return (int(r*255),int(g*255),int(b*255))




cam_colors = {cam: get_seq_color() for cam in cams}


# カメラの中心位置を保存するリスト
cam_locations = []


# アウトラインの画像、マスクを作る
def make_outline_image(mask_img,color=(0,255,0),name="non"):
    global cam_locations
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask_img.shape
    outline_image = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.drawContours(outline_image, contours, -1, color, 2)

    for contour in contours:
        print("contour",name)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cam_locations.append((cX,cY,name))
            cv2.putText(outline_image, name, (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    outline_mask = cv2.cvtColor(outline_image, cv2.COLOR_BGR2GRAY)

    return outline_image, outline_mask






def stitch(mask_dir: str, pj_file: str, src_dir: str,border, imgflag) -> np.ndarray:
    # load constants
    print("Gen masks")

    with open(pj_file) as f:
        pj_dict: dict[str, dict[str, int | list[list[float]]]] = json.load(f)

    # prepare constants
    cam_names = pj_dict.keys() 
    pjs, frm_size, _ = _crop({n: np.array(pj_dict[n]["projective_matrix"], dtype=np.float64) for n in cam_names})


#    print(pjs)
    warped_masks = {n: cv2.warpPerspective(cv2.imread(path.join(mask_dir, MASK_REG_EXP(n)), flags=cv2.IMREAD_GRAYSCALE), pjs[n], frm_size) for n in cam_names}

#　　ここで各マスクの枠も作りたい！(ボーダー表示の有無)
    if border:
       warped_masks_outline = {n: make_outline_image(warped_masks[n],cam_colors[n],n) for n in cam_names}

    # stitch
    status = {}
    stitched_frm = np.zeros((frm_size[1], frm_size[0], 3), dtype=np.uint8)
    for n in cam_names:
        print("Stitching",n)
#        image = cv2.imread(path.join(src_dir,"camera"+n+".jpg"))
        if imgflag:
            image = calibImage(n)
            cv2.copyTo(cv2.warpPerspective(image, pjs[n], frm_size), warped_masks[n], dst=stitched_frm)
        if border:
            outline, omask = warped_masks_outline[n]
            cv2.copyTo(outline, omask, dst=stitched_frm)

        # カメラの中心に文字入れたい。
#        print( (int(frm_size[0]/2), int(frm_size[1]/2)), n)
#        cv2.putText(stitched_frm, n, (int(frm_size[0]/2), int(frm_size[1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return stitched_frm



class App(tk.Frame):
    def __init__(self,master = None):
        super().__init__(master)
        self.workers = None
        self.current_id = -1
        self.pallet_id = -1
        self.image_frame = tk.Frame(self.master)
        self.button_frame = tk.Frame(self.master)
        self.track_info_frame = tk.Frame(self.master)
        self.json_sub = tk.Frame(self.button_frame)
        self.set_frame = tk.Frame(self.button_frame)
        self.slider_frame = tk.Frame(self.track_info_frame)
        self.button_sub = tk.Frame(self.button_frame)

        self.mask_editor = None
        self.prj_editor = None

        self.canvas = tk.Canvas(self.image_frame, width = cw, height = ch)
        
        self.canvas.pack(expand = True, fill = tk.BOTH, anchor=tk.CENTER, padx=10)

#        self.kakudai = tk.Canvas(self.button_frame, width = kw, height = kh)
#        self.kakudai.delete("all")
#        self.kakudai.pack(expand = True, fill=tk.X, anchor=tk.CENTER,padx=2,pady=2)
#        self.kakudai.create_rectangle(0, 0, kw, kh, fill="black", outline="")


        self.csv_button2 = tk.Button(self.json_sub, text="loadAll", command=self.loadImages, width=10)
        self.csv_button2.pack(expand = True, side=tk.LEFT,padx=10)
#        self.csv_button = tk.Button(self.json_sub, text="ID_JSON", command=self.loadImages, width=10)
#        self.csv_button.pack(expand = True, side=tk.LEFT,padx=10 )
        self.json_sub.pack(expand = True,  padx=10)

        self.check_frame = tk.Frame(self.button_frame)
        self.cborder = tk.BooleanVar(value=True)
        self.cimg = tk.BooleanVar(value=True)
        self.check_pallet = tk.Checkbutton(self.check_frame, text="border" ,width=10,variable = self.cborder)
        self.check_pallet.pack(side=tk.LEFT,padx=10)
        self.check_track = tk.Checkbutton(self.check_frame, text="image", width=10,variable = self.cimg)
        self.check_track.pack(side=tk.LEFT,padx=10)
        self.check_frame.pack(expand = True, fill = tk.X, padx=10, pady=10)

# IDを登録する仕組み！
        self.id_box = tk.Label(self.button_sub,text="Camera:  ")
        self.id_box.pack(side=tk.LEFT,padx=0,pady=10)
        self.subj_box = tk.Entry(self.button_sub,width=15)
        self.subj_box.insert(tk.END,"")
        self.subj_box.pack(side=tk.LEFT, padx=0,pady=10)
        self.button_sub.pack(expand = True, fill = tk.X, padx=10, pady=10)

        self.id_button = tk.Button(self.set_frame,text="change_mask", command=self.mask_dialog,width=10)
        self.id_button.pack(side=tk.LEFT,padx=10)

        self.cid_button = tk.Button(self.set_frame,text="Change Projection", command=self.projection,width=15)
        self.cid_button.pack(side=tk.LEFT,padx=10)
        self.set_frame.pack(padx=10, pady=10)

 #       self.track_frame = tk.Frame(self.button_frame)
 #       self.sid_button = tk.Button(self.track_frame,text="Search TrackID", command=self.loadImages,width=15)
 #       self.sid_button.pack(side=tk.LEFT,padx=10)

  #      self.cinfo_button = tk.Button(self.track_frame,text="Clear TrackInfo", command=self.loadImages,width=15)
  #      self.cinfo_button.pack(side=tk.LEFT, padx=10)
  #      self.track_frame.pack(padx=10, pady=10)


   #     self.save_frame = tk.Frame(self.button_frame)
   #     self.save_button = tk.Button(self.save_frame, text="Save PalJSON", command=self.loadImages)
   #     self.save_button.pack(side=tk.LEFT,padx=10)

    #    self.save_button2 = tk.Button(self.save_frame, text="Save JSON", command=self.loadImages)
    #    self.save_button2.pack(side=tk.LEFT,padx=10)
    #    self.save_frame.pack( padx=10, pady=10)

     #   self.pallet_id_frame = tk.Frame(self.button_frame)
     #   self.pallet_id_label = tk.Label(self.pallet_id_frame,text="PID: 0")
     ##   self.pallet_id_label.pack(side=tk.LEFT,padx=10)
      #  self.pallet_id_box = tk.Entry(self.pallet_id_frame,width=10)
      #  self.pallet_id_box.pack(side=tk.LEFT,padx=10)
      #  self.pallet_set = tk.Button(self.pallet_id_frame, text="set", command=self.loadImages)
      #  self.pallet_set.pack(side=tk.LEFT,padx=10)
      #  self.pallet_id_frame.pack(padx=10, pady=10)

  #      self.pal_edit_frame = tk.Frame(self.button_frame)
  #      self.pal_line = tk.Button(self.pal_edit_frame, text="Check_Line", command=self.loadImages)
  #      self.pal_line.pack(side=tk.LEFT,padx=10)
  #      self.pal_edit_frame.pack(padx=10, pady=10)


   #     self.frame_num = tk.Label(self.button_frame,text="<-Frame:_")
   #     self.frame_num.pack(expand=True, fill= tk.X, padx = 10, pady = 10)

#        self.frameVar = tk.IntVar()

#        self.track_info = tk.Canvas(self.track_info_frame, width = 1400, height = 10)
#        self.track_info.pack(expand = True, fill = tk.X, anchor=tk.CENTER, padx=10,pady=0)
#        self.track_info.create_rectangle(0, 0, 1400, 10, fill="black", outline="")
        
        self.image_frame.grid(column=0, rowspan=2)
        self.button_frame.grid(column=1, row=1)
        self.track_info_frame.grid(column=0,row=2, columnspan=2)
#        self.slider_frame.grid(column=0,row=2, columnspan=2)

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        

        self.canvas.bind("<ButtonPress-1>", self.check_id)

#        self.master.bind('<Configure>', change_size)

    def loadImages(self):
        self.oimg = stitch(mask_dir, pj_file,empty_image_dir , self.cborder.get(),self.cimg.get())
        cv_image = cv2.resize(self.oimg,dsize=(cw,ch))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(cv_image)
        self.pimg  = ImageTk.PhotoImage(image=self.pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(
                int(cw/2),       # 画像表示位置(Canvasの中心)
                int(ch/2),                   
                image=self.pimg  # 表示画像データ
                )

    def check_id(self, event):
        x = int(event.x / cscale)
        y = int(event.y / cscale)
        print(x,y)
        dist = 100000000
        mincam = ""
        for (x0,y0,cam) in cam_locations:
            xx = int((x0)*SCALE)
            yy = int((y0)*SCALE)
            d = (x-xx)**2 + (y-yy)**2
            if d < dist:
                dist = d
                mincam = cam
        print("camera",mincam, x, y)

        self.subj_box.delete(0,tk.END)
        self.subj_box.insert(tk.END,mincam)

    def mask_dialog(self):
        if self.mask_editor is None:
            self.mask_top = tk.Toplevel(self.master)
            self.mask_editor = MaskEditor(self.mask_top)
        self.mask_editor.set_cam(self.subj_box.get())
        self.mask_editor.set_mask(mask_dir+"/"+self.subj_box.get()+".png")
        self.mask_editor.set_image(empty_image_dir+"/camera"+self.subj_box.get()+".jpg")

    def projection(self):
        if self.subj_box.get() == "":
            return
        if self.prj_editor is None:
            self.pjr_top = tk.Toplevel(self.master)
            self.prj_editor = ProjectionEditor(self.pjr_top)

        self.prep_stitch()
        self.prj_editor.set_cam(self.subj_box.get(),self)

    def update_pjs(self, cam, pjs):
        self.pjs = pjs
#        print("Update PJS",cam,pjs)
        self.local_stitch(cam)

    def prep_stitch(self):
        with open(pj_file) as f:
            pj_dict: dict[str, dict[str, int | list[list[float]]]] = json.load(f)
        cam_names = pj_dict.keys() 
        self.pjs, self.frm_size , _ = _crop({n: np.array(pj_dict[n]["projective_matrix"], dtype=np.float64) for n in cam_names})
        self.warped_masks = {n: cv2.warpPerspective(cv2.imread(path.join(mask_dir, MASK_REG_EXP(n)), flags=cv2.IMREAD_GRAYSCALE), self.pjs[n], self.frm_size) for n in cam_names}
        self.warped_masks_outline = {n: make_outline_image(self.warped_masks[n],cam_colors[n],n) for n in cam_names}


# 特定のカメラだけ stitch する
    def local_stitch(self,cam):
        if self.cimg.get():
            image = calibImage(cam)
            cv2.copyTo(cv2.warpPerspective(image, self.pjs[cam], self.frm_size), self.warped_masks[cam], dst=self.oimg)
        self.warped_masks[cam] =cv2.warpPerspective(cv2.imread(path.join(mask_dir, MASK_REG_EXP(cam)), flags=cv2.IMREAD_GRAYSCALE), self.pjs[cam], self.frm_size)
        self.warped_masks_outline[cam] =make_outline_image(self.warped_masks[cam],cam_colors[cam],cam) 

        if self.cborder.get():
            outline, omask = self.warped_masks_outline[cam]
            cv2.copyTo(outline, omask, dst=self.oimg)

        cv_image = cv2.resize(self.oimg,dsize=(cw,ch))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(cv_image)
        self.pimg  = ImageTk.PhotoImage(image=self.pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(
                int(cw/2),       # 画像表示位置(Canvasの中心)
                int(ch/2),                   
                image=self.pimg  # 表示画像データ
        )




    
if __name__ == "__main__":
    root = tk.Tk()
    root.title("FloorImage Placement 2024-10-03") 
    app = App(master = root)
    app.mainloop()
