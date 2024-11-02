import os
import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps  # 画像データ用

scale = 0.45
width , height = int(1920 *scale), int(1080*scale)

from util.dscamera import DSCamera

from util.dsutil import *


class MaskEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.mask = None
        self.bsize = 10
        self.mask_path = ""
        self.mask_image = None
        self.image_path = ""
        self.mask_image_path = None
        self.create_widgets()
        self.pack()



    def create_widgets(self):


        self.mask_image = tk.Canvas(self,width=width, height=height)     
#        self.mask_image.pack()
        self.mask_image.bind("<Button-1>", self.mask_image_click)
        self.mask_image.bind("<Button-2>", self.change_bsize)
        self.mask_image.bind("<Button-3>", self.mask_image_click)
        self.mask_image.bind('<B1-Motion>', self.mask_image_click)
        self.mask_image.bind('<B3-Motion>', self.mask_image_click)


        self.image_canvas = tk.Canvas(self, width=width, height=height)   
#        self.image_canvas.pack()  
        self.image_canvas.bind("<Button-1>", self.mask_image_click)
        self.image_canvas.bind("<Button-3>", self.mask_image_click)
        self.image_canvas.bind('<B1-Motion>', self.mask_image_click)
        self.image_canvas.bind('<B3-Motion>', self.mask_image_click)
        self.dimage_canvas = tk.Canvas(self, width=width, height=height)   
#        self.dimage_canvas.pack()  


        self.button_frame = tk.Frame(self)

        self.mask_image_label = tk.Label(self.button_frame, text="mask:"+self.mask_path)
        self.mask_image_label.pack()

        self.undis_button = tk.Button(self.button_frame, text="Undistort", command=self.undistort)
        self.undis_button.pack(pady=10)

        self.undis_mask_button = tk.Button(self.button_frame, text="undis_mask", command=self.undis_mask)
        self.undis_mask_button.pack(pady=10)


        self.save_button = tk.Button(self.button_frame, text="Save", command=self.save)
        self.save_button.pack(pady=10)


#        self.save_button = tk.Button(self, text="Save", command=self.save)
#        self.save_button.pack()

        self.mask_image.grid(column=0, row=0)
        self.image_canvas.grid(column=1, row=0)
        self.dimage_canvas.grid(column=0, row=1)
        self.button_frame.grid(column=1, row=1)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)


    def set_cam(self, cam):
        self.cam =cam
        self.cam_calib = DSCamera(calib_dir+"/camera"+cam+".json")

    def save(self): 
        print("save"+self.cam)
        fname = "mask/"+self.cam+".png"
        if os.path.exists(fname):
            timestamp= os.path.getmtime(fname)
            mod_time_str = time.strftime("%m%d%H%M", time.localtime(timestamp))
            next_name = "mask/"+self.cam+"_"+mod_time_str+".png"
            while os.path.exists(next_name):
                next_name = next_name[:-4] + "_.png"
            os.rename(fname, next_name)
        cv2.imwrite("mask/"+self.cam+".png", self.or_mask)

    def set_mask(self, mask_path):
        self.mask_path = mask_path
        self.mask_image_label.config(text="mask:"+self.mask_path)
        self.or_mask = cv2.imread(mask_path)
        self.mask = cv2.resize(self.or_mask, (width, height))
        self.pil_image = Image.fromarray(self.mask)
        self.pimg  = ImageTk.PhotoImage(image=self.pil_image)
        self.mask_image_path = mask_path
        self.mask_image.create_image(0, 0, anchor=tk.NW, image=self.pimg)

    def set_image(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, (width, height))
        self.main_image = Image.fromarray(self.image)
        self.main_pimg  = ImageTk.PhotoImage(image=self.main_image)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.main_pimg)

    def undistort(self):
        self.dimg = calibImage(self.cam)
        self.dimg = cv2.cvtColor(self.dimg, cv2.COLOR_BGR2RGB)
        self.dimg = cv2.resize(self.dimg, (width, height))
        self.undis_image = Image.fromarray(self.dimg)
        self.undis_image  = ImageTk.PhotoImage(image=self.undis_image)
        self.dimage_canvas.create_image(0, 0, anchor=tk.NW, image=self.undis_image)

# mask した画像を表示
    def undis_mask(self):
#        self.nimg= np.zeros((height,width,3),dtype=np.uint8)
        self.nimg= np.full((height,width,3), (0,0,255), dtype=np.uint8)
        cv2.copyTo(self.dimg, self.mask, self.nimg)        
        self.main_image = Image.fromarray(self.nimg)
        self.main_pimg  = ImageTk.PhotoImage(image=self.main_image)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.main_pimg)

    def change_bsize(self, event):
        self.bsize = int(event.y/2)
        print("Change bsize", self.bsize)


    def mask_image_click(self, event):
#        print("mask_image_click", event.x, event.y)
        print(event.num, event.state)
        x = int(event.x / scale)
        y = int(event.y / scale)
        if event.num == 2:
            self.change_bsize(event)
            return
        if event.num == 3 or event.state == 1024:
            cv2.rectangle(self.or_mask, (x-self.bsize, y-self.bsize), (x+self.bsize, y+self.bsize), (0, 0, 0), cv2.FILLED, cv2.LINE_AA)	
        else:
            cv2.rectangle(self.or_mask, (x-self.bsize, y-self.bsize), (x+self.bsize, y+self.bsize), (255, 255, 255), cv2.FILLED	, cv2.LINE_AA)

        self.mask = cv2.resize(self.or_mask, (width, height))
        self.pil_image = Image.fromarray(self.mask)
        self.pimg  = ImageTk.PhotoImage(image=self.pil_image)
        self.mask_image.create_image(0, 0, anchor=tk.NW, image=self.pimg)
        self.undis_mask()


    def do_drag(self  ,event):
        print("do_drag", event.x, event.y)
        print(event)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("MaskEditor 2024-10-03") 
    app = MaskEditor(master = root)

    app.set_mask("mask/B8.png")
    app.set_cam("B8")
    app.set_image("img/cameraB8.jpg")
    app.mainloop()
