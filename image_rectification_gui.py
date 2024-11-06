import cv2
from util.dscamera import DSCamera

from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps  # 画像データ用
import json
import os
import numpy as np
import time

class CalibEditor(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.input_images_dir = './input_images'
        self.output_images_dir = './output_images'
        self.calib_files_dir = './calib_files/'

        filetype = [("jpg","*.jpg"), ("png","*.png")]
#        self.import_image_file_path = tk.filedialog.askopenfilename(filetypes = filetype, initialdir = self.input_images_dir)
#        self.img = cv2.imread(self.import_image_file_path)
        self.img = None
        self.import_image_file_path=""

        self.calib_file = "./calib_files/base_trusco.json"
        self.cam = DSCamera(self.calib_file)

        self.master.title("DoubleSphere Calibration Tool")
        self.master.geometry("1920x1080") 
        
        # frame
        self.image_frame = tk.Frame(self.master)
        self.scale_frame = tk.Frame(self.master)
        self.button_frame = tk.Frame(self.master)

        # canvsa
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.image_frame, width = self.canvas_width, height=self.canvas_height)
        self.canvas.bind('<Button-1>', self.canvas_click)
        self.canvas.pack(expand = True, fill = tk.BOTH, anchor=tk.CENTER, padx=10)

        # fx scale
        self.label_fx_scale = Label(self.scale_frame, text='fx')
        self.var_scale_fx = tk.DoubleVar(value = self.cam.fx)
        self.scale_fx = tk.Scale(
            self.scale_frame, 
            from_=400, to=800, 
            variable=self.var_scale_fx, 
            orient=tk.HORIZONTAL, 
            resolution=0.1,
            length = 500,
            command=self.action_scale_fx)
        self.label_fx_scale.grid(column=0, row=0, sticky=tk.EW)
        self.scale_fx.grid(column=1, row=0, sticky=tk.EW)
    
        # fy scale
        self.label_fy_scale = Label(self.scale_frame, text='fy')
        self.var_scale_fy = tk.DoubleVar(value = self.cam.fy)
        self.scale_fy = tk.Scale(
            self.scale_frame, 
            from_=400, to=800, 
            variable=self.var_scale_fy, 
            orient=tk.HORIZONTAL, 
            resolution=0.1,
            length = 500,
            command=self.action_scale_fy)
        self.label_fy_scale.grid(column=0, row=1, sticky=tk.EW)
        self.scale_fy.grid(column=1, row=1, sticky=tk.EW)

        # cx scale
        self.label_cx_scale = Label(self.scale_frame, text='cx')
        self.var_scale_cx = tk.DoubleVar(value = self.cam.cx)
        self.scale_cx = tk.Scale(
            self.scale_frame, 
            from_=500, to=1500, 
            variable=self.var_scale_cx, 
            orient=tk.HORIZONTAL, 
            resolution=0.1,
            length = 500,
            command=self.action_scale_cx)
        self.label_cx_scale.grid(column=0, row=2, sticky=tk.EW)
        self.scale_cx.grid(column=1, row=2, sticky=tk.EW)

        # cy scale
        self.label_cy_scale = Label(self.scale_frame, text='cy')
        self.var_scale_cy = tk.DoubleVar(value = self.cam.cy)
        self.scale_cy = tk.Scale(
            self.scale_frame, 
            from_=500, to=1500, 
            variable=self.var_scale_cy, 
            orient=tk.HORIZONTAL, 
            resolution=0.1,
            length = 500,
            command=self.action_scale_cy)
        self.label_cy_scale.grid(column=0, row=3, sticky=tk.EW)
        self.scale_cy.grid(column=1, row=3, sticky=tk.EW)

        # xi scale
        self.label_xi_scale = Label(self.scale_frame, text='xi')
        self.var_scale_xi = tk.DoubleVar(value = self.cam.xi)
        self.scale_xi = tk.Scale(
            self.scale_frame, 
            from_=-1, to=1, 
            variable=self.var_scale_xi, 
            orient=tk.HORIZONTAL, 
            resolution=0.01,
            length = 500,
            command=self.action_scale_xi)
        self.label_xi_scale.grid(column=0, row=4, sticky=tk.EW)
        self.scale_xi.grid(column=1, row=4, sticky=tk.EW)

        # alpha scale
        self.label_alpha_scale = Label(self.scale_frame, text='alpha')
        self.var_scale_alpha = tk.DoubleVar(value = self.cam.alpha)
        self.scale_alpha = tk.Scale(
            self.scale_frame, 
            from_=0, to=1, 
            variable=self.var_scale_alpha, 
            orient=tk.HORIZONTAL, 
            resolution=0.001,
            length = 500,
            command=self.action_scale_alpha)
        self.label_alpha_scale.grid(column=0, row=5, sticky=tk.EW)
        self.scale_alpha.grid(column=1, row=5, sticky=tk.EW)

        # f scale
        self.label_f_scale = Label(self.scale_frame, text='f')
        self.var_scale_f = tk.DoubleVar(value = 0.5)
        self.scale_f = tk.Scale(
            self.scale_frame, 
            from_=0, to=1, 
            variable=self.var_scale_f, 
            orient=tk.HORIZONTAL, 
            resolution=0.001,
            length = 500,
            command=self.action_scale_f)
        self.label_f_scale.grid(column=0, row=6, sticky=tk.EW)
        self.scale_f.grid(column=1, row=6, sticky=tk.EW)

        self.f = 0.5

        # image file label
        self.label_using_image_file = Label(self.button_frame, text="Image file : " + os.path.basename(self.import_image_file_path))
        self.label_using_image_file.pack(expand = True, fill = tk.X, padx=10, pady=10)

        # import image button
        self.import_json_button = tk.Button(self.button_frame, text="Import image file", command=self.import_image, width=40)
        self.import_json_button.pack(expand = True, fill = tk.X, padx=10, pady=10)

        # export image button
        self.export_json_button = tk.Button(self.button_frame, text="Export image file", command=self.export_image, width=40, bg='#2DBE60')
        self.export_json_button.pack(expand = True, fill = tk.X, padx=10, pady=10)

        # calib file label
        self.label_using_json_file = Label(self.button_frame, text="Calibration file : " + os.path.basename(self.calib_file))
        self.label_using_json_file.pack(expand = True, fill = tk.X, padx=10, pady=10)

        # import calibration file button
        self.import_json_button = tk.Button(self.button_frame, text="Import calibration file", command=self.import_json, width=40)
        self.import_json_button.pack(expand = True, fill = tk.X, padx=10, pady=10)

        # export calibration file button
        self.export_json_button = tk.Button(self.button_frame, text="Export calibration file", command=self.export_json, width=40, bg='#2DBE60')
        self.export_json_button.pack(expand = True, fill = tk.X, padx=10, pady=10)

        # save calibration file button
        self.save_json_button = tk.Button(self.button_frame, text="Svave file", command=self.save_calibfile, width=40, bg='#2DBE60')
        self.save_json_button.pack(expand = True, fill = tk.X, padx=10, pady=10)

        # reset param button
        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset_conf, width=40)
        self.reset_button.pack(expand = True, fill = tk.X, padx=10, pady=10)

        self.disp_id = None

        self.image_frame.grid(column=0, rowspan=2)
        self.scale_frame.grid(column=1, row=0)
        self.button_frame.grid(column=1, row=1)

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

        if self.disp_id is None:
            self.disp_image()
        else:
            self.after_cancel(self.disp_id)
            self.disp_id = None

    def action_scale_fx(self, event=None):
        self.cam.fx = self.scale_fx.get()
        self.disp_image()
    
    def action_scale_fy(self, event=None):
        self.cam.fy = self.scale_fy.get()
        self.disp_image()

    def action_scale_cx(self, event=None):
        self.cam.cx = self.scale_cx.get()
        self.disp_image()

    def action_scale_cy(self, event=None):
        self.cam.cy = self.scale_cy.get()
        self.disp_image()

    def action_scale_xi(self, event=None):
        self.cam.xi = self.scale_xi.get()
        self.disp_image()

    def action_scale_alpha(self, event=None):
        self.cam.alpha = self.scale_alpha.get()
        self.disp_image()

    def action_scale_f(self, event=None):
        self.f = self.scale_f.get()
        self.disp_image()
    
    def import_img_call(self, img_file_path):
        self.import_image_file_path = img_file_path
        self.img = cv2.imread(self.import_image_file_path)
        self.label_using_image_file["text"] = "Image file : " + os.path.basename(self.import_image_file_path)


    def import_image(self, event=None):
        filetype = [("jpg","*.jpg"), ("png","*.png")]
        self.import_image_file_path = tk.filedialog.askopenfilename(filetypes = filetype, initialdir = self.input_images_dir)
        self.img = cv2.imread(self.import_image_file_path)
        self.label_using_image_file["text"] = "Image file : " + os.path.basename(self.import_image_file_path)

    def export_image(self, event=None):
        filetype = [("jpg","*.jpg"), ("png","*.png")]
        self.export_image_file_path = tk.filedialog.asksaveasfilename(initialfile=os.path.basename(self.import_image_file_path), filetypes = filetype, initialdir = self.output_images_dir)
        self.pil_image.save(self.export_image_file_path)

    def reset_conf(self, event=None):
        self.cam = DSCamera(self.calib_file) 
        self.scale_fx.set(self.cam.fx)
        self.scale_fy.set(self.cam.fy)
        self.scale_cx.set(self.cam.cx)
        self.scale_cy.set(self.cam.cy)
        self.scale_xi.set(self.cam.xi)
        self.scale_alpha.set(self.cam.alpha)
        self.scale_f.set(0.5)

    def import_json_call(self, json_file_path):
        self.calib_file = json_file_path
        self.cam = DSCamera(self.calib_file) 
        self.scale_fx.set(self.cam.fx)
        self.scale_fy.set(self.cam.fy)
        self.scale_cx.set(self.cam.cx)
        self.scale_cy.set(self.cam.cy)
        self.scale_xi.set(self.cam.xi)
        self.scale_alpha.set(self.cam.alpha)
        self.label_using_json_file["text"] = "Calibration file : " + os.path.basename(self.calib_file)
        self.disp_image()

    def import_json(self, event=None):
        filetype = [("Json","*.json")]
        self.calib_file = tk.filedialog.askopenfilename(filetypes = filetype, initialdir = self.calib_files_dir)
        self.import_json_call(self.calib_file)

    def export_json(self, event=None):
        filetype = [("Json","*.json")]
        init_calib_file_name = os.path.splitext(os.path.basename(self.import_image_file_path))[0] + ".json"
        json_file = filedialog.asksaveasfilename(initialfile=init_calib_file_name, filetypes = filetype, initialdir = self.calib_files_dir)

        calib_data = {
            "value0": {
                "intrinsics": [
                    {
                        "camera_type": "ds",
                        "intrinsics": 
                            {
                            "fx": self.cam.fx,
                            "fy": self.cam.fy,
                            "cx": self.cam.cx,
                            "cy": self.cam.cy,
                            "xi": self.cam.xi,
                            "alpha": self.cam.alpha
                            }
                    }
                ],
                "resolution": [
                    [
                        self.cam.img_size[1],
                        self.cam.img_size[0]
                    ]
                ]
            }
        }
        with open(json_file, 'w') as f:
            json.dump(calib_data, f, indent=2)


    def prep_calib_dirs(self, cam, calib_folder,edit_master):
        self.edit_master = edit_master
        self.edit_cam = cam
        self.calib_folder = calib_folder
    
    def save_calibfile(self,event = None):

        calib_data = {
            "value0": {
                "intrinsics": [
                    {
                        "camera_type": "ds",
                        "intrinsics": 
                            {
                            "fx": self.cam.fx,
                            "fy": self.cam.fy,
                            "cx": self.cam.cx,
                            "cy": self.cam.cy,
                            "xi": self.cam.xi,
                            "alpha": self.cam.alpha
                            }
                    }
                ],
                "f": self.f,
                "resolution": [
                    [
                        self.cam.img_size[1],
                        self.cam.img_size[0]
                    ]
                ]
            }
        }
        fname = self.calib_folder+"/camera"+self.edit_cam+".json"
        if os.path.exists(fname):
            timestamp= os.path.getmtime(fname)
            mod_time_str = time.strftime("%m%d%H%M", time.localtime(timestamp))
            next_name = fname[:-5]+mod_time_str+".json"
            while os.path.exists(next_name):
                next_name = next_name[:-5] + "_.json"
            os.rename(fname, next_name)

        with open(fname, 'w') as f:
            json.dump(calib_data, f, indent=2)
        # ここで main に返す！
        self.edit_master.update_dscam(self.edit_cam)


    def canvas_click(self, event):
        if self.disp_id is None:
            self.disp_image()
        else:
            self.after_cancel(self.disp_id)
            self.disp_id = None


    def disp_image(self):
        if self.img is None:
            return
        cv_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.cam.f = self.f # for new dscamera
        perspective = self.cam.to_perspective(cv_image, img_size=(1080, 1920), f=self.f)
#        print(perspective.shape)
        self.pil_image = Image.fromarray(perspective)

        # Canvasへ表示
        # canvas_width = self.canvas.winfo_width()
        # canvas_height = self.canvas.winfo_height()        
        canvas_width = self.canvas_width
        canvas_height = self.canvas_height
        print(canvas_width, canvas_height)
        self.pil_image_for_canvas = ImageOps.pad(self.pil_image, (canvas_width, canvas_height))
        np_image_for_canvas = np.asarray(self.pil_image_for_canvas).copy()
        y_step = 100
        x_step = 100
        y_img, x_img = np_image_for_canvas.shape[:2]
        np_image_for_canvas[y_step:y_img:y_step, :, :] = [255, 0, 0]
        np_image_for_canvas[:, x_step:x_img:x_step, :] = [255, 0, 0]
        self.pil_image_for_canvas = Image.fromarray(np_image_for_canvas)

        self.photo_image_for_canvas = ImageTk.PhotoImage(image=self.pil_image_for_canvas)


        # 画像の描画
        self.canvas.delete("all")
        self.canvas.create_image(
                canvas_width / 2,       # 画像表示位置(Canvasの中心)
                canvas_height / 2,                   
                image=self.photo_image_for_canvas  # 表示画像データ
                )

        # disp_image()を10msec後に実行する
#        self.disp_id = self.after(10, self.disp_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = CalibEditor(master = root)
    app.mainloop()
