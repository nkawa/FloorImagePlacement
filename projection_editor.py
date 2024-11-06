import os
import time
import json
import numpy as np
import tkinter as tk
import cv2

# プロジェクション Matrix を編集

from util.dsutil import mask_dir, empty_image_dir, pj_file, calibImage,_crop


class ProjectionEditor(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.step = tk.IntVar()
        self.step.set(2)
        self.trape = tk.IntVar() # 台形の値
        self.trape.set(2)
        self.zoom = tk.DoubleVar()
        self.zoom.set(1.1)
        self.turn = tk.DoubleVar()
        self.turn.set(0.1)
        self.create_widgets()
        self.pack()

    def change_step(self,event):
        pass
    
    def create_widgets(self):

        self.tslider = tk.Scale(self, 
                               variable=self.turn,
                               command=self.change_step,
                               resolution=0.05,
                               from_=-5, to=5,length=150, orient=tk.HORIZONTAL)
        self.tslider.pack(pady=5)

        self.bt_frame = tk.Frame(self)
        self.left_button = tk.Button(self.bt_frame, text="Turn LEFT", command=self.turn_left, width=10)
        self.left_button.pack( side=tk.LEFT,padx=10)

        self.right_button = tk.Button(self.bt_frame, text="Turn RIGHT", command=self.turn_right, width=10)
        self.right_button.pack( side=tk.LEFT,padx = 10,pady=10)
        self.bt_frame.pack()



        self.slider = tk.Scale(self, 
                               variable=self.step,
                               command=self.change_step,
                               from_=1, to=150,length=150, orient=tk.HORIZONTAL)
        self.slider.pack(pady=5)



        self.up_button = tk.Button(self, text="UP", command=self.up, width=10)
        self.up_button.pack(pady=5)

        self.b_frame = tk.Frame(self)
        self.left_button = tk.Button(self.b_frame, text="LEFT", command=self.left, width=10)
        self.left_button.pack( side=tk.LEFT,padx=10)

        self.right_button = tk.Button(self.b_frame, text="RIGHT", command=self.right, width=10)
        self.right_button.pack( side=tk.LEFT,padx = 10,pady=5)
        self.b_frame.pack()

        self.down_button = tk.Button(self, text="DOWN", command=self.down, width=10)
        self.down_button.pack(pady=5)

        self.zslider = tk.Scale(self, 
                               variable=self.zoom,
                               command=self.change_step,
                               resolution=0.01,
                               from_=1.01, to=1.2,length=150, orient=tk.HORIZONTAL)
        self.zslider.pack(pady=5)

        self.zup_button = tk.Button(self, text="ZoomUp", command=self.zoom_up, width=10)
        self.zup_button.pack(pady=5)

        self.z_frame = tk.Frame(self)

        self.zlr_button = tk.Button(self.z_frame, text="LR_zoom", command=self.lr_zoom, width=10)
        self.zlr_button.pack(side=tk.LEFT,padx=10)

        self.zud_button = tk.Button(self.z_frame, text="UD_zoom", command=self.ud_zoom, width=10)
        self.zud_button.pack(side=tk.LEFT,padx=10)
        self.z_frame.pack()

        self.zdown_button = tk.Button(self, text="ZoomDown", command=self.zoom_down, width=10)
        self.zdown_button.pack(pady=5)


        self.tslider = tk.Scale(self, 
                               variable=self.trape,
                               command=self.change_step,
                               resolution=1,
                               from_=1, to=100,length=150, orient=tk.HORIZONTAL)
        self.tslider.pack(pady=5)
        self.t_frame = tk.Frame(self)
        self.trape_button = tk.Button(self.t_frame, text="Tr UX+", command=self.trapezoid_UX_Plus, width=5)
        self.trape_button.pack( side=tk.LEFT,padx=10)
        self.trape2_button = tk.Button(self.t_frame, text="Tr UX-", command=self.trapezoid_UX_Minus, width=5)
        self.trape2_button.pack( side=tk.LEFT,padx=10)
        
        self.trape3_button = tk.Button(self.t_frame, text="Tr DX+", command=self.trapezoid_X_Plus, width=5)
        self.trape3_button.pack( side=tk.LEFT,padx=10)
        self.trape4_button = tk.Button(self.t_frame, text="Tr DX-", command=self.trapezoid_X_Minus, width=5)
        self.trape4_button.pack( side=tk.LEFT,padx=10)
        self.t_frame.pack()

        self.t_frame2 = tk.Frame(self)
        self.trapey_button = tk.Button(self.t_frame2, text="Tr LY+", command=self.trapezoid_LY_Plus, width=5)
        self.trapey_button.pack( side=tk.LEFT,padx=10)
        self.trapey2_button = tk.Button(self.t_frame2, text="Tr LY-", command=self.trapezoid_LY_Minus, width=5)
        self.trapey2_button.pack( side=tk.LEFT,padx=10)
        self.trapey3_button = tk.Button(self.t_frame2, text="Tr RY+", command=self.trapezoid_Y_Plus, width=5)
        self.trapey3_button.pack( side=tk.LEFT,padx=10)
        self.trapey4_button = tk.Button(self.t_frame2, text="Tr RY-", command=self.trapezoid_Y_Minus, width=5)
        self.trapey4_button.pack( side=tk.LEFT,padx=10)
        self.t_frame2.pack()


        self.save_button = tk.Button(self, text="Save", command=self.save, width=10)
        self.save_button.pack(pady=10)

    def set_cam(self,cam,main_window):
        self.main = main_window
        self.cam = cam
        with open(pj_file) as f:
            self.pj_dict: dict[str, dict[str, int | list[list[float]]]] = json.load(f)

        cam_names = self.pj_dict.keys() 
        self.pjs, self.frame_size, self.org_diff = _crop({n: np.array(self.pj_dict[n]["projective_matrix"], dtype=np.float64) for n in cam_names})
#        print("SetCam",cam, self.pjs[cam])
 
    def up(self):
        move_mat = np.array(((1,0,0), (0,1, -self.step.get()), (0, 0, 1)), dtype=np.float64)
        self.pjs[self.cam] = np.dot(move_mat,self.pjs[self.cam])
        self.main.update_pjs(self.cam,self.pjs)


    def down(self):
        move_mat = np.array(((1,0,0), (0,1, self.step.get()), (0, 0, 1)), dtype=np.float64)
        self.pjs[self.cam] = np.dot(move_mat,self.pjs[self.cam])
        self.main.update_pjs(self.cam,self.pjs)

    def left(self):
        move_mat = np.array(((1,0,-self.step.get()), (0,1, 0), (0, 0, 1)), dtype=np.float64)
        self.pjs[self.cam] = np.dot(move_mat,self.pjs[self.cam])
        self.main.update_pjs(self.cam,self.pjs)

    def right(self):
        move_mat = np.array(((1,0,self.step.get()), (0,1, 0), (0, 0, 1)), dtype=np.float64)
        self.pjs[self.cam] = np.dot(move_mat,self.pjs[self.cam])
        self.main.update_pjs(self.cam,self.pjs)
    
    def turn_left(self):
        theta = -self.turn.get()*np.pi/180
        turn_mat = np.array(((np.cos(theta), -np.sin(theta), 0), (np.sin(theta), np.cos(theta), 0), (0, 0, 1)), dtype=np.float64)
        move_mat = np.array(((1,0, 1920/2), (0,1, 1080/2), (0, 0, 1)), dtype=np.float64)
        back_mat = np.array(((1,0, -1920/2), (0,1, -1080/2), (0, 0, 1)), dtype=np.float64)
        tmp_mat = np.dot(turn_mat,back_mat)
        tmp_mat = np.dot(move_mat,tmp_mat)
        print("Tmp",tmp_mat)
#        tmp_mat = np.dot(move_mat,back_mat)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],tmp_mat)
        self.main.update_pjs(self.cam,self.pjs)

    def turn_right(self):
        theta = self.turn.get()*np.pi/180
        turn_mat = np.array(((np.cos(theta), -np.sin(theta), 0), (np.sin(theta), np.cos(theta), 0), (0, 0, 1)), dtype=np.float64)
        move_mat = np.array(((1,0, 1920/2), (0,1, 1080/2), (0, 0, 1)), dtype=np.float64)
        back_mat = np.array(((1,0, -1920/2), (0,1, -1080/2), (0, 0, 1)), dtype=np.float64)
        tmp_mat = np.dot(turn_mat,back_mat)
        tmp_mat = np.dot(move_mat,tmp_mat)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],tmp_mat)
        self.main.update_pjs(self.cam,self.pjs)

    def zoom_up(self):
        zoom = self.zoom.get()
#        move_mat = np.array(((1,0, -self.frame_size[0]/2), (0,1, -self.frame_size[1]/2), (0, 0, 1)), dtype=np.float64)
#        back_mat = np.array(((1,0, self.frame_size[0]*zoom/2), (0,1, self.frame_size[1]*zoom/2), (0, 0, 1)), dtype=np.float64)
        diffx = 1920*zoom - 1920
        diffy = 1080*zoom - 1080
        move_mat = np.array(((1,0,-diffx/2), (0,1,-diffy/2), (0, 0, 1)), dtype=np.float64)
 #       back_mat = np.array(((1,0, -1920/2), (0,1, -1080/2), (0, 0, 1)), dtype=np.float64)
        zoom_mat = np.array(((zoom, 0, 0), (0, zoom, 0), (0, 0, 1)), dtype=np.float64)
#        self.pjs[self.cam] = np.dot(self.pjs[self.cam],np.dot(move_mat, np.dot(zoom_mat,back_mat)))
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],np.dot(move_mat,zoom_mat))
        self.main.update_pjs(self.cam,self.pjs)

    def zoom_down(self):
        zoom = self.zoom.get()
        diffx = 1920/zoom - 1920
        diffy = 1080/zoom - 1080
        move_mat = np.array(((1,0,-diffx/2), (0,1,-diffy/2), (0, 0, 1)), dtype=np.float64)
        zoom_mat = np.array(((1/zoom, 0, 0), (0, 1/zoom, 0), (0, 0, 1)), dtype=np.float64)
#        move_mat = np.array(((1,0, 1920/2), (0,1, 1080/2), (0, 0, 1)), dtype=np.float64)
#        back_mat = np.array(((1,0, -1920/zoom/2), (0,1, -1080/zoom/2), (0, 0, 1)), dtype=np.float64)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],np.dot(move_mat,zoom_mat))
        self.main.update_pjs(self.cam,self.pjs)


    def lr_zoom(self):
        zoom = self.zoom.get()
        diffx = 1920*zoom - 1920
        diffy = 0
        move_mat = np.array(((1,0,-diffx/2), (0,1,-diffy/2), (0, 0, 1)), dtype=np.float64)
        zoom_mat = np.array(((zoom, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],np.dot(move_mat,zoom_mat))
        self.main.update_pjs(self.cam,self.pjs)

    def ud_zoom(self):
        zoom = self.zoom.get()
        diffx = 0
        diffy = 1080*zoom - 1080
        move_mat = np.array(((1,0,-diffx/2), (0,1,-diffy/2), (0, 0, 1)), dtype=np.float64)
        zoom_mat = np.array(((1, 0, 0), (0, zoom,0), (0, 0, 1)), dtype=np.float64)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],np.dot(move_mat,zoom_mat))
        self.main.update_pjs(self.cam,self.pjs)


    def trapezoid_X_Plus(self):
        trape = self.trape.get()
        base_pos = np.array([[0,0],[1920,0],[1920,1080],[0,1080]],dtype=np.float32)
        trans_pos =np.array([[0,0],[1920,0],[1920+trape,1080],[0-trape,1080]], dtype=np.float32)
        trape_mat = cv2.getPerspectiveTransform(base_pos, trans_pos)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],trape_mat)
        self.main.update_pjs(self.cam,self.pjs)

    def trapezoid_X_Minus(self):
        trape = self.trape.get()
        base_pos = np.array([[0,0],[1920,0],[1920,1080],[0,1080]],dtype=np.float32)
        trans_pos =np.array([[0,0],[1920,0],[1920-trape,1080],[0+trape,1080]], dtype=np.float32)
        trape_mat = cv2.getPerspectiveTransform(base_pos, trans_pos)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],trape_mat)
        self.main.update_pjs(self.cam,self.pjs)

    def trapezoid_UX_Plus(self):
        trape = self.trape.get()
        base_pos = np.array([[0,0],[1920,0],[1920,1080],[0,1080]],dtype=np.float32)
        trans_pos =np.array([[0-trape,0],[1920+trape,0],[1920,1080],[0,1080]], dtype=np.float32)
        trape_mat = cv2.getPerspectiveTransform(base_pos, trans_pos)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],trape_mat)
        self.main.update_pjs(self.cam,self.pjs)

    def trapezoid_UX_Minus(self):
        trape = self.trape.get()
        base_pos = np.array([[0,0],[1920,0],[1920,1080],[0,1080]],dtype=np.float32)
        trans_pos = np.array([[0+trape,0],[1920-trape,0],[1920,1080],[0,1080]], dtype=np.float32)
        trape_mat = cv2.getPerspectiveTransform(base_pos, trans_pos)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],trape_mat)
        self.main.update_pjs(self.cam,self.pjs)


    def trapezoid_Y_Plus(self):
        trape = self.trape.get()
        base_pos = np.array([[0,0],[1920,0],[1920,1080],[0,1080]],dtype=np.float32)
        trans_pos =np.array([[0,0],[1920,-trape],[1920,1080+trape],[0,1080]], dtype=np.float32)
        trape_mat = cv2.getPerspectiveTransform(base_pos, trans_pos)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],trape_mat)
        self.main.update_pjs(self.cam,self.pjs)

    def trapezoid_Y_Minus(self):
        trape = self.trape.get()
        base_pos = np.array([[0,0],[1920,0],[1920,1080],[0,1080]],dtype=np.float32)
        trans_pos =np.array([[0,0],[1920,trape],[1920,1080-trape],[0,1080]], dtype=np.float32)
        trape_mat = cv2.getPerspectiveTransform(base_pos, trans_pos)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],trape_mat)
        self.main.update_pjs(self.cam,self.pjs)

    def trapezoid_LY_Plus(self):
        trape = self.trape.get()
        base_pos = np.array([[0,0],[1920,0],[1920,1080],[0,1080]],dtype=np.float32)
        trans_pos = np.array([[0,-trape],[1920,0],[1920,1080],[0,1080+trape]], dtype=np.float32)
        trape_mat = cv2.getPerspectiveTransform(base_pos, trans_pos)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],trape_mat)
        self.main.update_pjs(self.cam,self.pjs)

    def trapezoid_LY_Minus(self):
        trape = self.trape.get()
        base_pos = np.array([[0,0],[1920,0],[1920,1080],[0,1080]],dtype=np.float32)
        trans_pos = np.array([[0,trape],[1920,0],[1920,1080],[0,1080-trape]], dtype=np.float32)
        trape_mat = cv2.getPerspectiveTransform(base_pos, trans_pos)
        self.pjs[self.cam] = np.dot(self.pjs[self.cam],trape_mat)
        self.main.update_pjs(self.cam,self.pjs)

    def save(self):
        print("save projection"+self.cam)
        fname = pj_file

        recover_pjs = np.dot(np.array((
            (1, 0, self.org_diff[0]),
            (0, 1, self.org_diff[1]),
            (0, 0, 1)),dtype=np.float64),self.pjs[self.cam])

        self.pj_dict[self.cam]["projective_matrix"] = recover_pjs.tolist()

        if os.path.exists(fname):
            timestamp= os.path.getmtime(fname)
            mod_time_str = time.strftime("%m%d%H%M", time.localtime(timestamp))
            next_name = fname[:-5]+mod_time_str+".json"
            while os.path.exists(next_name):
                next_name = next_name[:-5] + "_.json"
            os.rename(fname, next_name)
        # file名は確定した

        with open(fname, "w") as f:
            json.dump(self.pj_dict, f, indent=4)


