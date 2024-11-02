from .dscamera import DSCamera
from glob import glob
import cv2
import pathlib


calib_dir = "calib_files"
empty_image_dir = "img"
mask_dir = "mask"
pj_file = "calib_files/projective_matrices_fixed_for_20241003.json"


def load_ds_calib(calib_d):
    calibs = glob(calib_d+"/*")
    calibs.sort()

    dsCams = {}
    for x in calibs:
        p =pathlib.Path(x)
        if "camera" in p.name:
            dsc = DSCamera(x)
            cam = p.name[6:-5]
            dsCams[cam]=dsc
    return dsCams

dsCams = load_ds_calib(calib_dir)

def calibImage(cam):
    emptyImage = cv2.imread(empty_image_dir+"/camera"+cam+".jpg")
    cimg = dsCams[cam].to_perspective(emptyImage, img_size=(1080,1920),f=0.2)
    return cimg