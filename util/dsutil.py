from .dscamera import DSCamera
from glob import glob
import numpy as np
import cv2
import pathlib


# for normal
berth_mode = False

calib_dir = "calib_files"
empty_image_dir = "img"
mask_dir = "mask"
#pj_file = "calib_files/projective_matrices_fixed_for_20241003.json"
pj_file = "calib_files/projective_matrices_nkawa_handcraft_1934.json"

alpha_mask_image_dir = "A_mask"
# for berth_mode

if berth_mode:
    calib_dir = "berth/calib_files"
    empty_image_dir = "berth/img"
    alpha_mask_image_dir = "berth/A_mask"
    mask_dir = "berth/mask"
    pj_file = "berth/calib_files/projective_matrices_berth.json"


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

def updateDScam(cam):
    calib_file = calib_dir+"/camera"+cam+".json"
    dsc = DSCamera(calib_file)
    return dsc

def calibImage(cam):
    emptyImage = cv2.imread(empty_image_dir+"/camera"+cam+".jpg")
    cimg = dsCams[cam].to_perspective(emptyImage, img_size=(1080,1920),f=0.2)
    return cimg

def _crop(pjs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    stitched_ltrb = [np.inf, np.inf, -np.inf, -np.inf]
    for p in pjs.values():
        tf_corners = cv2.perspectiveTransform(np.array(((0, 0), (1920, 0), (0, 1080), (1920, 1080)), dtype=np.float32)[:, np.newaxis], p).squeeze(axis=1)
        stitched_ltrb[0] = min(stitched_ltrb[0], tf_corners[0, 0], tf_corners[2, 0])
        stitched_ltrb[1] = min(stitched_ltrb[1], tf_corners[0, 1], tf_corners[1, 1])
        stitched_ltrb[2] = max(stitched_ltrb[2], tf_corners[1, 0], tf_corners[3, 0])
        stitched_ltrb[3] = max(stitched_ltrb[3], tf_corners[2, 1], tf_corners[3, 1])
    for n, p in pjs.items():
        pjs[n] = np.dot(np.array((
            (1, 0, -stitched_ltrb[0]),
            (0, 1, -stitched_ltrb[1]),
            (0, 0, 1)
        ), dtype=np.float64), p)

    return pjs, (int(stitched_ltrb[2] - stitched_ltrb[0]), int(stitched_ltrb[3] - stitched_ltrb[1])), (stitched_ltrb[0],stitched_ltrb[1])