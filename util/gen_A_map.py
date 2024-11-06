import cv2
import numpy as np
import os
import glob
import json
import fnmatch

#pj_file = "/mnt/bigdata/01_projects/2024_trusco/asset/pj/projective_matrices_berth.json"
#out_dir = "/mnt/bigdata/01_projects/2024_trusco/asset/fastrusco_mask/A_maps_berth_1103"
# マスクファイルをロード
#folder_name = "nkawa_berth_mask"

def generate_A_map(pj_dict,mask_folder_name,  out_dir):
# 各カメラの中心座標を設定
    camera_resolution = (1920, 1080)
    camera_center = (camera_resolution[0] // 2, camera_resolution[1] // 2)
#    paths = glob.glob(mask_folder_name+"/[A-B][0-9]*.png")
    camera_ids = list(pj_dict.keys())
    paths = [mask_folder_name+f"/{cam}.png" for cam in camera_ids]
    print("Camera IDs",camera_ids)

# 合成後の出力画像の設定
    output_width, output_height = 8000, 4000

# 画像毎のプロジェクティブ変換
    P = []
    frames = []

    for i, path in enumerate(paths):
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("Reading ", path)

        if frame is None:
            print(f"Failed to load image: {path}")
            continue

    # プロジェクティブ変換
        P.append(np.array(pj_dict[f"{camera_ids[i]}"]["projective_matrix"]))
        invP = np.linalg.inv(P[i])  # 逆行列
        transformed_frame = cv2.warpPerspective(frame, P[i], (output_width, output_height))

        transformed_frame = np.where(transformed_frame>127, 255, 0) # 二値化画像として整える

        frames.append(transformed_frame)

# 合成ループ
    if frames:
    # すべての画像のパラメータをピクセル毎で足し合わせ255で割ることで,その部分が写った画像の数を要素とする,画像と同サイズの行列を取得
        sum_frames = np.zeros_like(frames[0])
        sum_frames = np.sum(frames, axis=0)
        sum_frames = sum_frames / 255
        not_sum_zero = (sum_frames != 0)


        for i, frame in enumerate(frames):
            A_map = np.zeros_like(frames[0])
         # 各画像の画素値をそのピクセル毎に画像数で割る
            A_map[not_sum_zero] = (frame[not_sum_zero]/sum_frames[not_sum_zero])
            A_map = np.expand_dims(A_map, axis=-1).astype(np.uint8)
            invP = np.linalg.inv(P[i]).astype(np.float32)  # 逆行列
            A_map = cv2.warpPerspective(A_map, invP, camera_resolution)
            print("saving", f"{out_dir}/{camera_ids[i]}.png", A_map.shape)
            cv2.imwrite(f"{out_dir}/{camera_ids[i]}.png", A_map)

        print("A_maps have been created.")
    else:
        print("No frames available for merging.")

