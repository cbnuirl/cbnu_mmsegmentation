import os
from tqdm import tqdm
import shutil

path = "../datasets/morai_2nd/"

# file_path = path + "images/"
seg_file_path = path + "segmentation/"

# scenario_list = os.listdir(file_path)
scenario_list = os.listdir(seg_file_path)
scenario_list.sort()

for sce in tqdm(scenario_list):
    # camera_path = file_path + sce + "/camera/"
    seg_path = seg_file_path + sce
    # camera_list = os.listdir(camera_path)
    seg_list = os.listdir(seg_path)
    # camera_list.sort()
    seg_list.sort()

    # dest_path = "data/morai_2nd_syn/images/" + sce + "/camera/"
    dest_path = "data/morai_2nd_syn/labels/" + sce

    if not os.path.isdir(dest_path):
        # os.mkdir("data/morai_2nd_syn/images/" + sce)
        os.mkdir("data/morai_2nd_syn/labels/" + sce)
        # os.mkdir(dest_path)

    count = 0
    for cam in seg_list:
        if count >= 200 and sce[6:8] != "T2":
            break
        elif count >= 466 and sce[6:8] == "T2":
            break
        else:
            count += 1
        # print(seg_path + '/' + cam, dest_path + '/' + cam)
        shutil.copyfile(seg_path + '/' + cam, dest_path + '/' + cam)
