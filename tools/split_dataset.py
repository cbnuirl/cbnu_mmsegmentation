import numpy as np
import os
from tqdm import tqdm
import shutil

image_path = "data/morai_2nd_syn/images/"
label_path = "data/morai_2nd_syn/labels/"
index_path = "data/morai_2nd_syn/split_index/"

gt_path = "data/morai_2nd_syn/gtFine/"
left_path = "data/morai_2nd_syn/leftImg8bit/"

test_index = np.loadtxt(index_path + "test.csv", dtype=int)
train_index = np.loadtxt(index_path + "train.csv", dtype=int)
val_index = np.loadtxt(index_path + "val.csv", dtype=int)

image_list = os.listdir(image_path)
image_list.sort()
label_list = os.listdir(label_path)
label_list.sort()

for image in tqdm(image_list):
    if int(image[23:26]) in test_index:
        shutil.move(image_path + image, left_path + "test/" + image)
    elif int(image[23:26]) in train_index:
        shutil.move(image_path + image, left_path + "train/" + image)
    elif int(image[23:26]) in val_index:
        shutil.move(image_path + image, left_path + "val/" + image)

for label in tqdm(label_list):
    if int(label[23:26]) in test_index:
        shutil.move(label_path + label, gt_path + "test/" + label)
    elif int(label[23:26]) in train_index:
        shutil.move(label_path + label, gt_path + "train/" + label)
    elif int(label[23:26]) in val_index:
        shutil.move(label_path + label, gt_path + "val/" + label)