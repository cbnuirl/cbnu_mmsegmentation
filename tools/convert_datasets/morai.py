import os
import cv2
import numpy as np
from tqdm import tqdm
# from numba import jit, cuda
import argparse
import shutil

parser = argparse.ArgumentParser(description="Convert MORAI into Cityscapes")
parser.add_argument("--data_path", type=str, help="MORAI data path")
args = parser.parse_args()
if args.data_path[len(args.data_path) - 1] != '/':
    args.data_path = args.data_path + '/'

color_map = [
    [255, 90, 241],     # vehicle
    [66, 7, 158],       # bus
    [0, 243, 64],       # truck
    [248, 158, 235],    # policeCar
    [211, 222, 241],    # ambulance
    [255, 255, 255],    # schoolBus
    [112, 32, 48],      # otherCar
    [255, 255, 255],    # motorcycle
    [248, 171, 255],    # bicycle
    [255, 255, 255],    # twoWheeler
    [255, 0, 233],      # pedestrian
    [255, 255, 255],    # rider
    [9, 161, 181],      # freespace
    [80, 180, 98],      # curb
    [118, 78, 176],     # sidewalk
    [255, 255, 255],    # crossWalk
    [0, 207, 255],      # safetyZone
    [134, 59, 141],     # speedBump
    [248, 28, 81],      # roadMark
    [255, 188, 123],    # whiteLane
    [255, 100, 0],      # yellowLane
    [152, 255, 141],    # blueLane
    [255, 255, 255],    # redLane
    [106, 163, 145],    # stopLane
    [241, 90, 41],      # trafficSign
    [0, 233, 197],      # trafficlight
    [0, 182, 255],      # constructionGuide
    [136, 153, 189],    # trafficDrum
    [255, 255, 255],    # rubberCone
    [255, 255, 255],    # warningTriangle
    [247, 234, 110],    # fence
    [255, 255, 255],    # egoVehicle
    [210, 229, 168]     # background
]
color_map_gray = [round(0.299*R + 0.587*G + 0.114*B) for (R, G, B) in color_map]
color_map_gray = np.asarray(color_map_gray)
# print(color_map_gray)
path = args.data_path + "label/"
output = args.data_path + "output/"
label_list = os.listdir(path)
label_list.sort()

# @jit(target_backend="cuda")
def rgb2gray():
    for label in tqdm(label_list):

        # Setup image and convert into grayscale
        img = cv2.imread(path + label)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Change color of trafficDrum(because grayscale index conflicts with safetyzone)
        # (136, 153, 179) -> (136, 153, 189) RGB order
        lower = np.array([108, 61, 179])
        upper = np.array([108, 61, 179])
        mask = cv2.inRange(hsv, lower, upper)
        img[mask > 0] = (189, 153, 136)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert elements into index
        for idx, gray in enumerate(color_map_gray):
            img_gray = np.where(img_gray == gray, idx, img_gray)

        # Write image
        if not os.path.isdir(output):
            os.mkdir(output)
        cv2.imwrite(output + label[:-4] + "_labelIds.png", img_gray)

        # Show image
        img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
        img_gray = cv2.resize(img_gray, (0, 0), fx=0.4, fy=0.4)

        cv2.imshow("img", img)
        cv2.imshow("img_gray", img_gray)
        cv2.waitKey(1)

def move_output():
    train_img = args.data_path + "leftImg8bit/train/"
    val_img = args.data_path + "leftImg8bit/val/"
    train_gt = args.data_path + "gtFine/train/"
    val_gt = args.data_path + "gtFine/val/"

    if not os.path.isdir(train_gt):
        os.makedirs(train_gt)
    if not os.path.isdir(val_gt):
        os.makedirs(val_gt)
    
    train_list = os.listdir(train_img)
    val_list = os.listdir(val_img)

    for img in train_list:
        shutil.move(output + img[:-4] + "_labelIds.png", train_gt + img[:-4] + "_labelIds.png")
    for img in val_list:
        shutil.move(output + img[:-4] + "_labelIds.png", val_gt + img[:-4] + "_labelIds.png")
    
    os.rmdir(output)


if __name__=="__main__":
    rgb2gray()
    move_output()
    # print(color_map_gray)
