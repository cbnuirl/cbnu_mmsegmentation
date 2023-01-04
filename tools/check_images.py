import os
import cv2

path = "/home/hyundo/Workspace/mmsegmentation/data/morai_2nd_syn/leftImg8bit/train/"

image_list = os.listdir(path)
image_list.sort()

for image in image_list:
    print(image)
    img = cv2.imread(path + image)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("image", img)
    cv2.waitKey(1)
    if list(img.shape) != [540, 960, 3]:
        print(image, img.shape)
        break