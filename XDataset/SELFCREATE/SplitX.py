import json
import os
from json import *
import cv2
import os

from sympy.physics.quantum.circuitplot import pyplot as plt

# k = 0
# rois = dict()
# for i in range(0, 10):
#     for j in range(0, 10):
#         x, y = i * 28 + i, j * 28 + j
#         d_x, d_y = 28, 28
#
#         k += 1
#         rois[k] = [x, y, d_x, d_y]
#
# with open('ROIS.json', 'w') as f:
#     dump(rois, f, indent=2)

pics_count = 200
with open('ROIS.json',"r") as f:
    data = load(f)

k = 0
for img_path in os.listdir("PAGES"):
    img = cv2.imread(rf'C:\Users\us3r02\PycharmProjects\CNN_numbers_model\XDataset\SELFCREATE\PAGES\{img_path}')

    cv2.imshow("win",img)

    for X in range(1, 101):
        k += 1
        x, y, d_x, d_y = data[str(X)]
        X_roi = img[ y:y+ d_y, x:x+d_x]
        # cv2.imshow('f', X_roi)
        # cv2.waitKey(0)
        print(fr'C:\SELFCREATE\XPics\{k}.jpg')
        cv2.imwrite(fr'..\SELFCREATE\XPics\{k}.jpg', X_roi)