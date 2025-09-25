import json
import os
from json import *
import cv2
import os
from torchvision import transforms

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

import cv2
import numpy as np
import torch
from torchvision import transforms
import os

def preprocess_roi(roi, save_debug=False, debug_dir="debug_steps"):
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081

    # если RGB → серый
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # --- бинаризация ---
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # инверсия (хотим белый символ на чёрном фоне)
    if np.mean(thresh) > 127:
        bin_img = 255 - thresh
    else:
        bin_img = thresh

    # --- убираем края (рамку) ---
    margin = 5
    bin_img[:margin, :] = 0
    bin_img[-margin:, :] = 0
    bin_img[:, :margin] = 0
    bin_img[:, -margin:] = 0

    # --- ищем контуры ---
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # берём самый крупный контур
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        cut = bin_img[y:y + bh, x:x + bw]
    else:
        cut = bin_img.copy()

    # # --- вписываем в квадрат ---
    # size = max(cut.shape)
    # square = np.zeros((size, size), dtype=np.uint8)
    # y0 = (size - cut.shape[0]) // 2
    # x0 = (size - cut.shape[1]) // 2
    # square[y0:y0 + cut.shape[0], x0:x0 + cut.shape[1]] = cut

    # уменьшаем символ и вставляем в 28x28
    target_size = 15
    resized = cv2.resize(cut, (target_size, target_size), interpolation=cv2.INTER_AREA)
    final = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - target_size) // 2
    y_offset = (28 - target_size) // 2
    final[y_offset:y_offset + target_size, x_offset:x_offset + target_size] = resized

    # --- опционально сохраняем шаги ---
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_thresh.png"), thresh)
        cv2.imwrite(os.path.join(debug_dir, "02_bin_img.png"), bin_img)
        cv2.imwrite(os.path.join(debug_dir, "03_cut.png"), cut)
        cv2.imwrite(os.path.join(debug_dir, "04_final.png"), cut)

    # --- нормализация под MNIST ---
    arr = final.astype(np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).type(torch.float32)

    return tensor, final


for img_path in os.listdir("PAGES"):
    img = cv2.imread(rf'C:\Users\us3r02\PycharmProjects\CNN_numbers_model\XDataset\SELFCREATE\PAGES\{img_path}')

    cv2.imshow("win",img)
    for X in range(1, 101):
        k += 1
        x, y, d_x, d_y = data[str(X)]
        X_roi = img[ y:y+ d_y, x:x+d_x]
        X_roi = preprocess_roi(X_roi)
        # cv2.imshow('f', X_roi)
        # cv2.waitKey(0)
        print(fr'C:\SELFCREATE\XPics\{k}.jpg')
        cv2.imwrite(fr'..\SELFCREATE\XPics\{k}.jpg', X_roi[1])