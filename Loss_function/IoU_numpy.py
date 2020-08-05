# -*- coding: utf-8 -*-
# @Time    : 8/5/20 11:36 AM
# @Author  : Zeqi@@
# @FileName: IoU_numpy.py
# @Software: PyCharm

import math
import numpy as np

def Iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    iou = inter_area / (area1+area2-inter_area+1e-6)

    return iou

def GIou(box1,box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    C_width = max(0, max(xmax1, xmax2)- min(xmin1, xmin2))
    C_height = max(0, max(ymax1, ymax2)- min(ymin1, ymin2))
    area_C = C_width*C_height
    giou = iou - (area_C - (area1+area2-inter_area))/area_C
    return giou


def DIou(box1, box2):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    iou = Iou(box1, box2)

    center_x_1 = (xmax1 + xmin1)/2
    center_y_1 = (ymax1 + ymin1)/2
    center_x_2 = (xmax2 + xmin2)/2
    center_y_2 = (ymax2 + ymin2)/2

    euclidean_c12 = np.sqrt((center_x_2 - center_x_1)**2 + (center_y_2 - center_y_1)**2)

    C_x1, C_y1 = max(xmax1, xmax2), max(ymax1, ymax2)
    C_x2, C_y2 = min(xmin1, xmin2), min(ymin1, ymin2)

    euclidean_C = np.sqrt((C_x2 - C_x1)**2 + (C_y2 - C_y1)**2)
    diou = iou - euclidean_c12**2/euclidean_C**2
    return diou


def CIou(box1, box2):
    # b1 the truth
    # b2 the pred
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    iou = Iou(box1, box2)

    center_x_1 = (xmax1 + xmin1)/2
    center_y_1 = (ymax1 + ymin1)/2
    center_x_2 = (xmax2 + xmin2)/2
    center_y_2 = (ymax2 + ymin2)/2

    euclidean_c12 = np.sqrt((center_x_2 - center_x_1)**2 + (center_y_2 - center_y_1)**2)

    C_x1, C_y1 = max(xmax1, xmax2), max(ymax1, ymax2)
    C_x2, C_y2 = min(xmin1, xmin2), min(ymin1, ymin2)

    euclidean_C = np.sqrt((C_x2 - C_x1)**2 + (C_y2 - C_y1)**2)

    diou = iou - euclidean_c12**2/euclidean_C**2

    w_gt = max(0, xmax1 - xmin1)
    h_gt = max(0, ymax1 - ymin1)
    w_pred = max(0, xmax2 - xmin2)
    h_pred = max(0, ymax2 - ymin2)
    # print('w_gt: {0}, h_gt: {1}, w_pred: {2}, h_pred: {3}'.format(w_gt, h_gt, w_pred, h_pred))

    v = 4/(math.pi**2)*(np.arctan(w_gt/h_gt) - np.arctan(w_pred/h_pred))**2
    S = 1 - iou
    alpha = v/(S+v)

    ciou = diou - alpha*v
    return ciou
