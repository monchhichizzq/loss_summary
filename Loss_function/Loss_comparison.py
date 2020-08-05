# -*- coding: utf-8 -*-
# @Time    : 8/5/20 10:59 AM
# @Author  : Zeqi@@
# @FileName: Custom_loss.py
# @Software: PyCharm


# IoU
# GIoU
# DIoU
# CIoU
import sys
from prettytable import PrettyTable
# reload(sys)
# sys.setdefaultencoding('utf8')

import time
from Loss_function.IoU_numpy import *
from Loss_function.tf_IoU import  *
import matplotlib.pyplot as plt

def time_out(start, end):
    return np.round((end-start)*1000, 1)

def plot_rects(boxes1, boxes2):
    #   b1_ymin, b1_xmin, b1_ymax, b1_xmax
    #   top, left, bottom, right
    fig = plt.figure()
    ax = fig.add_subplot(111)
    currentAxis = plt.gca()
    for idx, (gt, pt) in enumerate(zip(boxes1, boxes2)):
        # Left, bottom, width, height
        print(gt, pt)
        currentAxis.add_patch(plt.Rectangle((gt[2], gt[1]), (gt[1] + gt[3])/2, (gt[0] + gt[2])/2,
                                            fill=False, edgecolor='green', linewidth=2))

        currentAxis.add_patch(plt.Rectangle((pt[2], pt[1]), (pt[1] + pt[3])/2, (pt[0] + pt[2])/2,
                                            fill=False, edgecolor='red', linewidth=2))

    # plt.xticks(np.arange(0, img_width + 1, 40))
    # plt.yticks(np.arange(0, img_height + 1, 40))
    currentAxis.invert_yaxis()
    plt.show()

if __name__=='__main__':
    # Two bounding boxes

    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    plot_rects(boxes1.numpy(), boxes2.numpy())

    # IoU
    start = time.time()
    np_loss = []
    for i in range(len(boxes1)):
        box1, box2 = boxes1[i].numpy(), boxes2[i].numpy()
        np_loss.append(1 - Iou(box1, box2))
    middle = time.time()
    tf_loss = xiou_loss(boxes1, boxes2, mode='iou')
    end = time.time()

    np_IoU_time = time_out(start, middle)
    tf_IoU_time = time_out(middle, end)

    # GIoU
    start_g = time.time()
    np_giou_loss = []
    for i in range(len(boxes1)):
        box1, box2 = boxes1[i].numpy(), boxes2[i].numpy()
        np_giou_loss.append(1 - GIou(box1, box2))
    middle_g = time.time()
    tf_giou_loss = xiou_loss(boxes1, boxes2, mode='giou')
    end_g = time.time()

    np_GIoU_time = time_out(start_g, middle_g)
    tf_GIoU_time = time_out(middle_g, end_g)

    # DIoU
    start_d = time.time()
    np_diou_loss = []
    for i in range(len(boxes1)):
        box1, box2 = boxes1[i].numpy(), boxes2[i].numpy()
        np_diou_loss.append(1 - DIou(box1, box2))
    middle_d = time.time()
    tf_diou_loss = xiou_loss(boxes1, boxes2, mode='diou')
    end_d = time.time()

    np_DIoU_time = time_out(start_d, middle_d)
    tf_DIoU_time = time_out(middle_d, end_d)

    # CIoU
    start_c = time.time()
    np_ciou_loss = []
    for i in range(len(boxes1)):
        box1, box2 = boxes1[i].numpy(), boxes2[i].numpy()
        np_ciou_loss.append(1 - CIou(box1, box2))
    middle_c = time.time()
    tf_ciou_loss = xiou_loss(boxes1, boxes2, mode='ciou')
    end_c = time.time()

    np_CIoU_time = time_out(start_c, middle_c)
    tf_CIoU_time = time_out(middle_c, end_c)

    table = PrettyTable(['Numpy', 'Loss', 'Value', 'Time (ms)'])
    table.add_row(['1', 'IoU', np_loss, np_IoU_time])
    table.add_row(['2', 'GIoU', np_giou_loss, np_GIoU_time])
    table.add_row(['3', 'DIoU', np_diou_loss, np_DIoU_time])
    table.add_row(['4', 'CIoU', np_ciou_loss, np_CIoU_time])

    tf_table = PrettyTable(['Tensorflow', 'Loss', 'Value', 'Time (ms)'])
    tf_table.add_row(['1', 'IoU', tf_loss, tf_IoU_time])
    tf_table.add_row(['2', 'GIoU', tf_giou_loss, tf_GIoU_time])
    tf_table.add_row(['3', 'DIoU', tf_diou_loss, tf_DIoU_time])
    tf_table.add_row(['4', 'CIoU', tf_ciou_loss, tf_CIoU_time])

    print(table)
    print(tf_table)
