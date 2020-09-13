# -*- coding: utf-8 -*-
# @Time    : 9/10/20 3:00 PM
# @Author  : Zeqi@@
# @FileName: pytorch_loss.py
# @Software: PyCharm

import torch
import time
import numpy as np
from numpy_loss import timer

class torch_ImgSeg_Metrics:
    def __init__(self, gt, pred):
        self.gt = gt
        self.pred = pred

    def confusion_matrix(self):
        K = len(torch.unique(self.gt))
        results = torch.zeros((K, K))
        for i in range(len(self.gt)):
            results[self.gt[i]][self.pred[i]] += 1
        return results

    def pixel_accuracy(self):
        matrix = self.confusion_matrix()
        acc = torch.diag(matrix).sum() / matrix.sum()
        return acc

    def mean_pixel_accuracy(self):
        matrix = self.confusion_matrix()
        k = matrix.size()[0]
        acc_class_i = 0
        for i in range(k):
            acc_class_i += matrix[i][i] / (torch.sum(matrix[i]))
        return acc_class_i / k

    def mean_iou(self):
        """  compute the value of mean iou
        :param pred:  2d array, int, prediction
        :param gt: 2d array, int, ground truth
        :return:
            miou: float, the value of miou
        """
        matrix = self.confusion_matrix()
        classes = matrix.size()[0]
        iou = 0
        for i in range(classes):
            iou += matrix[i][i] / (torch.sum(matrix[i]) + torch.sum(matrix[:][i]) - matrix[i][i])
        return iou / classes

    def Frequently_mean_iou(self):
        """  compute the value of frequently mean iou by measuring the frequence of mean iou
        :param pred:  2d array, int, prediction
        :param gt: 2d array, int, ground truth
        :return:
            miou: float, the value of frequently mean iou
        """
        matrix = self.confusion_matrix()
        classes = matrix.size()[0]
        iou = 0
        for i in range(classes):
            iou += matrix[i][i] / (torch.sum(matrix[i]) + torch.sum(matrix[:][i]) - matrix[i][i])
        return iou / torch.sum(matrix)




if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('====>> Use device {}'.format(device))

    pred_ = np.array([[0, 0, 0, 2],
                   [0, 0, 2, 1],
                   [1, 1, 1, 0],
                   [1, 0, 1, 2]])

    gt_ = np.array([[0, 0, 0, 2],
                    [0, 2, 2, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 2]])

    gt_flatten = torch.from_numpy(gt_).flatten().to(device)
    pred_flatten = torch.from_numpy(pred_).flatten().to(device)
    torch_imgseg_metrics = torch_ImgSeg_Metrics(gt_flatten, pred_flatten)

    start_time = time.time()
    np_cm = torch_imgseg_metrics.confusion_matrix().to(device)
    end_time = time.time()
    print('Numpy confusion matrix: \n {0} \n Time: {1} ms'.format(np_cm, timer(start_time, end_time)))
    #
    # pixel_accuracy, mean_pixel_acc
    pixel_acc = torch_imgseg_metrics.pixel_accuracy().to(device)
    print('Pixel accuracy: {0}%, type: {1}'.format(np.round(pixel_acc * 100, 2), type(pixel_acc)))
    mean_pixel_acc = torch_imgseg_metrics.mean_pixel_accuracy().to(device)
    print('Mean pixel accuracy: {0}%, type: {1}'.format(np.round(mean_pixel_acc * 100, 2), type(mean_pixel_acc)))

    #  miou and fmiou
    miou = torch_imgseg_metrics.mean_iou().to(device)
    print('Mean IU: {0}, type: {1}'.format(np.round(miou, 2), type(miou)))
    fmiou = torch_imgseg_metrics.Frequently_mean_iou().to(device)
    print('Frequently mean IU: {0}, type: {1}'.format(np.round(fmiou, 2), type(fmiou)))
