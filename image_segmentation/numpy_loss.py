# -*- coding: utf-8 -*-
# @Time    : 9/10/20 12:00 PM
# @Author  : Zeqi@@
# @FileName: numpy_loss.py
# @Software: PyCharm

import time
import numpy as np
from sklearn.metrics import confusion_matrix

'''
    The metrics of image segmentation is realized by the first step confusion metrix of pixels 

'''

def timer(start_time, end_time):
    return np.round((end_time - start_time) * 1000, 2)

class numpy_ImgSeg_Metrics:
    def __init__(self, gt, pred):
        self.gt = gt
        self.pred = pred

    def compute_confusion_matrix(self):
      '''Computes a confusion matrix using numpy for two np.arrays
      true and pred.

      Results are identical (and similar in computation time) to:
        "from sklearn.metrics import confusion_matrix"

      However, this function avoids the dependency on sklearn.'''

      K = len(np.unique(self.gt)) # Number of classes
      result = np.zeros((K, K))

      for i in range(len(self.gt)):
        result[self.gt[i]][self.pred[i]] += 1

      return result

    def pixel_accuracy(self):
        matrix = confusion_matrix(y_true=self.gt, y_pred=self.pred)
        acc = np.diag(matrix).sum() / matrix.sum()
        return acc

    def mean_pixel_accuracy(self):
        matrix = confusion_matrix(y_true=self.gt, y_pred=self.pred)
        k = np.shape(matrix)[0]
        acc_class_i = 0
        for i in range(k):
            acc_class_i += matrix[i][i] / (np.sum(matrix[i]))
        return acc_class_i/k

    def mean_iou(self):
        """  compute the value of mean iou
        :param pred:  2d array, int, prediction
        :param gt: 2d array, int, ground truth
        :return:
            miou: float, the value of miou
        """
        matrix = confusion_matrix(y_true=self.gt, y_pred=self.pred)
        classes = np.shape(matrix)[0]
        iou = 0
        for i in range(classes):
            iou += matrix[i][i]/(np.sum(matrix[i]) + np.sum(matrix[:][i]) - matrix[i][i])
        return iou/classes

    def Frequently_mean_iou(self):
        """  compute the value of frequently mean iou by measuring the frequence of mean iou
        :param pred:  2d array, int, prediction
        :param gt: 2d array, int, ground truth
        :return:
            miou: float, the value of frequently mean iou
        """
        matrix = confusion_matrix(y_true=self.gt, y_pred=self.pred)
        classes = np.shape(matrix)[0]
        iou = 0
        for i in range(classes):
            iou += matrix[i][i]/(np.sum(matrix[i]) + np.sum(matrix[:][i]) - matrix[i][i])
        return iou/np.sum(matrix)


if __name__ == '__main__':
    pred_ = np.array([[0, 0, 0, 2],
                   [0, 0, 2, 1],
                   [1, 1, 1, 0],
                   [1, 0, 1, 2]])

    gt_ = np.array([[0, 0, 0, 2],
                    [0, 2, 2, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 2]])
    # Flatten
    pred_flatten = pred_.flatten()
    gt_flatten = gt_.flatten()
    numpy_imgSeg_metrics = numpy_ImgSeg_Metrics(gt_flatten, pred_flatten)

    start_time = time.time()
    np_cm = numpy_imgSeg_metrics.compute_confusion_matrix()
    end_time = time.time()
    print('Numpy confusion matrix: \n {0} \n Time: {1} ms'.format(np_cm, timer(start_time, end_time)))

    start_time = time.time()
    sk_cm = confusion_matrix(gt_flatten, pred_flatten)
    end_time = time.time()
    print('Sklearn confusion matrix: \n {0} \n Time: {1} ms'.format(sk_cm, timer(start_time, end_time)))

    # pixel_accuracy, mean_pixel_acc
    pixel_acc = numpy_imgSeg_metrics.pixel_accuracy()
    print('Pixel accuracy: {0}%'.format(np.round(pixel_acc*100, 2)))
    mean_pixel_acc = numpy_imgSeg_metrics.mean_pixel_accuracy()
    print('Mean pixel accuracy: {0}%'.format(np.round(mean_pixel_acc*100, 2)))

    # miou and fmiou
    miou = numpy_imgSeg_metrics.mean_iou()
    print('Mean IU: {0}'.format(np.round(miou, 2)))
    fmiou = numpy_imgSeg_metrics.Frequently_mean_iou()
    print('Frequently mean IU: {0}'.format(np.round(fmiou, 2)))



