# -*- coding: utf-8 -*-
# @Time    : 9/10/20 3:29 PM
# @Author  : Zeqi@@
# @FileName: tensorflow_loss.py
# @Software: PyCharm


import tensorflow as tf
import time
import numpy as np
from numpy_loss import timer
import tensorflow.keras.backend as k


"""
Note:
    There is no assignment in tensorflow, so we can't do tensor[i][j] += 1, in order to have the same effect, we do:

    indices = tf.constant([[1,1]])
    updates = tf.constant([1])
    tensor = tf.zeros([3,3], dtype=tf.int32)
    updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
    print(updated)
    updated = tf.tensor_scatter_nd_add(updated, indices, updates)
    print(updated)

"""

class tensorflow_ImgSeg_Metrics:
    def __init__(self, gt, pred):
        self.gt = gt
        self.pred = pred

    def confusion_matrix(self):
        K, _ = tf.unique(self.gt)
        K = K.shape[0]

        # Tensor value assignment
        results = tf.zeros([K, K], dtype=tf.int32)
        updates = tf.constant([1])
        for i in range(len(self.gt)):
            indices = tf.constant([[self.gt[i].numpy(), self.pred[i].numpy()]])
            results = tf.tensor_scatter_nd_add(results, indices, updates)
        return results

    def pixel_accuracy(self):
        matrix = self.confusion_matrix()
        diag = [matrix[i][i] for i in range(matrix.shape[0])]
        acc = tf.divide(tf.reduce_sum(diag), tf.reduce_sum(matrix))
        return acc

    def mean_pixel_accuracy(self):
        matrix = self.confusion_matrix()
        k = matrix.shape[0]
        acc_class_i = 0
        for i in range(k):
            acc_class_i += tf.divide(matrix[i][i], (tf.reduce_sum(matrix[i])))
        return acc_class_i / k

    def mean_iou(self):
        """  compute the value of mean iou
        :param pred:  2d array, int, prediction
        :param gt: 2d array, int, ground truth
        :return:
            miou: float, the value of miou
        """
        matrix = self.confusion_matrix()
        classes = matrix.shape[0]
        iou = 0
        for i in range(classes):
            iou += tf.divide(matrix[i][i], tf.subtract(tf.add(tf.reduce_sum(matrix[i]), tf.reduce_sum(matrix[:][i])),matrix[i][i]))
        return tf.divide(iou, classes)

    def Frequently_mean_iou(self):
        """  compute the value of frequently mean iou by measuring the frequence of mean iou
        :param pred:  2d array, int, prediction
        :param gt: 2d array, int, ground truth
        :return:
            miou: float, the value of frequently mean iou
        """
        matrix = self.confusion_matrix()
        classes = matrix.shape[0]
        iou = 0
        for i in range(classes):
            iou += tf.divide(matrix[i][i], tf.subtract(tf.add(tf.reduce_sum(matrix[i]), tf.reduce_sum(matrix[:][i])),matrix[i][i]))
        matrix = tf.cast(matrix, dtype=tf.float64)
        return tf.divide(iou, tf.reduce_sum(matrix))




if __name__ == '__main__':
    pred_ = np.array([[0, 0, 0, 2],
                   [0, 0, 2, 1],
                   [1, 1, 1, 0],
                   [1, 0, 1, 2]])

    gt_ = np.array([[0, 0, 0, 2],
                    [0, 2, 2, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 2]])

    gt_ = tf.convert_to_tensor(gt_, dtype=tf.int32)
    pred_ = tf.convert_to_tensor(pred_,dtype=tf.int32)
    gt_flatten = k.flatten(gt_)
    pred_flatten = k.flatten(pred_)
    tensorflow_imgseg_metrics = tensorflow_ImgSeg_Metrics(gt_flatten, pred_flatten)

    start_time = time.time()
    np_cm = tensorflow_imgseg_metrics.confusion_matrix()
    end_time = time.time()
    print('Numpy confusion matrix: \n {0} \n Time: {1} ms'.format(np_cm, timer(start_time, end_time)))
    #
     # pixel_accuracy, mean_pixel_acc
    pixel_acc = tensorflow_imgseg_metrics.pixel_accuracy()
    print('Pixel accuracy: {0}%, type: {1}'.format(np.round(pixel_acc * 100, 2), type(pixel_acc)))
    mean_pixel_acc = tensorflow_imgseg_metrics.mean_pixel_accuracy()
    print('Mean pixel accuracy: {0}%, type: {1}'.format(np.round(mean_pixel_acc * 100, 2), type(mean_pixel_acc)))

    #  miou and fmiou
    miou = tensorflow_imgseg_metrics.mean_iou()
    print('Mean IU: {0}, type: {1}'.format(np.round(miou, 2), type(miou)))
    fmiou = tensorflow_imgseg_metrics.Frequently_mean_iou()
    print('Frequently mean IU: {0}, type: {1}'.format(np.round(fmiou, 2), type(fmiou)))
