# -*- coding: utf-8 -*-
# @Time    : 9/10/20 2:27 PM
# @Author  : Zeqi@@
# @FileName: main.py.py
# @Software: PyCharm

import time
import numpy as np
from numpy_loss import timer
from numpy_loss import numpy_ImgSeg_Metrics


if __name__=='__main__':
    is_torch = False
    is_tf = True

    pred_ = np.array([[0, 0, 0, 2],
                      [0, 0, 2, 1],
                      [1, 1, 1, 0],
                      [1, 0, 1, 2]])

    gt_ = np.array([[0, 0, 0, 2],
                    [0, 2, 2, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 2]])
    #  Flatten
    pred_flatten = pred_.flatten()
    gt_flatten = gt_.flatten()
    numpy_imgSeg_metrics = numpy_ImgSeg_Metrics(gt_flatten, pred_flatten)

    start_time = time.time()
    np_cm = numpy_imgSeg_metrics.compute_confusion_matrix()
    end_time = time.time()
    print('Numpy confusion matrix: \n {0} \n Time: {1} ms'.format(np_cm, timer(start_time, end_time)))

    start_time = time.time()
    #  pixel_accuracy, mean_pixel_acc
    pixel_acc = numpy_imgSeg_metrics.pixel_accuracy()
    print('Pixel accuracy: {0}%'.format(np.round(pixel_acc * 100, 2)))
    mean_pixel_acc = numpy_imgSeg_metrics.mean_pixel_accuracy()
    print('Mean pixel accuracy: {0}%'.format(np.round(mean_pixel_acc * 100, 2)))

    #  miou and fmiou
    miou = numpy_imgSeg_metrics.mean_iou()
    print('Mean IU: {0}'.format(np.round(miou, 2)))
    fmiou = numpy_imgSeg_metrics.Frequently_mean_iou()
    print('Frequently mean IU: {0}'.format(np.round(fmiou, 2)))
    end_time = time.time()
    print('Numpy process: \n Time: {} ms'.format(timer(start_time, end_time)))

    ######################################################################################
    if is_torch:
        import torch
        from pytorch_loss import torch_ImgSeg_Metrics

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('\n====>> Use device {}'.format(device))

        gt_flatten = torch.from_numpy(gt_).flatten().to(device)
        pred_flatten = torch.from_numpy(pred_).flatten().to(device)
        torch_imgseg_metrics = torch_ImgSeg_Metrics(gt_flatten, pred_flatten)

        start_time = time.time()
        np_cm = torch_imgseg_metrics.confusion_matrix().to(device)
        end_time = time.time()
        print('Numpy confusion matrix: \n {0} \n Time: {1} ms'.format(np_cm, timer(start_time, end_time)))


        start_time = time.time()
        #  pixel_accuracy, mean_pixel_acc
        pixel_acc = torch_imgseg_metrics.pixel_accuracy().to(device)
        print('Pixel accuracy: {0}%, type: {1}'.format(np.round(pixel_acc * 100, 2), type(pixel_acc)))
        mean_pixel_acc = torch_imgseg_metrics.mean_pixel_accuracy().to(device)
        print('Mean pixel accuracy: {0}%, type: {1}'.format(np.round(mean_pixel_acc * 100, 2), type(mean_pixel_acc)))

        #  miou and fmiou
        miou = torch_imgseg_metrics.mean_iou().to(device)
        print('Mean IU: {0}, type: {1}'.format(np.round(miou, 2), type(miou)))
        fmiou = torch_imgseg_metrics.Frequently_mean_iou().to(device)
        print('Frequently mean IU: {0}, type: {1}'.format(np.round(fmiou, 2), type(fmiou)))
        end_time = time.time()
        print('Torch process: \n Time: {} ms'.format(timer(start_time, end_time)))

    if is_tf:
        import tensorflow as tf
        from tensorflow_loss import tensorflow_ImgSeg_Metrics
        import tensorflow.keras.backend as k

        gt_ = tf.convert_to_tensor(gt_, dtype=tf.int32)
        pred_ = tf.convert_to_tensor(pred_, dtype=tf.int32)
        gt_flatten = k.flatten(gt_)
        pred_flatten = k.flatten(pred_)
        tensorflow_imgseg_metrics = tensorflow_ImgSeg_Metrics(gt_flatten, pred_flatten)

        start_time = time.time()
        np_cm = tensorflow_imgseg_metrics.confusion_matrix()
        end_time = time.time()
        print('Numpy confusion matrix: \n {0} \n Time: {1} ms'.format(np_cm, timer(start_time, end_time)))

        start_time = time.time()
        #  pixel_accuracy, mean_pixel_acc
        pixel_acc = tensorflow_imgseg_metrics.pixel_accuracy()
        print('Pixel accuracy: {0}%, type: {1}'.format(np.round(pixel_acc * 100, 2), type(pixel_acc)))
        mean_pixel_acc = tensorflow_imgseg_metrics.mean_pixel_accuracy()
        print('Mean pixel accuracy: {0}%, type: {1}'.format(np.round(mean_pixel_acc * 100, 2), type(mean_pixel_acc)))

        #  miou and fmiou
        miou = tensorflow_imgseg_metrics.mean_iou()
        print('Mean IU: {0}, type: {1}'.format(np.round(miou, 2), type(miou)))
        fmiou = tensorflow_imgseg_metrics.Frequently_mean_iou()
        print('Frequently mean IU: {0}, type: {1}'.format(np.round(fmiou, 2), type(fmiou)))
        end_time = time.time()

        print('Tensorflow process: \n Time: {} ms'.format(timer(start_time, end_time)))



