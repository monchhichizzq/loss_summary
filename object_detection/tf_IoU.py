# -*- coding: utf-8 -*-
# @Time    : 8/5/20 11:07 AM
# @Author  : Zeqi@@
# @FileName: IoU.py
# @Software: PyCharm

"""Implements GIoU loss."""

import math
import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="Addons")
class XIoULoss(LossFunctionWrapper):
    """Implements the different IoU loss functions.
    IoU, GIoU, DIoU, CIoU
    IoU: Intersection over Union
    GIoU: Generalized Intersection over Union
    DIoU: Distance Intersection over Union
    CIoU: Complete Intersection over Union

    XIoU:
    A Metric and A Loss for Bounding Box Regression

      Usage with tf.keras API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=XIoULoss())
    ```
    Args:
      mode: one of ['iou', 'giou', 'diou', 'ciou'], decided to calculate IoU or GIoU or DIoU or CIoU loss.
    """

    def __init__(
        self,
        mode: str = "xiou",
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: Optional[str] = "xiou_loss",
    ):
        super().__init__(xiou_loss, name=name, reduction=reduction, mode=mode)

def xiou_loss(y_pred, y_true, mode: str = "xiou") -> tf.Tensor:
    """
    Args:
        y_true: true targets tensor. The coordinates of the each bounding
            box in boxes are encoded as [y_min, x_min, y_max, x_max].
        y_pred: predictions tensor. The coordinates of the each bounding
            box in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['iou', 'giou', 'diou', 'ciou'], decided to calculate IoU or GIoU or DIoU or CIoU loss.
    Returns:
        xIoU loss float `Tensor`.
    """
    if mode not in ['iou', 'giou', 'diou', 'ciou']:
        raise ValueError("Value of mode should be 'iou', 'giou', 'diou', 'ciou'")
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
 
    xiou = tf.squeeze(_calculate_xiou(y_pred, y_true, mode))
    return 1 - xiou


def _calculate_xiou(b1, b2, mode: str = "giou") -> tf.Tensor:
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['iou', 'giou', 'diou', 'ciou'], decided to calculate IoU or GIoU or DIoU or CIoU loss.
    Returns:
        XIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    # print(b1_ymin.numpy(), b1_xmin.numpy(), b1_ymax.numpy(), b1_xmax.numpy())
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height

    if mode == "giou":
        giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
        return giou


    center_x_1 = tf.math.divide(tf.add(b1_xmin, b1_xmax), 2)
    center_y_1 = tf.math.divide(tf.add(b1_ymin, b1_ymax), 2)
    center_x_2 = tf.math.divide(tf.add(b2_xmin, b2_xmax), 2)
    center_y_2 = tf.math.divide(tf.add(b2_ymin, b2_ymax), 2)

    euclidean_c12 = tf.sqrt(tf.square(tf.subtract(center_x_2,center_x_1)) + tf.square(tf.subtract(center_y_2,center_y_1)))

    C_x1, C_y1 = enclose_xmin, enclose_ymin
    C_x2, C_y2 = enclose_xmax, enclose_ymax
    euclidean_C = tf.sqrt(tf.square(tf.subtract(C_x2, C_x1)) + tf.square(tf.subtract(C_y2, C_y1)))

    diou = iou - tf.square(euclidean_c12) / tf.square(euclidean_C)

    if mode == "diou":
        return diou

    elif mode == "ciou":
        w_gt = tf.maximum(0, b1_xmax - b1_xmin)
        h_gt = tf.maximum(0, b1_ymax - b1_ymin)
        w_pred = tf.maximum(0, b2_xmax - b2_xmin)
        h_pred = tf.maximum(0, b2_ymax - b2_ymin)
        # print('w_gt: {0}, h_gt: {1}, w_pred: {2}, h_pred: {3}'.format(w_gt, h_gt, w_pred, h_pred))
        a = tf.divide(4, tf.square(math.pi))
        b = tf.square(tf.subtract(tf.math.atan(w_gt / h_gt),tf.math.atan(w_pred / h_pred)))
        v =  tf.multiply(a, b)
        S = tf.subtract(1,iou)
        alpha = tf.divide(v, tf.add(S, v))

        ciou = tf.subtract(diou, tf.multiply(alpha, v))
        ciou = tf.clip_by_value(ciou, clip_value_min=-1.0, clip_value_max=1.0)
        return ciou


if __name__ == '__main__':
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    loss = xiou_loss(boxes1, boxes2, mode='iou')
    # loss = gl(boxes1, boxes2)
    print('Loss: ', loss.numpy(), type(loss))  # Loss: [1.07500000298023224, 1.9333333373069763]