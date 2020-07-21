
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
import numpy as np


# Regression loss

def rmse(predictions, targets):
    '''
        Root Mean Square Error/ Quadratic Loss/ L2 Loss
    '''
    rmse_val = np.sqrt(np.mean((predictions - targets)**2))
    return rmse_val

def tf_rmse(predictions, targets):
    '''
        Root Mean Square Error/ Quadratic Loss/ L2 Loss
    '''
    rmse_val = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(predictions, targets), 2)))
    return rmse_val

def mae(predictions, targets):
    '''
        Mean Absolute Error
    '''
    mean_absolute_differences = np.mean(np.absolute(predictions - targets))
    return mean_absolute_differences

def tf_mae(predictions, targets):
    '''
        Mean Absolute Error
    '''
    mae = tf.reduce_mean(tf.abs(tf.subtract(predictions, targets)))
    return mae

def binary_crossentropy(prediction, target, epsilon):
    loss = []
    for i in range(2):
        if target[i] == 1:
            loss_= np.log(prediction[i])
        else:
            loss_= np.log(1 - prediction[i])
        loss.append(loss_)
    return np.sum(loss)/2
  

def CrossEntropy(predictions, targets, epsilon=1e-10, binary=False):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    if binary:
        cross_entropy = [binary_crossentropy(pred, targets[i], epsilon) for i, pred in enumerate(predictions)]
    else:
        cross_entropy = np.sum(targets * np.log(predictions + 1e-5))
    N = predictions.shape[0]
    ce_loss = -np.sum(cross_entropy)/N
    return ce_loss

def sigmoid(z):
  return 1 / (1 + np.exp(-z))


if __name__=='__main__':

    # Regression
    # test vectors
    y_hat = np.array([0.000, 0.166, 0.333])
    y_true = np.array([0.000, 0.254, 0.998])

    tf_y_hat = tf.cast(y_hat, tf.float32)
    tf_y_true = tf.cast(y_true, tf.float32)
    print(tf_y_hat, tf_y_true)

    # Root mean square error
    start=time.time()
    rmse_val = rmse(y_hat, y_true)
    middle=time.time()
    tf_rmse_val = tf_rmse(tf_y_hat, tf_y_true)
    end=time.time()
    print('\nRoot Mean Square Error:')
    print("numpy: {:^5f}, time: {:^5f}ms".format(rmse_val, (middle-start)*1000))
    print("tensorlow: {:^5f}, time: {:^5f}ms".format(tf_rmse_val, (end-middle)*1000))

    # Mean absolute error
    start=time.time()
    mae_val = mae(y_hat, y_true)
    middle=time.time()
    tf_mae_val = tf_mae(tf_y_hat, tf_y_true)
    end=time.time()
    print('\nMean Aboluste Error:')
    print("numpy: {:^5f}, time: {:^5f}ms".format(mae_val, (middle-start)*1000))
    print("tensorlow: {:^5f}, time: {:^5f}ms".format(tf_mae_val, (end-middle)*1000))


    # Classfication losses
    targets = [[0, 1, 0], [0, 0, 1]]
    predictions = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    
    start=time.time()
    cat_ce = CrossEntropy(predictions, targets, epsilon=1e-10, binary=False)
    middle=time.time()
    tf_cce = CategoricalCrossentropy()
    tf_cat_ce = tf_cce(targets, predictions).numpy()
    end=time.time()
    print('\nCategorial Crossentropy:')
    print("numpy: {:^5f}, time: {:^5f}ms".format(cat_ce, (middle-start)*1000))
    print("tensorlow: {:^5f}, time: {:^5f}ms".format(tf_cat_ce, (end-middle)*1000))


    targets = [[0, 1], [1, 0], [1,0]]
    predictions = [[0.05, 0.95], [0.8, 0.2], [0.6, 0.4]]
    start=time.time()
    binary_ce = CrossEntropy(predictions, targets, binary=True)
    middle=time.time()
    tf_cce = BinaryCrossentropy()
    tf_binary_ce = tf_cce(targets, predictions).numpy()
    end=time.time()
    print('\nBinary Crossentropy:')
    print("numpy: {:^5f}, time: {:^5f}ms".format(binary_ce, (middle-start)*1000))
    print("tensorlow: {:^5f}, time: {:^5f}ms".format(tf_binary_ce, (end-middle)*1000))


