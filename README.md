<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# loss_summary
A simple summary of loss functions in machine learning

## Table of Contents

- [loss summary](#loss-summary)

- [loss_summary](#loss_summary)
  - [Table of Contents](#table-of-contents)
  - [Intoduction](#intoduction)
    - [What is a loss function?](#what-is-a-loss-function)
      - [Note](#note)
  - [Regression Losses](#regression-losses)
      - [Mean Square Error](#mean-square-error)
      - [Mean Absolute Error/ L1 Loss](#mean-absolute-error-l1-loss)
    - [Hinge Loss/Multi class SVM Loss](#hinge-lossmulti-class-svm-loss)
    - [Cross Entropy Loss](#cross-entropy-loss)

## Intoduction
### What is a loss function?
In the context of an optimization algorithm, the function used to evaluate a candidate solution is referred to as the objective function.

Typically, with neural networks, we may seek to minimize a **loss function** (objective function) so as to search for a candidate solution that quanlify the model best.


#### Note
|  Param name |     | 
|:--------|:--------:|
| n | Number of training examples | 
| M | Number of classes | 
| i | ith training example in a data set | 
| c | class label | 
| y(i) | Ground truth label for ith training example | 
| y_hat(i) | Prediction for ith training example | 

## Regression Losses
#### Mean Square Error
  * simplified as 'mse', also known as Quadratic Loss/ L2 Loss
  * measured as the average of squared difference between predictions and actual observations
  * easier to calculate the gradients

$$MSE = \frac{\sum_{n}^{1}(y_{i} - \hat{y}_{i})^{2}}{n}$$


#### Mean Absolute Error/ L1 Loss
* measured as the average of sum of absolute differences between predictions and actual observations
* more robust to outliers
* hard to calculate the gradients

$$MAE = \frac{\sum_{n}^{1}\left |y_{i} - \hat{y}_{i}  \right |}{n}$$

## Classification Losses
### Hinge Loss/Multi class SVM Loss

* the score of correct category should be greater than sum of scores of all incorrect categories by some safety margin (usually one)
  
$$hingloss =\sum_{j\neq y_i}max(0, s_j - s_y +1)$$

### Cross Entropy Loss
$$CrossEntropyLoss = -\frac{1}{n}\sum(y_ilog(\hat{y_i})) +(1-y_i)log(1-\hat{y_i}))$$
    if $M > 2$ (i.e. multiclass classification), we calculate a separate loss for each class label per obeservation and sum the results:

$$mulCrossEntropyLoss = -\frac{1}{n}\sum_{1}^{n}\sum_{1}^{M}y_{i, c}log(p_{i,c})$$
