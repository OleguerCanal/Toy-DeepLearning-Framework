import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """ Abstact class to represent Loss functions
        An activation has:
            - forward method to apply activation to the inputs
            - backward method to add activation gradient to the backprop
    """
    def __init__(self):
        self._EPS = 1e-5

    @abstractmethod
    def __call__(self, Y_pred, Y_real):
        """ Computes loss value according to predictions and true labels"""
        pass

    @abstractmethod
    def backward(self, Y_pred, Y_real):
        """ Computes loss gradient according to predictions and true labels"""
        pass


# LOSSES IMPLEMNETATIONS  #########################################

class CrossEntropy(Loss):
    def __call__(self, Y_pred, Y_real):
        return -np.sum(np.log(np.sum(np.multiply(Y_pred, Y_real), axis=0)))/float(Y_pred.shape[1])

    def backward(self, Y_pred, Y_real):
        # d(-log(x))/dx = -1/x
        f_y = np.multiply(Y_real, Y_pred)
        # Element-wise inverse
        loss_diff = - \
            np.reciprocal(f_y, out=np.zeros_like(
                Y_pred), where=abs(f_y) > self._EPS)
        return loss_diff/float(Y_pred.shape[1])


class CategoricalHinge(Loss):
    def __call__(self, Y_pred, Y_real):
        # L = SUM_data (SUM_dim_j(not yi) (MAX(0, y_pred_j - y_pred_yi + 1)))
        pos = np.sum(np.multiply(Y_real, Y_pred),
                     axis=0)  # Val of right result
        neg = np.multiply(1-Y_real, Y_pred)  # Val of wrong results
        val = neg + 1. - pos
        val = np.multiply(val, (val > 0))
        return np.sum(val)/float(Y_pred.shape[1])

    def backward(self, Y_pred, Y_real):
        # Forall j != yi: (y_pred_j - y_pred_yi + 1 > 0)
        # If     j == yi: -1 SUM_j(not yi) (y_pred_j - y_pred_yi + 1 > 0)
        pos = np.sum(np.multiply(Y_real, Y_pred),
                     axis=0)  # Val of right result
        neg = np.multiply(1-Y_real, Y_pred)  # Val of wrong results
        wrong_class_activations = np.multiply(
            1-Y_real, (neg + 1. - pos > 0))  # Val of wrong results
        wca_sum = np.sum(wrong_class_activations, axis=0)
        neg_wca = np.einsum("ij,j->ij", Y_real, np.array(wca_sum).flatten())
        return (wrong_class_activations - neg_wca)/float(Y_pred.shape[1])
