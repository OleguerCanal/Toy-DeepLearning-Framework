import numpy as np

def accuracy(Y_pred, Y_real):
	return np.sum(np.multiply(Y_pred, Y_real))/Y_pred.shape[1]
