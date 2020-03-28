import matplotlib.pyplot as plt
import numpy as np
import cv2

def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open('data/'+filename, 'rb') as fo:
		dictionary = pickle.load(fo, encoding='bytes')
	return dictionary

def getXY(dataset, num_classes=10):
	"""Splits dataset into 2 np mat x, y (dim along rows)"""
	# 1. Convert labels to one-hot vectors
	labels = np.array(dataset[b"labels"])
	one_hot_labels = np.zeros((labels.size, num_classes))
	one_hot_labels[np.arange(labels.size), labels] = 1
	return np.mat(dataset[b"data"]).T, np.mat(one_hot_labels).T

def plot(flatted_image, shape=(32, 32, 3), order='F'):
	image = np.reshape(flatted_image, shape, order=order)
	cv2.imshow("image", image)
	cv2.waitKey()

def prob_to_class(Y_pred_prob):
	"""Given array of prob, returns max prob in one-hot fashon"""
	idx = np.argmax(Y_pred_prob, axis=0)
	Y_pred_class = np.zeros(Y_pred_prob.shape)
	Y_pred_class[idx, np.arange(Y_pred_class.shape[1])] = 1
	return Y_pred_class

def accuracy(Y_pred_classes, Y_real):
	return np.sum(np.multiply(Y_pred_classes, Y_real))/Y_pred_classes.shape[1]

def minibatch_split(X, Y, batch_size):
	"""Yields splited X, Y matrices in minibatches of given batch_size"""
	if (batch_size is None) or (batch_size > X.shape[1]):
		batch_size = X.shape[1]
	indx = list(range(X.shape[1]))
	np.random.shuffle(indx)
	for i in range(int(X.shape[1]/batch_size)):
		# Get minibatch
		X_minibatch = X[:, indx[i:i+batch_size]]
		Y_minibatch = Y[:, indx[i:i+batch_size]]
		if i == int(X.shape[1]/batch_size) - 1:  # Get all the remaining
			X_minibatch = X[:, indx[i:]]
			Y_minibatch = Y[:, indx[i:]]
		yield X_minibatch, Y_minibatch