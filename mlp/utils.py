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
	return np.array(dataset[b"data"]).T, np.array(one_hot_labels).T

def LoadXY(filename):
	return getXY(LoadBatch(filename))

def plot(flatted_image, shape=(32, 32, 3), order='F'):
	image = np.reshape(flatted_image, shape, order=order)
	cv2.imshow("image", image)
	cv2.waitKey()

def accuracy(Y_pred_classes, Y_real):
	return np.sum(np.multiply(Y_pred_classes, Y_real))/Y_pred_classes.shape[1]

def minibatch_split(X, Y, batch_size, shuffle=True, compansate=False):
	"""Yields splited X, Y matrices in minibatches of given batch_size"""
	if (batch_size is None) or (batch_size > X.shape[-1]):
		batch_size = X.shape[-1]

	if not compansate:
		indx = list(range(X.shape[-1]))
		if shuffle:
			np.random.shuffle(indx)
		for i in range(int(X.shape[-1]/batch_size)):
			pos = i*batch_size
			# Get minibatch
			X_minibatch = X[..., indx[pos:pos+batch_size]]
			Y_minibatch = Y[..., indx[pos:pos+batch_size]]
			if i == int(X.shape[-1]/batch_size) - 1:  # Get all the remaining
				X_minibatch = X[..., indx[pos:]]
				Y_minibatch = Y[..., indx[pos:]]
			yield X_minibatch, Y_minibatch
	else:
		class_sum = np.sum(Y, axis=1)*Y.shape[0]
		class_count = np.reciprocal(class_sum, where=abs(class_sum) > 0)
		x_probas = np.dot(class_count, Y)
		n = X.shape[-1]
		for i in range(int(n/batch_size)):
			indxs = np.random.choice(range(n), size=batch_size, replace=True, p=x_probas)
			yield X[..., indxs], Y[..., indxs]