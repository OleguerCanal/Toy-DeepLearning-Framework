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


def plot_confusion_matrix(Y_pred, Y_real, class_names, path=None):
	heatmap = np.zeros((Y_pred.shape[0], Y_pred.shape[0]))
	for n in range(Y_pred.shape[-1]):
		i = np.where(Y_pred[:, n]==1)[0][0]
		j = np.where(Y_real[:, n]==1)[0][0]
		heatmap[i, j] += 1
		
	import seaborn as sn
	import pandas as pd
	
	df_cm = pd.DataFrame(heatmap, index = [i for i in class_names],
								  columns = [i for i in class_names])
	plt.figure(figsize = (10,10))
	sn.heatmap(df_cm, robust=True, square=True,annot=True, fmt='g')
	if path is not None:
		plt.savefig(path)
	plt.show()

def one_hotify(x, num_classes):
	size = 1
	try:
		size = x.size
	except :
		pass
	one_hot = np.zeros((size, num_classes))
	one_hot[np.arange(size), x] = 1
	return one_hot.T

def stringify(encoded_string, ind_to_char):
	string = ''
	elems = np.argmax(encoded_string, axis=0)
	for elem in elems:
		string += ind_to_char[elem]
	return string

def generate_sequence(rnn_layer, first_elem, ind_to_char, char_to_ind, length=10):
	k = len(ind_to_char)
	x = np.array([char_to_ind[elem] for elem in first_elem])
	x = one_hotify(x, num_classes = k)
	string = first_elem
	for i in range(length):
		probs = rnn_layer(x)
		string += stringify(probs, ind_to_char)
		next_elem = np.argmax(probs, axis=0)
		pred = [ind_to_char[elem] for elem in next_elem]
		x = one_hotify(next_elem, k)
	print(string)