import numpy as np

class Accuracy:
	def __init__(self):
		self.name = "Accuracy"

	def __call__(self, Y_pred, Y_real):
		Y_pred_class = self.__prob_to_class(Y_pred)
		return np.sum(np.multiply(Y_pred_class, Y_real))/Y_pred_class.shape[1]


	def __prob_to_class(self, Y_pred_prob):
		"""Given array of prob, returns max prob in one-hot fashon"""
		idx = np.argmax(Y_pred_prob, axis=0)
		Y_pred_class = np.zeros(Y_pred_prob.shape)
		Y_pred_class[idx, np.arange(Y_pred_class.shape[1])] = 1
		return Y_pred_class
