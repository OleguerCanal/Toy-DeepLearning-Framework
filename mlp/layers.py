import numpy as np
import matplotlib
import time
import math
from matplotlib import pyplot as plt
import copy
import sys


class Dense:
    def __init__(self, nodes, input_dim, weight_initialization="fixed"):
        self.nodes = nodes
        self.input_shape = input_dim
        self.__initialize_weights(weight_initialization)
        self.dw = np.zeros(self.weights.shape)  # Weight updates

    def forward(self, inputs):
        self.inputs = np.append(
            inputs, [np.ones(inputs.shape[1])], axis=0)  # Add biases
        return self.weights*self.inputs

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        # Previous layer error propagation
        left_layer_gradient = (
            self.weights.T*in_gradient)[:-1, :]  # Remove bias

        # Regularization
        regularization_weights = copy.deepcopy(self.weights)
        regularization_weights[:, -1] = 0  # Bias col to 0
        regularization_term = 2*l2_regularization * \
            regularization_weights  # Only current layer weights != 0

        # Weight update
        self.dw = momentum*self.dw + \
            (1-momentum)*(in_gradient*self.inputs.T + regularization_term)
        self.weights -= lr*self.dw
        # return left_layer_gradient
        return in_gradient*self.inputs.T + regularization_term  # TODO(oelguer) CHANGE THIS BACK!!!!

    def __initialize_weights(self, weight_initialization):
        if weight_initialization == "fixed":
            self.weights = np.matrix(np.random.normal(
                0, 1./100.,
                                    (self.nodes, self.input_shape+1)))  # Add biases
        if weight_initialization == "in_dim":
            self.weights = np.matrix(np.random.normal(
                0, 1./float(self.input_shape),
                (self.nodes, self.input_shape+1)))  # Add biases
        if weight_initialization == "xavier":
            limit = np.sqrt(6/(self.nodes+self.input_shape))
            self.weights = np.matrix(np.random.uniform(
                low=-limit,
                high=limit,
                size=(self.nodes, self.input_shape+1)))  # Add biases


class Activation:
    def __init__(self, activation="softmax"):
        self.activation_type = activation
        self.weights = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = None
        if self.activation_type == "softmax":
            self.outputs = self.__softmax(x=inputs)
        if self.activation_type == "relu":
            self.outputs = self.__relu(x=inputs)
        return self.outputs

    def backward(self, in_gradient, **kwargs):
        # Gradient of activation function evaluated at pre-activation
        if self.activation_type == "softmax":
            return self.__softmax_diff(in_gradient=in_gradient)
        if self.activation_type == "relu":
            return self.__relu_diff(in_gradient=in_gradient)
        return None

    # ACTIVATION FUNCTIONS ###########################################################
    # TODO(oleguer): Maybe this shouldt be here
    def __relu(self, x):
        return np.multiply(x, (x > 0))

    def __relu_diff(self, in_gradient):
        # TODO(Oleguer): review this
        return np.multiply((self.inputs > 0), in_gradient)

    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def __softmax_diff(self, in_gradient):
        diags = np.einsum("ik,ij->ijk", self.outputs,
                          np.eye(self.outputs.shape[0]))
        out_prod = np.einsum("ik,jk->ijk", self.outputs, self.outputs)
        gradient = np.einsum("ijk,jk->ik", (diags - out_prod), in_gradient)
        return gradient


if __name__ == "__main__":
    pass
