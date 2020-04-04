from abc import ABC, abstractmethod
import copy
import numpy as np


class Layer(ABC):
    """ Abstact class to represent Layer layers
        An Layer has:
            - __call__ method to apply Layer function to the inputs
            - gradient method to add Layer function gradient to the backprop
    """

    def __init__(self):
        self.weights = None

    @abstractmethod
    def __call__(self, inputs):
        """ Applies a function to given inputs """
        pass

    @abstractmethod
    def backward(self, in_gradient):
        """ Receives right-layer gradient and multiplies it by current layer gradient """
        pass


# TRAINABLE LAYERS ######################################################

class Dense(Layer):
    def __init__(self, nodes, input_dim, weight_initialization="in_dim"):
        self.nodes = nodes
        self.input_shape = input_dim
        self.__initialize_weights(weight_initialization)
        self.dw = np.zeros(self.weights.shape)  # Weight updates

    def __call__(self, inputs):
        self.inputs = np.append(
            inputs, [np.ones(inputs.shape[1])], axis=0)  # Add biases
        return self.weights*self.inputs

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        # Previous layer error propagation
        # Remove bias TODO Think about this
        left_layer_gradient = (self.weights.T*in_gradient)[:-1, :]

        # Regularization
        regularization_weights = copy.deepcopy(self.weights)
        regularization_weights[:, -1] = 0  # Bias col to 0
        regularization_term = 2*l2_regularization * \
            regularization_weights  # Only current layer weights != 0

        # Weight update
        # TODO: Rremove self if not going to update it
        self.gradient = in_gradient*self.inputs.T + regularization_term
        self.dw = momentum*self.dw + (1-momentum)*self.gradient
        self.weights -= lr*self.dw
        return left_layer_gradient

    def __initialize_weights(self, weight_initialization):
        if weight_initialization == "normal":
            self.weights = np.matrix(np.random.normal(
                0.0, 1./100.,
                                    (self.nodes, self.input_shape+1)))  # Add biases
        if weight_initialization == "in_dim":
            self.weights = np.matrix(np.random.normal(
                0.0, 1./float(np.sqrt(self.input_shape)),
                (self.nodes, self.input_shape+1)))  # Add biases
        if weight_initialization == "xavier":
            limit = np.sqrt(6/(self.nodes+self.input_shape))
            self.weights = np.matrix(np.random.uniform(
                low=-limit,
                high=limit,
                size=(self.nodes, self.input_shape+1)))  # Add biases


# Activation Layers ######################################################
class Softmax(Layer):
    def __call__(self, x):
        self.outputs = np.exp(x) / np.sum(np.exp(x), axis=0)
        return self.outputs

    def backward(self, in_gradient, **kwargs):
        diags = np.einsum("ik,ij->ijk", self.outputs,
                          np.eye(self.outputs.shape[0]))
        out_prod = np.einsum("ik,jk->ijk", self.outputs, self.outputs)
        gradient = np.einsum("ijk,jk->ik", (diags - out_prod), in_gradient)
        return gradient


class Relu(Layer):
    def __call__(self, x):
        self.inputs = x
        return np.multiply(x, (x > 0))

    def backward(self, in_gradient, **kwargs):
        # TODO(Oleguer): review this
        return np.multiply((self.inputs > 0), in_gradient)

class Dropout(Layer):
    def __init__(self, ones_ratio=0.7):
        self.ones_ratio = ones_ratio

    def __call__(self, x):
        self.mask = np.random.choice([0, 1], size=(x.shape), p=[1 - self.ones_ratio, self.ones_ratio])
        return np.multiply(self.mask, x)

    def backward(self, in_gradient, **kwargs):
        return np.multiply(self.mask, in_gradient)
