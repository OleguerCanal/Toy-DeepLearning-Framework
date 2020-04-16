from abc import ABC, abstractmethod
import copy
import numpy as np

class Layer(ABC):
    """ Abstact class to represent Layer layers
        An Layer has:
            - __call__ method to apply Layer function to the inputs
            - gradient method to add Layer function gradient to the backprop
    """
    name = "GenericLayer"

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


# Activation Layers #####################################################
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

# MISC LAYERS ###########################################################
class Dropout(Layer):
    def __init__(self, ones_ratio=0.7):
        self.name = "Dropout"
        self.ones_ratio = ones_ratio

    def __call__(self, x, apply=True):
        if apply:
            self.mask = np.random.choice([0, 1], size=(x.shape), p=[1 - self.ones_ratio, self.ones_ratio])
            return np.multiply(self.mask, x)
        return x

    def backward(self, in_gradient, **kwargs):
        return np.multiply(self.mask, in_gradient)

class Flatten(Layer):
    def __call__(self, inputs):
        self.__in_shape = inputs.shape  # Store inputs shape to use in backprop
        m = inputs.shape[0]*inputs.shape[1]*inputs.shape[2]
        n_points = inputs.shape[3]
        return inputs.reshape((m, n_points))

    def backward(self, in_gradient):
        return in_gradient.reshape(self.__in_shape)
        
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


class Conv2D(Layer):
    def __init__(self, num_filters = 5, kernel_shape = (5, 5, 1)):
        self.filters = []
        self.biases = np.zeros(num_filters)
        # self.biases = np.array([0, 1])
        self.kernel_shape = kernel_shape
        for i in range(num_filters):
            # kernel = np.matrix(np.random.normal(0.0, 1./100., kernel_shape))
            kernel = np.ones(kernel_shape)/3
            if len(kernel.shape) == 2:
                kernel = np.expand_dims(kernel, axis=2)
            self.filters.append(kernel)
        self.filters = np.array(self.filters)
        pass

    def __call__(self, inputs):
        """ Forward pass of Conv Layer
            input should have shape (height, width, channels, n_images)
            channels should match kernel_shape
        """
        assert(len(inputs.shape) == 4) # Input must have shape (height, width, channels, n_images)
        assert(self.kernel_shape[2] == inputs.shape[2])  # Filter number of channels must match input channels

        # Compute shapes
        self.inputs = inputs
        (in_h, in_w, in_c, in_n) = self.inputs.shape
        (ker_h, ker_w, _) = self.kernel_shape
        out_h = in_h - ker_h + 1
        out_w = in_w - ker_w + 1

        # Compute convolution
        output = np.empty(shape=(out_h, out_w, self.filters.shape[0], in_n))
        for i in range(out_h):
            for j in range(out_w):
                output[i, j, :, :] = np.einsum("ijcn,kijc->kn",\
                     inputs[i:i+ker_h, j:j+ker_w, :, :], self.filters)
        # Add biases
        output += np.einsum("ijcn,c->ijcn", np.ones(output.shape), self.biases)
        return output

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        """ Weight update
        """
        left_layer_gradient = np.zeros(self.input_shape)
        filter_gradients = np.zeros(self.filters.shape)
        bias_gradients = np.average(in_gradient, axis=(0, 1, 3))

        (in_h, in_w, in_c, in_n) = self.inputs.shape
        (out_h, out_w, out_c, out_n) = in_gradient.shape
        (ker_h, ker_w, _) = self.kernel_shape

        for i in range(out_h):
            for j in range(out_w):
                in_block = inputs[i:i+ker_h, j:j+ker_w, :, :]
                grad_block = in_gradient[i, j, :, :]
                filter_gradients += np.average(np.einsum("ijcn,cn->ijcn",\
                    in_block, grad_block), axis=3)
                left_layer_gradient[i:i+ker_h, j:j+ker_w, :, :] +=\
                    np.einsum("kijc,kn->ijcn", self.filters, grad_block)
        
        self.filters += lr*filter_gradients  #TODO(oleguer): Add momentum and regularization
        self.biases += lr*bias_gradients
        return left_layer_gradient