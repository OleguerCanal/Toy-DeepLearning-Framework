from abc import ABC, abstractmethod
import copy
import numpy as np

# Layer Templates #######################################################
class Layer(ABC):
    """ Abstact class to represent Layers
        A Layer has:
            - compile  method which computes input/output shapes (and intializes weights)
            - __call__ method to apply Layer function to the inputs
            - gradient method to add Layer function gradient to the backprop
    """
    name = "GenericLayer"

    def __init__(self):
        self.weights = None
        self.is_compiled = False
        self.input_shape = None
        self.output_shape = None
        # NOTE: INPUT/OUTPUT shapes ignore 

    @abstractmethod
    def compile(self, input_shape):
        """ Updates self.input_shape (and self.output_shape)
            If input_dim not set by user, when added to a model 
            this method is called based on previous layer output_shape
            
            For trainable layers, it also should initializes weights and
            gradient placeholders according to input_shape.
            
            Note: input_shape should be a tuple of the shape
            ignoring the number of samples (last elem)
        """
        input_shape = input_shape if type(input_shape) is tuple else (input_shape, )
        self.input_shape = input_shape
        self.is_compiled = True
        # print("compiled")

    @abstractmethod
    def __call__(self, inputs):
        """ Applies a function to given inputs """
        pass

    @abstractmethod
    def backward(self, in_gradient):
        """ Receives right-layer gradient and multiplies it by current layer gradient """
        pass

class ConstantShapeLayer(Layer):
    """ Common structure of Layers which do not modify the shape of input and output
    """
    def __init__(self, input_shape = None):
        super().__init__()
        if input_shape is not None:
            self.compile(input_shape)
        
    def compile(self, input_shape):
        super().compile(input_shape)  # Populates self.input_shape and self.is_compiled
        self.output_shape = self.input_shape


# Activation Layers #####################################################
class Softmax(ConstantShapeLayer):
    def __call__(self, x):
        self.outputs = np.exp(x) / np.sum(np.exp(x), axis=0)
        return self.outputs

    def backward(self, in_gradient, **kwargs):
        diags = np.einsum("ik,ij->ijk", self.outputs,
                          np.eye(self.outputs.shape[0]))
        out_prod = np.einsum("ik,jk->ijk", self.outputs, self.outputs)
        gradient = np.einsum("ijk,jk->ik", (diags - out_prod), in_gradient)
        return gradient

class Relu(ConstantShapeLayer):
    def __call__(self, x):
        self.inputs = x
        return np.multiply(x, (x > 0))

    def backward(self, in_gradient, **kwargs):
        # TODO(Oleguer): review this
        return np.multiply((self.inputs > 0), in_gradient)

# MISC LAYERS ###########################################################
class Dropout(ConstantShapeLayer):
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
    def __init__(self, input_shape=None):
        super().__init__()
        if input_shape is not None:
            self.compile(input_shape)

    def compile(self, input_shape):
        super().compile(input_shape)  # Updates self.input_shape and self.is_compiled
        self.output_shape = (np.prod(self.input_shape),)

    def __call__(self, inputs):
        self.__in_shape = inputs.shape  # Store inputs shape to use in backprop
        m = self.output_shape[0]
        n_points = inputs.shape[3]
        return inputs.reshape((m, n_points))

    def backward(self, in_gradient, **kwargs):
        return np.array(in_gradient).reshape(self.__in_shape)


# TRAINABLE LAYERS ######################################################
class Dense(Layer):
    def __init__(self, nodes, input_dim=None, weight_initialization="in_dim"):
        super().__init__()
        self.nodes = nodes
        self.weight_initialization = weight_initialization
        if input_dim is not None:  # If user sets input, automatically compile
            self.compile(input_dim)

    def compile(self, input_shape):
        super().compile(input_shape)  # Updates self.input_shape and self.is_compiled
        self.__initialize_weights(self.weight_initialization)
        self.dw = np.zeros(self.weights.shape)  # Weight updates
        self.output_shape = (self.nodes,)

    def __call__(self, inputs):
        self.inputs = np.append(
            inputs, [np.ones(inputs.shape[1])], axis=0)  # Add biases
        return np.dot(self.weights, self.inputs)

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        # Previous layer error propagation
        # Remove bias TODO Think about this
        left_layer_gradient = (np.dot(self.weights.T,in_gradient))[:-1, :]

        # Regularization
        regularization_weights = copy.deepcopy(self.weights)
        regularization_weights[:, -1] = 0  # Bias col to 0
        regularization_term = 2*l2_regularization * \
            regularization_weights  # Only current layer weights != 0

        # Weight update
        # TODO: Rremove self if not going to update it
        self.gradient = np.dot(in_gradient, self.inputs.T) + regularization_term
        self.dw = momentum*self.dw + (1-momentum)*self.gradient
        self.weights -= lr*self.dw
        return left_layer_gradient

    def __initialize_weights(self, weight_initialization):
        if weight_initialization == "normal":
            self.weights = np.matrix(np.random.normal(
                0.0, 1./100.,
                                    (self.nodes, self.input_shape[0]+1)))  # Add biases
        if weight_initialization == "in_dim":
            self.weights = np.matrix(np.random.normal(
                0.0, 1./float(np.sqrt(self.input_shape[0])),
                (self.nodes, self.input_shape[0]+1)))  # Add biases
        if weight_initialization == "xavier":
            limit = np.sqrt(6/(self.nodes+self.input_shape[0]))
            self.weights = np.matrix(np.random.uniform(
                low=-limit,
                high=limit,
                size=(self.nodes, self.input_shape[0]+1)))  # Add biases


class Conv2D(Layer):
    def __init__(self, num_filters = 5, kernel_shape = (5, 5), stride = 1, input_shape=None):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_shape = kernel_shape
        self.stride = stride  # TODO(oleguer) Implement stride
        if input_shape is not None:
            self.compile(input_shape)  # Only care about channels

    def compile(self, input_shape):
        assert(len(input_shape) == 3) # Input shape must be (height, width, channels,)
        super().compile(input_shape)
        (ker_h, ker_w) = self.kernel_shape
        out_h = input_shape[0] - ker_h + 1
        out_w = input_shape[1] - ker_w + 1
        self.output_shape = (out_h, out_w, self.num_filters,)
        self.__initialize_weights()

    def __call__(self, inputs):
        """ Forward pass of Conv Layer
            input should have shape (height, width, channels, n_images)
            channels should match kernel_shape
        """
        assert(len(inputs.shape) == 4) # Input must have shape (height, width, channels, n_images)
        assert(inputs.shape[:3] == self.input_shape)  # Set input shape does not match with input sent
        assert(self.filters.shape[3] == inputs.shape[2])  # Filter number of channels must match input channels

        # Get shapes
        (ker_h, ker_w) = self.kernel_shape
        (out_h, out_w, _,) = self.output_shape

        # Compute convolution
        self.inputs = inputs  # Will be used in back pass
        output = np.empty(shape = self.output_shape + (self.inputs.shape[3],))
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
        # Get shapes
        (out_h, out_w, _, _) = in_gradient.shape
        (ker_h, ker_w) = self.kernel_shape
        assert(out_h == self.output_shape[0])  # Incoming gradient shape must match layer output shape
        assert(out_w == self.output_shape[1])  # Incoming gradient shape must match layer output shape

        # Instantiate gradients
        left_layer_gradient = np.zeros(self.input_shape + (in_gradient.shape[-1],))
        self.filter_gradients = np.zeros(self.filters.shape)  # Save it to compare with numerical (DEBUG)
        self.bias_gradients = np.sum(np.sum(in_gradient, axis=(0, 1,)), axis=-1)

        for i in range(out_h):
            for j in range(out_w):
                in_block = self.inputs[i:i+ker_h, j:j+ker_w, :, :]
                grad_block = in_gradient[i, j, :, :]
                filter_grad = np.sum(\
                    np.einsum("ijcn,kn->kijcn", in_block, grad_block), axis=-1)
                self.filter_gradients += filter_grad
                left_layer_gradient[i:i+ker_h, j:j+ker_w, :, :] +=\
                    np.einsum("kijc,kn->ijcn", self.filters, grad_block)

        self.dw = momentum*self.dw + (1-momentum)*self.filter_gradients
        self.filters -= lr*self.dw  #TODO(oleguer): Add regularization
        self.biases -= lr*self.bias_gradients
        return left_layer_gradient

    def __initialize_weights(self):
        self.filters = []
        self.biases = np.random.normal(0.0, 1./100., self.num_filters)
        # self.biases = np.zeros(self.num_filters)
        full_kernel_shape = self.kernel_shape + (self.input_shape[2],)
        # self.biases = np.array([0, 1])
        for i in range(self.num_filters):
            kernel = np.random.normal(0.0, 1./100., full_kernel_shape)
            # kernel = np.ones(full_kernel_shape)/3
            if len(kernel.shape) == 2:
                kernel = np.expand_dims(kernel, axis=2)
            self.filters.append(kernel)
        self.filters = np.array(self.filters)
        self.dw = 0

    def show_filters(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(self.filters.shape[0])
        for i in range(self.filters.shape[0]):
            axes[i].imshow(self.filters[i][:, :, 0])
        plt.show()