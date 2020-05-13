from abc import ABC, abstractmethod
import copy
import numpy as np
import time

try:  # If installed try to use parallelized einsum
    from einsum2 import einsum2 as einsum
except:
    print("Did not find einsum2, using numpy einsum (SLOWER)")
    from numpy import einsum as einsum

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
        input_shape = input_shape if type(
            input_shape) is tuple else (input_shape, )
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

    def __init__(self, input_shape=None):
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
            self.mask = np.random.choice([0, 1], size=(x.shape), p=[
                                         1 - self.ones_ratio, self.ones_ratio])
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


class MaxPool2D(Layer):
    def __init__(self, kernel_shape=(2, 2), stride=1, input_shape=None):
        super().__init__()
        self.kernel_shape = kernel_shape
        self.s = stride  # TODO(oleguer) Implement stride
        if input_shape is not None:
            self.compile(input_shape)  # Only care about channels

    def compile(self, input_shape):
        # Input shape must be (height, width, channels,)
        assert(len(input_shape) == 3)
        super().compile(input_shape)
        (ker_h, ker_w) = self.kernel_shape
        out_h = int((input_shape[0] - ker_h)/self.s) + 1
        out_w = int((input_shape[1] - ker_w)/self.s) + 1
        self.output_shape = (out_h, out_w, input_shape[2],)

    def __call__(self, inputs):
        """ Forward pass of MaxPool2D
            input should have shape (height, width, channels, n_images)
        """
        assert(len(inputs.shape) ==
               4)  # Input must have shape (height, width, channels, n_images)
        # Set input shape does not match with input sent
        assert(inputs.shape[:3] == self.input_shape)

        # Get shapes
        (ker_h, ker_w) = self.kernel_shape
        (out_h, out_w, _,) = self.output_shape

        # Compute convolution
        self.inputs = inputs  # Will be used in back pass
        output = np.empty(shape=self.output_shape + (self.inputs.shape[3],))
        for i in range(out_h):
            for j in range(out_w):
                # TODO(oleguer): Not sure if np.amax is parallel, look into  numexpr
                in_block = self.inputs[self.s*i:self.s *
                                       i+ker_h, self.s*j:self.s*j+ker_w, :, :]
                output[i, j, :, :] = np.amax(in_block, axis=(0, 1,))
        return output

    def backward(self, in_gradient, **kwargs):
        """ Pass gradient to left layer """
        # Get shapes
        (out_h, out_w, n_channels, n_points) = in_gradient.shape
        (ker_h, ker_w) = self.kernel_shape

        # Incoming gradient shape must match layer output shape
        assert(out_h == self.output_shape[0])
        # Incoming gradient shape must match layer output shape
        assert(out_w == self.output_shape[1])

        # Instantiate gradients
        left_layer_gradient = np.zeros(
            self.input_shape + (in_gradient.shape[-1],))
        for i in range(out_h):
            for j in range(out_w):
                in_block = self.inputs[self.s*i:self.s *
                                       i+ker_h, self.s*j:self.s*j+ker_w, :, :]
                mask = np.equal(in_block, np.amax(
                    in_block, axis=(0, 1,))).astype(int)
                masked_gradient = mask*in_gradient[i, j, :, :]
                left_layer_gradient[self.s*i:self.s*i+ker_h,
                                    self.s*j:self.s*j+ker_w, :, :] += masked_gradient
        return left_layer_gradient

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
        left_layer_gradient = (np.dot(self.weights.T, in_gradient))[:-1, :]

        # Regularization
        regularization_weights = copy.deepcopy(self.weights)
        regularization_weights[:, -1] = 0  # Bias col to 0
        regularization_term = 2*l2_regularization * \
            regularization_weights  # Only current layer weights != 0

        # Weight update
        # TODO: Rremove self if not going to update it
        self.gradient = np.dot(
            in_gradient, self.inputs.T) + regularization_term
        self.dw = momentum*self.dw + (1-momentum)*self.gradient
        self.weights -= lr*self.dw
        return left_layer_gradient

    def __initialize_weights(self, weight_initialization):
        if weight_initialization == "normal":
            self.weights = np.array(np.random.normal(
                0.0, 1./100.,
                                    (self.nodes, self.input_shape[0]+1)))  # Add biases
        if weight_initialization == "in_dim":
            self.weights = np.array(np.random.normal(
                0.0, 1./float(np.sqrt(self.input_shape[0])),
                (self.nodes, self.input_shape[0]+1)))  # Add biases
        if weight_initialization == "xavier":
            limit = np.sqrt(6/(self.nodes+self.input_shape[0]))
            self.weights = np.array(np.random.uniform(
                low=-limit,
                high=limit,
                size=(self.nodes, self.input_shape[0]+1)))  # Add biases


class Conv2D(Layer):
    def __init__(self, num_filters=5, kernel_shape=(5, 5), stride=1, dilation_rate=1, input_shape=None):
        super().__init__()
        self.num_filters = num_filters
        self.s = stride
        self.dilation = dilation_rate
        self.kernel_shape = kernel_shape
        aug_ker_h = (self.kernel_shape[0]-1)*self.dilation + 1
        aug_ker_w = (self.kernel_shape[1]-1)*self.dilation + 1
        # Kernel size considering dilation_rate
        self.aug_kernel_shape = (aug_ker_h, aug_ker_w)
        if input_shape is not None:
            self.compile(input_shape)  # Only care about channels

    def compile(self, input_shape):
        # Input shape must be (height, width, channels,)
        assert(len(input_shape) == 3)
        super().compile(input_shape)
        (ker_h, ker_w) = self.aug_kernel_shape
        out_h = int((input_shape[0] - ker_h)/self.s) + 1
        out_w = int((input_shape[1] - ker_w)/self.s) + 1
        self.output_shape = (out_h, out_w, self.num_filters,)
        self.__initialize_weights()

    def __call__(self, inputs):
        """ Forward pass of Conv Layer
            input should have shape (height, width, channels, n_images)
            channels should match kernel_shape
        """
        assert(len(inputs.shape) == 4)  # Input (height, width, channels, n_images)
        # Set input shape does not match with input sent
        assert(inputs.shape[:3] == self.input_shape)
        # Filter number of channels must match input channels
        assert(self.filters.shape[3] == inputs.shape[2])

        # Get shapes
        (aug_ker_h, aug_ker_w) = self.aug_kernel_shape
        (out_h, out_w, _,) = self.output_shape

        # Compute convolution
        self.inputs = inputs  # Will be used in back pass
        output = np.empty(shape=self.output_shape + (self.inputs.shape[3],))
        for i in range(out_h):
            for j in range(out_w):
                in_block = inputs[self.s*i:self.s*i+aug_ker_h:self.dilation,
                                  self.s*j:self.s*j+aug_ker_w:self.dilation, :, :]
                output[i, j, :, :] = einsum(
                    "ijcn,kijc->kn", in_block, self.filters)

        # Add biases
        output += einsum("ijcn,c->ijcn", np.ones(output.shape), self.biases)
        return output

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        """ Weight update
        """
        # Get shapes
        (out_h, out_w, _, _) = in_gradient.shape
        (aug_ker_h, aug_ker_w) = self.aug_kernel_shape

        # Incoming gradient shape must match layer output shape
        assert(out_h == self.output_shape[0])
        # Incoming gradient shape must match layer output shape
        assert(out_w == self.output_shape[1])

        # Instantiate gradients
        left_layer_gradient = np.zeros(
            self.input_shape + (in_gradient.shape[-1],))
        # Save it to compare with numerical (DEBUG)
        self.filter_gradients = np.zeros(self.filters.shape)
        self.bias_gradients = np.sum(in_gradient, axis=(0, 1, 3))

        for i in range(out_h):
            for j in range(out_w):
                in_block = self.inputs[self.s*i:self.s*i+aug_ker_h:self.dilation,
                                       self.s*j:self.s*j+aug_ker_w:self.dilation, :, :]
                grad_block = in_gradient[i, j, :, :]
                filter_grad = einsum("ijcn,kn->kijc", in_block, grad_block)
                self.filter_gradients += filter_grad
                left_layer_gradient[self.s*i:self.s*i+aug_ker_h:self.dilation,
                                    self.s*j:self.s*j+aug_ker_w:self.dilation, :, :] +=\
                    einsum("kijc,kn->ijcn", self.filters, grad_block)


        self.filter_gradients += 2*l2_regularization*self.filters

        if np.array_equal(self.dw, np.zeros(self.filters.shape)):
            self.dw = self.filter_gradients
        else:
            self.dw = momentum*self.dw + (1-momentum)*self.filter_gradients

        self.filters -= lr*self.dw  # TODO(oleguer): Add regularization
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
        self.dw = np.zeros(self.filters.shape)

    def show_filters(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(self.filters.shape[0])
        for i in range(self.filters.shape[0]):
            axes[i].imshow(self.filters[i][:, :, 0])
        plt.show()


class VanillaRNN(Layer):
    def __init__(self, output_size, state_size=100, input_size=None):
        super().__init__()
        self.EPS_ = 1e-9
        self.state_size = state_size
        self.output_size = output_size
        self.input_size = input_size
        if input_size is not None:
            self.compile(input_size)

    def compile(self, input_shape):
        super().compile(input_shape)  # Populates self.input_shape
        self.output_shape = (self.output_size,)
        self.__initialize_weights()
        self.reset_state()
        self.m_w = np.zeros(self.W.shape)
        self.m_u = np.zeros(self.U.shape)
        self.m_v = np.zeros(self.V.shape)
        self.m_b = np.zeros(self.b.shape)
        self.m_c = np.zeros(self.c.shape)
    
    def reset_state(self, state=None):
        if state is None:
            self.h = np.zeros((self.state_size, 1))
        else:
            self.h = state
        # print("reset")
        # self.m_w = np.zeros(self.W.shape)
        # self.m_u = np.zeros(self.U.shape)
        # self.m_v = np.zeros(self.V.shape)
        # self.m_b = np.zeros(self.b.shape)
        # self.m_c = np.zeros(self.c.shape)

    def __call__(self, inputs):
        # print("call")
        # print(np.sum(np.abs(self.h)))
        # print(self.h.shape)

        outputs = np.zeros(inputs.shape)
        self.inputs = inputs
        self.states = []
        self.a_ts = []
        for indx in range(inputs.shape[1]):
            x = np.expand_dims(inputs[:, indx], axis=1)
            a_t = np.dot(self.W, self.h) + np.dot(self.U, x) + self.b
            self.h = np.tanh(a_t)
            o_t = np.dot(self.V, self.h) + self.c
            outputs[:, indx] = o_t.flatten()
            # save stuff
            self.a_ts.append(a_t)
            self.states.append(self.h[:, 0])

        # print(np.sum(np.abs(self.h)))
        # print("outputs")
        # print(outputs)
        # import sys
        # sys.exit()
        return outputs

    def backward(self, in_gradient, lr=0.001, momentum=0.9, l2_regularization=0.1):
        # Compute dl_dh, dl_da backtracking
        tau = in_gradient.shape[-1] - 1
        dl_dh_t = np.dot(in_gradient[:, tau].T, self.V)
        mat = np.diag(1-np.square(np.tanh(self.a_ts[tau][:, 0])))
        dl_da_t = np.dot(dl_dh_t, mat)
        dl_dh = [dl_dh_t]
        dl_da = [dl_da_t]  # g_t
        for t in reversed(list(range(tau))):
            dl_dh_t = np.dot(in_gradient[:, t].T, self.V) + np.dot(dl_da_t, self.W)
            dl_da_t = np.dot(dl_dh_t, np.diag(1-np.square(np.tanh(self.a_ts[t][:, 0]))))
            dl_dh.insert(0, dl_dh_t)
            dl_da.insert(0, dl_da_t)

        # Compute dl_dw, dl_du, dl_dv
        dl_dw = np.zeros(self.W.shape)
        dl_du = np.zeros(self.U.shape)
        dl_dv = np.zeros(self.V.shape)
        dl_dc = np.zeros(self.c.shape)
        dl_db = np.zeros(self.b.shape)
        for t in range(0, tau+1):
            dl_da_t = np.expand_dims(dl_da[t], axis=1)
            do_dt = np.expand_dims(in_gradient[:, t], axis=1)
            h_t = np.expand_dims(self.states[t], axis=1).T
            x_t = np.expand_dims(self.inputs[:, t], axis=1).T
            if t > 0:
                h_t_1 = np.expand_dims(self.states[t-1], axis=1).T
                dl_dw += np.dot(dl_da_t, h_t_1)
            dl_du += np.dot(dl_da_t, x_t)
            dl_dv += np.dot(do_dt, h_t)
            dl_dc += do_dt
            dl_db += dl_da_t

        # Only for debugging gradients
        # self.dl_dw = dl_dw
        # self.dl_du = dl_du
        # self.dl_dv = dl_dv
        # self.dl_db = dl_db
        # self.dl_dc = dl_dc
        
        # Avoid exploding gradients
        dl_dw = np.clip(dl_dw, -5, 5)
        dl_du = np.clip(dl_du, -5, 5)
        dl_dv = np.clip(dl_dv, -5, 5)
        dl_db = np.clip(dl_db, -5, 5)
        dl_dc = np.clip(dl_dc, -5, 5)

        if np.array_equal(self.m_w, np.zeros(self.W.shape)):
            self.m_w = np.square(dl_dw)
            self.m_u = np.square(dl_du)
            self.m_v = np.square(dl_dv)
            self.m_b = np.square(dl_db)
            self.m_c = np.square(dl_dc)
        else:
            self.m_w = 0.9*self.m_w + 0.1*np.square(dl_dw)
            self.m_u = 0.9*self.m_u + 0.1*np.square(dl_du)
            self.m_v = 0.9*self.m_v + 0.1*np.square(dl_dv)
            self.m_b = 0.9*self.m_b + 0.1*np.square(dl_db)
            self.m_c = 0.9*self.m_c + 0.1*np.square(dl_dc)
        # self.m_w += np.square(dl_dw)
        # self.m_u += np.square(dl_du)
        # self.m_v += np.square(dl_dv)
        # self.m_b += np.square(dl_db)
        # self.m_c += np.square(dl_dc)
        self.W = self.W - lr*np.multiply(np.reciprocal(np.sqrt(self.m_w + self.EPS_)), dl_dw)
        self.U = self.U - lr*np.multiply(np.reciprocal(np.sqrt(self.m_u + self.EPS_)), dl_du)
        self.V = self.V - lr*np.multiply(np.reciprocal(np.sqrt(self.m_v + self.EPS_)), dl_dv)
        self.b = self.b - lr*np.multiply(np.reciprocal(np.sqrt(self.m_b + self.EPS_)), dl_db)
        self.c = self.c - lr*np.multiply(np.reciprocal(np.sqrt(self.m_c + self.EPS_)), dl_dc)

    def __initialize_weights(self):
        self.W = np.array(np.random.normal(0.0, 1./100.,
                    (self.state_size, self.state_size)))
        self.U = np.array(np.random.normal(0.0, 1./100.,
                    (self.state_size, self.input_size)))
        # self.b = np.array(np.random.normal(0.0, 1./100.,
        #             (self.state_size,1)))
        self.V = np.array(np.random.normal(0.0, 1./100.,
                    (self.output_size, self.state_size)))
        # self.c = np.array(np.random.normal(0.0, 1./1000.,
        #             (self.output_size,1)))
        self.b = np.zeros((self.state_size,1))
        self.c = np.zeros((self.output_size,1))