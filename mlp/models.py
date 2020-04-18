import numpy as np
import matplotlib.pyplot as plt
import time
import math
import copy
from tqdm import tqdm
import pickle

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
from utils import minibatch_split

# from callbacks import Callback, LearningRateScheduler


class Sequential:
    def __init__(self, loss=None, pre_trained=None, metric=None):
        self.layers = []

        assert(loss is not None)  # You need a loss!!!
        self.loss = loss
        self.metric = metric

        if pre_trained is not None:
            self.load(pre_trained)

    def add(self, layer):
        """Add layer"""
        self.layers.append(layer)
        if len(self.layers) > 1: # Compile layer using output shape of previous layer
            assert(self.layers[-2].is_compiled)  # Input/Output shapes not set for previous layer!
            # Set input shape to be previous layer output_shape
            self.layers[-1].compile(input_shape=self.layers[-2].output_shape)

    def predict(self, X, apply_dropout=True):
        """Forward pass"""
        vals = X
        for layer in self.layers:
            if layer.name == "Dropout":
                vals = layer(vals, apply_dropout)
            else:
                vals = layer(vals)
        return vals

    def get_metric_loss(self, X, Y_real, use_dropout=True):
        """ Returns loss and classification accuracy """
        if X is None or Y_real is None:
            print("problem")
            return 0, np.inf
        Y_pred_prob = self.predict(X, use_dropout)
        metric_val = 0
        if self.metric is not None:
            metric_val = self.metric(Y_pred_prob, Y_real)
        loss = self.loss(Y_pred_prob, Y_real)
        return metric_val, loss

    def cost(self, Y_pred_prob, Y_real, l2_reg):
        """Computes cost = loss + regularization"""
        # Loss
        loss_val = self.loss(Y_pred_prob, Y_real)
        # Regularization
        w_norm = 0
        for layer in self.layers:
            if layer.weights is not None:
                w_norm += np.linalg.norm(layer.weights, 'fro')**2
        return loss_val + l2_reg*w_norm

    def fit(self, X, Y, X_val=None, Y_val=None, batch_size=None, epochs=None,
            iterations = None, lr=0.01, momentum=0.7, l2_reg=0.1,
            shuffle_minibatch=True, callbacks=[], **kwargs):
        """ Performs backrpop with given parameters.
            save_path is where model of best val accuracy will be saved
        """
        assert(epochs is None or iterations is None) # Only one can set it limit
        if iterations is not None:
            epochs = int(np.ceil(iterations/(X.shape[-1]/batch_size)))
        else:
            iterations = int(epochs*np.ceil((X.shape[-1]/batch_size)))

        # Store vars as class variables so they can be accessed by callbacks
        # TODO(think a better way)
        self.X = X
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.train_metric = 0
        self.val_metric = 0
        self.t = 0

        # Call callbacks
        for callback in callbacks:
            callback.on_training_begin(self)

        # Training
        stop = False
        pbar = tqdm(list(range(self.epochs)))
        for self.epoch in pbar:
        # for self.epoch in range(self.epochs):
            for X_minibatch, Y_minibatch in minibatch_split(X, Y, batch_size, shuffle_minibatch):
                # print("forward")
                Y_pred_prob = self.predict(X_minibatch)  # Forward pass
                # print("loss")
                gradient = self.loss.backward(
                    Y_pred_prob, Y_minibatch)  # Loss grad
                # print("backward")
                for layer in reversed(self.layers):  # Backprop (chain rule)
                    gradient = layer.backward(
                        in_gradient=gradient,
                        lr=self.lr,  # Trainable layer parameters
                        momentum=self.momentum,
                        l2_regularization=self.l2_reg)
                # Call callbacks
                for callback in callbacks:
                    callback.on_batch_end(self)
                if self.t >= iterations:
                    stop = True
                    break
                self.t += 1  # Step counter
            # Call callbacks
            for callback in callbacks:
                callback.on_epoch_end(self)
            # Update progressbar
            pbar.set_description("Train acc: " + str(self.train_metric) + ". Val acc: " + str(self.val_metric))
            if stop:
                break


    # IO functions ################################################
    def save(self, path):
        """ Saves current model to disk (Dont put file extension)"""
        directory = "/".join(path.split("/")[:-1])
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        with open(path + ".pkl", 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """ Loads model to disk (Dont put file extension)"""
        with open(path + ".pkl", 'rb') as input:
            tmp_dict = pickle.load(input)
            self.__dict__.update(tmp_dict)
