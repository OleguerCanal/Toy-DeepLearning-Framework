import numpy as np
import matplotlib
import time
import math
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm
import pickle

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))

from utils import prob_to_class, accuracy
from layers import Activation, Dense

class Sequential:
    def __init__(self, loss="cross_entropy", reg_term=0.1, pre_saved=None):
        self.layers = []
        self.loss_type = loss
        self.reg_term = reg_term
        if pre_saved is not None:
            self.load(pre_saved)

    def add(self, layer):
        """Add layer"""
        self.layers.append(layer)

    def predict(self, X):
        """Forward pass"""
        vals = X
        for layer in self.layers:
            vals = layer.forward(vals)
        return vals

    def get_classification_metrics(self, X, Y_real):
        """ Returns loss and classification accuracy """
        if X is None or Y_real is None:
            return 1, 0
        Y_pred_prob = self.predict(X)
        Y_pred_classes = prob_to_class(Y_pred_prob)
        acc = accuracy(Y_pred_classes, Y_real)
        loss = self.__loss(Y_pred_prob, Y_real)
        return acc, loss

    def fit(self, X, Y, X_val=None, Y_val=None, batch_size=None, epochs=100, lr=0.01, momentum=0.7, l2_reg=0.1):
        if (batch_size is None) or (batch_size > X.shape[1]):
            batch_size = X.shape[1]

        # Learning tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Training
        pbar = tqdm(list(range(epochs)))
        for epoch in pbar:
            indx = list(range(X.shape[1]))
            np.random.shuffle(indx)
            for i in range(int(X.shape[1]/batch_size)):
                # Get minibatch
                X_minibatch = X[:, indx[i:i+batch_size]]
                Y_minibatch = Y[:, indx[i:i+batch_size]]

                # Forward pass
                Y_pred_prob = self.predict(X_minibatch)
                
                # Backprop
                gradient = self.__loss_diff(Y_pred_prob, Y_minibatch)
                # Prev grad depends on each layer function
                for layer in reversed(self.layers):
                    gradient = layer.backward(
                        in_gradient=gradient,
                        lr=lr,  # Trainable layer parameters
                        momentum=momentum,
                        l2_regularization=l2_reg)
            # Error tracking:
            train_acc, train_loss = self.get_classification_metrics(X, Y)
            val_acc, val_loss = self.get_classification_metrics(X_val, Y_val)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            pbar.set_description("Val acc: " + str(val_acc))

    def plot_training_progress(self, show=True, save=False, name="model_results"):
        fig, ax1 = plt.subplots()
        # Losses
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        # ax1.set_ylim(bottom=0)
        # ax1.set_ylim(top=2*np.amax(self.val_losses))
        if len(self.val_losses) > 0:
            ax1.plot(list(range(len(self.val_losses))),
                     self.val_losses, label="Val loss", c="red")
        ax1.plot(list(range(len(self.train_losses))),
                 self.train_losses, label="Train loss", c="orange")
        ax1.tick_params(axis='y')
        plt.legend(loc='center right')

        # Accuracies
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(bottom=0)
        ax2.set_ylim(top=1)
        n = len(self.train_accuracies)
        ax2.plot(list(range(n)),
                 np.array(self.train_accuracies), label="Train acc", c="green")
        if len(self.val_accuracies) > 0:
            n = len(self.val_accuracies)
            ax2.plot(list(range(n)),
                     np.array(self.val_accuracies), label="Val acc", c="blue")
        ax2.tick_params(axis='y')

        # fig.tight_layout()
        plt.title("Training Evolution")
        plt.legend(loc='upper right')

        if save:
            directory = "/".join(name.split("/")[:-1])
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            plt.savefig(name + ".png")
        if show:
            plt.show()

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

    # LOSS FUNCTIONS ##############################################
    # TODO(Oleguer): Should all this be here?
    def __loss(self, Y_pred_prob, Y_real):
        if self.loss_type == "cross_entropy":
            return self.__cross_entropy(Y_pred_prob, Y_real)
        if self.loss_type == "categorical_hinge":
            return self.__categorical_hinge(Y_pred_prob, Y_real)
        return None

    def __loss_diff(self, Y_pred, Y_real):
        if self.loss_type == "cross_entropy":
            return self.__cross_entropy_diff(Y_pred, Y_real)
        if self.loss_type == "categorical_hinge":
            return self.__categorical_hinge_diff(Y_pred, Y_real)
        return None

    def __cross_entropy(self, Y_pred, Y_real):
        return -np.sum(np.log(np.sum(np.multiply(Y_pred, Y_real), axis=0)))/float(Y_pred.shape[1])

    def __cross_entropy_diff(self, Y_pred, Y_real):
        # d(-log(x))/dx = -1/x
        f_y = np.multiply(Y_real, Y_pred)
        # Element-wise inverse
        loss_diff = - \
            np.reciprocal(f_y, out=np.zeros_like(
                Y_pred), where=abs(f_y) > 0.000001)
        return loss_diff/float(Y_pred.shape[1])

    def __categorical_hinge(self, Y_pred, Y_real):
        # L = SUM_data (SUM_dim_j(not yi) (MAX(0, y_pred_j - y_pred_yi + 1)))
        pos = np.sum(np.multiply(Y_real, Y_pred), axis=0)  # Val of right result
        neg = np.multiply(1-Y_real, Y_pred)  # Val of wrong results
        val = neg + 1 - pos
        val = np.multiply(val, (val > 0))
        return np.sum(val)/float(Y_pred.shape[1])

    def __categorical_hinge_diff(self, Y_pred, Y_real):
        # Forall j != yi: (y_pred_j - y_pred_yi + 1 > 0)
        # If     j == yi: -1 SUM_j(not yi) (y_pred_j - y_pred_yi + 1 > 0)
        pos = np.sum(np.multiply(Y_real, Y_pred), axis=0)  # Val of right result
        neg = np.multiply(1-Y_real, Y_pred)  # Val of wrong results
        wrong_class_activations = np.multiply(1-Y_real, (neg + 1 - pos > 0))  # Val of wrong results
        wca_sum = np.sum(wrong_class_activations, axis=0)
        neg_wca = np.einsum("ij,j->ij", Y_real, np.array(wca_sum).flatten())
        return wrong_class_activations - neg_wca