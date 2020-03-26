import numpy as np
import matplotlib
import time
import math
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
from layers import Activation, Dense
from utils import prob_to_class, accuracy

class Sequential:
    def __init__(self, loss="cross_entropy", reg_term=0.1):
        self.layers = []
        self.loss_type = loss
        self.reg_term = reg_term

    def add(self, layer):
        """Add layer"""
        self.layers.append(layer)

    def predict(self, X):
        """Forward pass"""
        vals = X
        for layer in self.layers:
            vals = layer.forward(vals)
        return vals

    def get_metrics(self, X, Y_real):
        if X is None or Y_real is None:
            return 1, 0
        Y_pred_prob = self.predict(X)
        Y_pred_classes = prob_to_class(Y_pred_prob)
        acc = accuracy(Y_pred_classes, Y_real)
        loss = self.__loss(Y_pred_prob, Y_real)
        return acc, loss

    def __cross_entropy(self, Y_pred, Y_real):
        return -np.sum(np.log(np.sum(np.multiply(Y_pred, Y_real), axis=0)))

    def __loss(self, Y_pred_prob, Y_real):
        if self.loss_type == "cross_entropy":
            return self.__cross_entropy(Y_pred_prob, Y_real)
        return None

    def __loss_differential(self, Y_pred, Y_real):
        if self.loss_type == "cross_entropy":
            # d (-log(x))/dx = -1/x 
            f_y = np.multiply(Y_real, Y_pred)
            loss_diff = -np.reciprocal(f_y, out=np.zeros_like(Y_pred), where=abs(f_y)>0.000001)  # Element-wise inverse
            return loss_diff
            # G = - (Y_real - Y_pred)
            # return G*X.T/X.shape[1]; 
        return None

    def __cost(self, Y_pred_prob, Y_real):
        """Computes cost = loss + regularization"""
        # Loss
        loss = 0
        if self.loss_type == "cross_entropy":
            loss = self.__cross_entropy(Y_pred_prob, Y_real)

        # Regularization
        w_norm = 0
        for layer in self.layers:
            w_norm += np.linalg.norm(layer.weights, 'fro')**2

        return loss/Y_pred_prob.shape[1] + self.reg_term*w_norm

    def fit(self, X, Y, X_val=None, Y_val=None, batch_size=None, epochs=100, lr=0.01, momentum=0.7, l2_reg=0.1):
        if (batch_size is None) or (batch_size > X.shape[1]):
            batch_size = X.shape[1]

        # Learning tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        for epoch in tqdm(range(epochs)):
            indx = list(range(X.shape[1])) 
            np.random.shuffle(indx)
            for i in range(int(X.shape[1]/batch_size)):  # Missing last X.shape[1]%batch_size but should be ok
                # Get minibatch
                X_minibatch = X[:, indx[i:i+batch_size]]
                Y_minibatch = Y[:, indx[i:i+batch_size]]
                
                # Forward pass
                Y_pred_prob = self.predict(X_minibatch)

                # Backprop
                gradient = self.__loss_differential(Y_pred_prob, Y_minibatch)  # First error id (D loss)/(D weight)
                for layer in reversed(self.layers):  # Next errors given by each layer weights
                    gradient = layer.backward(
                                    in_gradient=gradient,
                                    lr=lr,
                                    momentum=momentum,
                                    l2_regularization=l2_reg)
                
            # Error tracking:
            train_acc, train_loss = self.get_metrics(X, Y)
            val_acc, val_loss = self.get_metrics(X_val, Y_val)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    def plot_training_progress(self, show=True, save=False, name="model_results"):
        fig, ax1 = plt.subplots()
        # Losses
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Losses")
        ax1.plot(list(range(len(self.train_losses))),
                 self.train_losses, label="Train loss", c="orange")
        if len(self.val_losses) > 0:
            ax1.plot(list(range(len(self.val_losses))),
                     self.val_losses, label="Val loss", c="red")
        ax1.tick_params(axis='y')
        plt.legend(loc='upper right')
        
        # Accuracies
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracies")
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
        plt.legend(loc='center right')
        
        if save:
            plt.savefig("figures/" + name + ".png")
        if show:
            plt.show()
