from abc import ABC, abstractmethod
import copy
import numpy as np
from matplotlib import pyplot as plt

import sys
import pathlib
import os
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
from models import Sequential
from mlp.utils import generate_sequence

class Callback(ABC):
    """ Abstract class to hold callbacks
    """

    def on_training_begin(self, model):
        pass

    def on_batch_end(self, model, **kwargs):
        pass

    # @abstractmethod
    def on_epoch_end(self, model, **kwargs):
        """ Getss called at the end of each epoch
            Can modify model training variables (eg: LR Scheduler)
            Can store information to be retrieved afterwards (eg: Metric Tracker)
        """
        pass


# INFORMATION STORING CALLBACKS ######################################################
class MetricTracker(Callback):
    """ Tracks training metrics to plot and save afterwards
    """

    def __init__(self, file_name="models/tracker", frequency=1):
        self.file_name = file_name
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []
        self.frequency = frequency
        self.iteration = 0

    def on_training_begin(self, model):
        self.metric_name = model.metric.name
        # self.__track(model)

    def on_batch_end(self, model, Y_real=None):
        if (self.iteration%self.frequency == 0):
            self.__batch_track(model, Y_real)
            self.save(self.file_name)
        self.iteration += 1
        pass

    def on_epoch_end(self, model):
        # self.__track(model)
        # self.save(self.file_name)
        pass

    def __batch_track(self, model, Y_real=None):
        if Y_real is None:
            return
        metric = model.metric(model.Y_pred_prob, Y_real)
        loss = model.loss(model.Y_pred_prob, Y_real)
        if self.iteration == 0:
            self.metric = metric
            self.loss = loss

        self.metric = 0.99*self.metric + 0.01*metric
        self.loss = 0.99*self.loss + 0.01*loss

        self.train_metrics.append(self.metric)
        self.train_losses.append(self.loss)
        model.train_metric = self.metric
        model.train_loss = self.loss

    def __track(self, model):
        train_metric, train_loss = model.get_metric_loss(model.X, model.Y, use_dropout=False)
        self.train_losses.append(train_loss)
        self.train_metrics.append(train_metric)
        self.learning_rates.append(model.lr)
        model.train_metric = train_metric
        model.train_loss = train_loss

        if model.X_val is not None and model.Y_val is not None:
            val_metric, val_loss = model.get_metric_loss(model.X_val, model.Y_val, use_dropout=False)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metric)
            model.val_metric = val_metric

    def plot_training_progress(self, show=True, save=False, name="model_results", subtitle=None):
        fig, ax1 = plt.subplots()
        # Losses
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_ylim(bottom=0)
        ax1.set_ylim(top=1.25*np.clip(np.nanmax(self.train_losses), 0, 100))
        if self.val_losses is not None:
            if len(self.val_losses) > 0:
                ax1.plot(list(range(len(self.val_losses))),
                        self.val_losses, label="Val loss", c="red")
        ax1.plot(list(range(len(self.train_losses))),
                 self.train_losses, label="Train loss", c="orange")
        ax1.tick_params(axis='y')
        plt.legend(loc='upper left')

        # Accuracies
        ax2 = ax1.twinx()
        ax2.set_ylabel(self.metric_name)
        ax2.set_ylim(bottom=0)
        ax2.set_ylim(top=1.2)
        n = len(self.train_metrics)
        ax2.plot(list(range(n)),
                 np.array(self.train_metrics), label="Train acc", c="green")
        if self.val_metrics is not None:
            if len(self.val_metrics) > 0:
                n = len(self.val_metrics)
                ax2.plot(list(range(n)),
                        np.array(self.val_metrics), label="Val acc", c="blue")
            ax2.tick_params(axis='y')

        # plt.tight_layout()
        plt.suptitle("Training Evolution")
        if subtitle is not None:
            plt.title(subtitle)
        plt.legend(loc='upper right')

        if save:
            directory = "/".join(name.split("/")[:-1])
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            print(name + ".png")
            plt.savefig(name + ".png")
            plt.close()
        if show:
            plt.show()

    def plot_lr_evolution(self, show=True, save=False, name="lr_evolution", subtitle=None):
        plt.suptitle("Learning rate evolution")
        plt.plot(self.learning_rates, label="Learning rate")
        plt.legend(loc='upper right')
        plt.xlabel("Iteration")
        if save:
            directory = "/".join(name.split("/")[:-1])
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            plt.savefig(name + ".png")
            plt.close()
        if show:
            plt.show()

    def plot_acc_vs_lr(self, show=True, save=False, name="acc_vs_lr", subtitle=None):
        plt.suptitle("Accuracy evolution for each learning rate")
        plt.plot(np.log(self.learning_rates), self.train_metrics, label="Accuracy")
        plt.legend(loc='upper right')
        plt.xlabel("Learning Rate")
        plt.ylabel("Train Accuracy")
        if save:
            directory = "/".join(name.split("/")[:-1])
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            plt.savefig(name + ".png")
            plt.close()
        if show:
            plt.show()

    def save(self, file):
        directory = "/".join(file.split("/")[:-1])
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        # np.save(file + "_lr", self.learning_rates)
        np.save(file + "_train_met", self.train_metrics)
        np.save(file + "_val_met", self.val_metrics)
        np.save(file + "_train_loss", self.train_losses)
        np.save(file + "_val_loss", self.val_losses)

    def load(self, file):
        self.metric_name = "Accuracy"
        self.train_metrics = np.load(file + "_train_met.npy").tolist()
        self.val_metrics = np.load(file + "_val_met.npy").tolist()
        self.train_losses = np.load(file + "_train_loss.npy").tolist()
        self.val_losses = np.load(file + "_val_loss.npy").tolist()


class BestModelSaver(Callback):
    def __init__(self, save_dir=None):
        self.save_dir = None
        if save_dir is not None:
            self.save_dir = os.path.join(save_dir, "best_model")
        self.best_metric = -np.inf
        self.best_model_layers = None
        self.best_model_loss = None
        self.best_model_metric = None

    def on_epoch_end(self, model):
        val_metric = model.get_metric_loss(model.X_val, model.Y_val)[0]
        if val_metric >= self.best_metric:
            self.best_metric = model.val_metric
            self.best_model_layers = copy.deepcopy(model.layers)
            self.best_model_loss = copy.deepcopy(model.loss)
            self.best_model_metric = copy.deepcopy(model.metric)
            if self.save_dir is not None:
                model.save(self.save_dir)

    def get_best_model(self):
        best_model = Sequential(loss=self.best_model_loss,
                                metric=self.best_model_metric)
        best_model.layers = self.best_model_layers
        return best_model


class TextSynthesiser(Callback):
    def __init__(self, ind_to_char, char_to_ind, first_letter='H', seq_length=200, frequency=50000, file_name="models/TextSynthesiser"):
        self.iteration = 0
        self.file = file_name
        self.seq_length = seq_length
        self.ind_to_char = ind_to_char
        self.char_to_ind = char_to_ind
        self.frequency = frequency
        self.first_letter = first_letter

        # Create folder if does not exist
        directory = "/".join(file_name.split("/")[:-1])
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    def on_batch_end(self, model, **kwargs):
        if self.iteration%self.frequency == 0:
            layer = copy.deepcopy(model.layers[0])
            string = generate_sequence(layer, self.first_letter, self.ind_to_char, self.char_to_ind, length=self.seq_length)
            file_name = self.file + "_" +  str(self.iteration) + ".txt"
            with open(file_name, "w") as text_file:
                text_file.write(string)
        self.iteration += 1

# LEARNING PARAMS MODIFIER CALLBACKS ######################################################

class LearningRateScheduler(Callback):
    def __init__(self, evolution="linear", lr_min=None, lr_max=None, ns=500):
        assert(evolution in ["constant", "linear", "cyclic"])
        self.type = evolution
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.ns = ns

    def on_training_begin(self, model):
        if self.type == "cyclic":
            model.lr = self.lr_min

    def on_batch_end(self, model):
        if self.type == "cyclic":
            slope = int(model.t/self.ns)%2
            lr_dif = float(self.lr_max - self.lr_min)
            if slope == 0:
                model.lr = self.lr_min + float(model.t%self.ns)*lr_dif/float(self.ns)
            if slope == 1:
                model.lr = self.lr_max - float(model.t%self.ns)*lr_dif/float(self.ns)
        if self.type == "linear":
            lr_dif = float(self.lr_max - self.lr_min)
            model.lr = self.lr_min + float(model.t)*lr_dif/float(self.ns)

    def on_epoch_end(self, model):
        if self.type == "constant":
            pass
        elif self.type == "linear":
            model.lr = 0.9*model.lr