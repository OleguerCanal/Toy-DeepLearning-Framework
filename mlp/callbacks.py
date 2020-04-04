from abc import ABC, abstractmethod
import copy
import numpy as np
from matplotlib import pyplot as plt

import sys
import pathlib
import os
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
from models import Sequential


class Callback(ABC):
    """ Abstract class to hold callbacks
    """

    def on_training_begin(self, model):\
        pass

    @abstractmethod
    def on_epoch_end(self, model):
        """ Getss called at the end of each epoch
            Can modify model training variables (eg: LR Scheduler)
            Can store information to be retrieved afterwards (eg: Metric Tracker)
        """
        pass


# INFORMATION STORING CALLBACKS ######################################################
class MetricTracker(Callback):
    """ Tracks training metrics to plot and save afterwards
    """

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def on_training_begin(self, model):
        self.metric_name = model.metric.name
        self.__track(model)

    def on_epoch_end(self, model):
        self.__track(model)

    def __track(self, model):
        train_metric, train_loss = model.get_metric_loss(model.X, model.Y)
        val_metric, val_loss = model.get_metric_loss(model.X_val, model.Y_val)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_metrics.append(train_metric)
        self.val_metrics.append(val_metric)
        model.val_metric = val_metric

    def plot_training_progress(self, show=True, save=False, name="model_results", subtitle=None):
        fig, ax1 = plt.subplots()
        # Losses
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_ylim(bottom=np.nanmin(self.val_losses)/2)
        ax1.set_ylim(top=1.25*np.nanmax(self.val_losses))
        if len(self.val_losses) > 0:
            ax1.plot(list(range(len(self.val_losses))),
                     self.val_losses, label="Val loss", c="red")
        ax1.plot(list(range(len(self.train_losses))),
                 self.train_losses, label="Train loss", c="orange")
        ax1.tick_params(axis='y')
        plt.legend(loc='center right')

        # Accuracies
        ax2 = ax1.twinx()
        ax2.set_ylabel(self.metric_name)
        ax2.set_ylim(bottom=0)
        ax2.set_ylim(top=1.0)
        n = len(self.train_metrics)
        ax2.plot(list(range(n)),
                 np.array(self.train_metrics), label="Train acc", c="green")
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
            plt.savefig(name + ".png")
            plt.close()
        if show:
            plt.show()


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


# LEARNING PARAMS MODIFIER CALLBACKS ######################################################

class LearningRateScheduler(Callback):
    def __init__(self, evolution="linear"):
        self.type = evolution

    def on_epoch_end(self, model):
        if self.type == "linear":
            model.lr = 0.9*model.lr
