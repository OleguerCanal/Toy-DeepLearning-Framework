import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from mlp.utils import getXY, LoadBatch, prob_to_class
from mlp.layers import Activation, Dense
from mlp.models import Sequential
from mpo.metaparamoptimizer import MetaParamOptimizer
from util.misc import dict_to_string

import numpy as np

np.random.seed(0)

# Define evaluator (function to run in MetaParamOptimizer)
def evaluator(x_train, y_train, x_val, y_val, experiment_path="", **kwargs):
    # Define model
    model = Sequential(loss="cross_entropy")
    model.add(
        Dense(nodes=10, input_dim=x_train.shape[0], weight_initialization="fixed"))
    model.add(Activation("softmax"))

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val, **kwargs)
    model.plot_training_progress(show=False, save=True, name="figures/" + dict_to_string(kwargs))
    model.save(experiment_path + "/" + dict_to_string(kwargs))

    # Minimizing value: validation accuracy
    val_acc = model.get_classification_metrics(x_val, y_val)[0] # Get accuracy
    result = {"value": val_acc, "model": model}  # Save score and model
    return result

if __name__ == "__main__":
    # Download & Extract CIFAR-10 Python (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    # Put it in a Data folder

    # Load data
    x_train, y_train = LoadXY("data_batch_1")
    x_val, y_val = LoadXY("data_batch_2")
    x_test, y_test = LoadXY("test_batch")

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    # Define list of parameters to try
    dicts_list = [
        { "l2_reg": 0.0, "lr": 0.1 },
        { "l2_reg": 0.0, "lr": 0.001 },
        { "l2_reg": 0.1, "lr": 0.001 },
        { "l2_reg": 1.0, "lr": 0.001 },
    ]
    # Define fixed params (constant through optimization)
    fixed_args = {
        "experiment_path" : "models/list_search/",
        "x_train" : x_train,
        "y_train" : y_train,
        "x_val" : x_val,
        "y_val" : y_val,
        "batch_size": 100,
        "epochs" : 40,
        "momentum" : 0.7,
    }
    # NOTE: The union of both dictionaries should contain all evaluator parameters

    # Perform optimization
    mpo = MetaParamOptimizer(save_path=fixed_args["experiment_path"])
    best_model = mpo.list_search(evaluator=evaluator,
                                dicts_list=dicts_list,
                                fixed_args=fixed_args)

    # Test model
    test_acc, test_loss = best_model["model"].get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)
