# Deep Learning Framework Playground

Simple Keras-inspired Deep Learning Framework implemented in Python with Numpy backend (using hand-written gradients) and MatPlotLib for plotting. For efficient multithreaded Einstein summation between tensors I use [einsum2 repo](https://github.com/jackkamm/einsum2).

As all my other repos, this is more an exercise for me to make sure I understand the main Deep Learning architectures and algorithms, rather than useful code to fit models. 
As well as a way to think of (relatively) efficient implementation of them.
Hope this (super) simplified "Keras" re-implementation also helps you understand them!

# Architectures:

## Multi Layer Perceptron (MLP)

Allows to Build, Train and Assess a modular Multi-Layer-Perceptron Squential architecture as you would do using Keras.
The model (as for now) presents the following features:

- **Layers:**
    - Trainable: Dense, Conv2D
    - Activation: Relu, Softmax
    - Regularization: Dropout
- **Losses:**
    - CrossEntropy
    - CategoricalHinge
- **Optimization:** Minibatch Gradient Descent BackProp Training with customizable:
    - Batch Size
    - Epochs / Iterations
    - Momentum
    - L2 Regularization Term
- **Callbacks:**
    - Learning Rate Scheduler: Constant, Linear, Cyclic
    - Loss & Metrics tracker
    - Early Stopper


Code Example:
```python
# Imports
from mlp.callbacks import MetricTracker, LearningRateScheduler
from mlp.layers import Dense, Softmax, Relu, Dropout
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import Accuracy

# Define model
model = Sequential(loss=CrossEntropy(), metric=Accuracy())
model.add(Dense(nodes=800, input_dim=x_train.shape[0]))
model.add(Relu())
model.add(Dropout(0.8))
model.add(Dense(nodes=10, input_dim=800))
model.add(Softmax())

# Define callbacks
mt = MetricTracker()  # Stores training evolution info (losses and metrics)
lrs = LearningRateScheduler(evolution="cyclic", lr_min=1e-5, lr_max=1e-1)
callbacks = [mt, lrs]

# Fit model
model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
        batch_size=100, epochs=100, l2_reg=0.01, momentum=0.1,
        callbacks=callbacks)
mt.plot_training_progress()

# Test model
test_acc, test_loss = model.get_metric_loss(x_test, y_test)
print("Test accuracy:", test_acc)
```

Example of metrics tracked during training:
![Training tracking](https://github.com/OleguerCanal/KTH_DeepLearning/blob/master/Assignment_2/figures/best.png)


**NOTE:** More architectures, layers and features (CNN, RBF, SOM, DBF) comming soon

# Utilities

## Meta-Parameter Optimization (MPO)

Metaparameter Optimization is commonly used when training these kind of models. To ease the process I implemented a MetaParamOptimizer class with methods such as Grid Search, additionally I jointly wrote a wrapper around [scikit-optimize](https://scikit-optimize.github.io/stable/) with [Federico Taschin](https://github.com/fedetask), to perform Bayesian Optimization [here](https://github.com/CampusAI/HyperParameter-Optimizer)).

1. Define the **search space** and **fixed args** and a of your model in two diferent dictionaries
2. Define an **evaluator** function which trains and evaluates your model in joined arguments, this function should return a ``dictionary`` with at least the key **"value"** (which MetaParamOptimizer will optimize).

Code example:
```python
from mpo.metaparamoptimizer import MetaParamOptimizer
from util.misc import dict_to_string

search_space = {  # Optimization will be performed on all combinations of these
    "batch_size": [100, 200, 400],     # Batch sizes
    "lr": [0.001, 0.01, 0.1],          # Learning rates
    "l2_reg": [0.01, 0.1]              # L2 Regularization terms
}
fixed_args = {  # These will be kept constant
    "x_train" : x_train,
    "y_train" : y_train,
    "x_val" : x_val,
    "y_val" : y_val,
    "epochs" : 100,
    "momentum" : 0.1,
}

def evaluator(x_train, y_train, x_val, y_val, **kwargs):
    # Define model
    model = Sequential(loss=CrossEntropy())
    model.add(Dense(nodes=10, input_dim=x_train.shape[0]))
    model.add(Softmax())

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val, **kwargs)
    model.plot_training_progress(show=False, save=True, name="figures/" + dict_to_string(kwargs)
    model.save("models/" + dict_to_string(kwargs))

    # Evaluator result (add model to retain best)
    value = model.get_classification_metrics(x_val, y_val)[0] # Get accuracy
    result = {"value": value, "model": model}  # MetaParamOptimizer will maximize value
    return result

# Get best model and best prams
mpo = MetaParamOptimizer(save_path="models/")
best_model = mpo.grid_search(evaluator=evaluator,
                            search_space=search_space,
                            fixed_args=fixed_args)
# This will run your evaluator function 3x3x3 = 27 times on all combinations of search_space params
```

Example of Gaussian Process Regression Optimizer hyperparameter analysis:

![Gaussian Process Regression Optimizer Analysis](https://github.com/OleguerCanal/KTH_DeepLearning/blob/master/Assignment_2/metaparam_search/evaluations.csv_objective_plot.png)


# Usage

Clone repo and install requirements:

`git clone https://github.com/OleguerCanal/Toy-DeepLearning-Framework.git`

`cd Toy-DeepLearning-Framework`

`pip install -r requirements.txt`

[OPTIONAL]
To parallelize Einstein sumations between tensors install [einsum2](https://github.com/jackkamm/einsum2), if not found will use numpy single-thread version instead (SLOWER).
