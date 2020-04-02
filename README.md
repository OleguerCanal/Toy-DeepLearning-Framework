# Deep Learning Framework Playground

Simple Keras-inspired Deep Learning Framework implemented in Python with Numpy backend.

As all my other repos, this is more an exercise for me to make sure I understand the main Deep Learning architectures and algorithms, rather than useful code.
Hope it also helps you understand them!

# Architectures:

## Multi Layer Perceptron (MLP)

Allows to Build, Train and Assess a modular Multi-Layer-Perceptron Squential architecture as you would do using Keras.
The model (as for now) presents the following features:

- **Layers:**
    - Trainable: Dense
    - Activation: Relu, Softmax
- **Losses:**
    - CrossEntropy
    - CategoricalHinge
- **Optimizer:** Minibatch Gradient Descent BackProp Training with customizable:
    - Batch Size
    - Epochs
    - Learning Rate
    - Momentum
    - L2 Regularization Term
    - Train/Val Loss & accuracy tracking


Code Example:
```python
from mlp.layers import Activation, Dense
from mlp.models import Sequential

# Define model
model = Sequential(loss="cross_entropy")
model.add(Dense(nodes=10, input_dim=x_train.shape[0]))
model.add(Activation("softmax"))

# Fit model
model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
        batch_size=200, epochs=100, lr=0.0001, momentum=0.1, l2_reg=0.1)
model.plot_training_progress()

# Test model
test_acc, test_loss = model.get_metrics(x_test, y_test)
print("Test accuracy:", test_acc)
```

**NOTE:** More architectures and features (RBF, SOM, DBF) comming soon

# Utilities

## Meta-Parameter Optimization (MPO)

Metaparameter Optimization is commonly used when training these kind of models. To ease the process I implemented a MetaParamOptimizer class with methods such as Grid Search (working on Gaussian Process Regression Optimization [here](https://github.com/fedetask/hyperparameter-optimization)):

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
    model = Sequential(loss="cross_entropy")
    model.add(Dense(nodes=10, input_dim=x_train.shape[0]))
    model.add(Activation("softmax"))

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



