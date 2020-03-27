# Deep Learning Framework Playground

Simple Keras-inspired Deep Learning Framework implemented in Python with Numpy backend.

As all my other repos, this is more an exercise for me to make sure I understand the main Deep Learning architectures and algorithms, rather than useful code.
Hope it also helps you understand them!

# Architectures:

## Multi Layer Perceptron (MLP)

Allows to Build, Train and Assess a modular Multi-Layer-Perceptron Squential architecture as you would do using Keras.
The model now presents the following features:

- **Layers:**
    - Trainable: Dense (Fully connected)
    - Activation: Relu, SoftMax
- **Losses:**
    - Cross-Entropy Loss
    - Categorical Hinge Loss (SVM MultiClass)
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

