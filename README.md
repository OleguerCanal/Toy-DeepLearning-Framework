# DeepLearning Framework PlayGround

Simple Keras-inspired DeepLearning Framework Python implementation using Numpy backend.
As all my other repos, this is more an exercise for me to make sure I understand the maim Deep Learning architectures and algorithms, rather than useful code.
Hope it also helps you understand them!

This repo contains:

## MLP

You can build, train and assess your (simple) Multi-Layer-Perceptron Squential architecture as you would do in Keras.
The model now presents the following features:

- **Layers:**
    - Dense (Fully connected)
    - Activation: Relu/SoftMax
- **Lossess:**
    - Cross-Entropy
    - SVM MultiClass
- **Minibatch Gradient Descent BackProp Training** with customizable:
    - Learning Rate
    - Momentum
    - Regularization
    - Train/Val Loss (& accuracy) tracking


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
