import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mlp.layers import Dense, Softmax, Relu
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import accuracy
from mlp.utils import LoadXY

np.random.seed(1)

# TODO(oleguer): Update this example to newest refactor

if __name__ == "__main__":
    # Download & Extract CIFAR-10 Python (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    # Put it in a data folder

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

    # Define model
    model = Sequential(loss=CrossEntropy())
    model.add(Dense(nodes=10, input_dim=x_train.shape[0]))
    model.add(Softmax())

    # Fit model
#     model.load("models/mlp_test")
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
                        batch_size=100, epochs=40, lr=0.001, momentum=0.0,
                        l2_reg=0.0, shuffle_minibatch=False)
    model.plot_training_progress(save=True, name="figures/mlp_test")
    model.save("models/mlp_test")

    # Test model
    test_acc, test_loss = model.get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)
