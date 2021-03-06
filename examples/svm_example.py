import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mlp.models import Sequential
from mlp.layers import Activation, Dense
from mlp.utils import getXY, LoadBatch, prob_to_class

np.random.seed(0)

if __name__ == "__main__":
    # Download & Extract CIFAR-10 Python (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    # Put it in a Data folder

    # Load data
    x_train, y_train = getXY(LoadBatch("data_batch_1"))
    x_val, y_val = getXY(LoadBatch("data_batch_2"))
    x_test, y_test = getXY(LoadBatch("test_batch"))

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    # Define SVM multi-class model
    model = Sequential(loss="categorical_hinge")
    model.add(Dense(nodes=10, input_dim=x_train.shape[0]))
    model.add(Activation("softmax"))

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
            batch_size=100, epochs=100, lr=0.0001, momentum=0.1, l2_reg=0.1)
    model.plot_training_progress()

    # Test model
    test_acc, test_loss = model.get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)