import numpy as np
from src.models import Sequential
from src.layers import Activation, Dense
from src.utils import getXY, LoadBatch, prob_to_class

np.random.seed(0)

if __name__ == "__main__":
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