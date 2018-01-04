import numpy as np
import random

TRAINING_SET = "./mnist/mnist.scale"
TEST_SET = "./mnist/mnist.scale.t"
DATA_SAMPLES = 60000
TEST_SAMPLES = 10000
FEATURES = 780
CLASSES = 10


# load libsvm dataset
def load_data(filename, n_samples):
    data = np.empty((n_samples, FEATURES))
    target = np.empty((n_samples,), dtype=np.int)
    i = 0
    for line in open(filename):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        xi = [0] * (FEATURES)
        for e in features.split():
            ind, val = e.split(":")
            xi[int(ind) - 1] = float(val)
        target[i] = np.asarray(int(label) - 1, dtype=np.int)
        data[i] = np.asarray(xi[0:FEATURES], dtype=np.float64)
        i += 1
        if i >= n_samples:
            break
    return data, target


# output batch data
def get_batch_data(x_train, y_train):
    size = len(x_train)
    batch_xs = x_train
    batch_ys = []
    # convert to 1-of-N vector
    for i in range(size):
        val = np.zeros((CLASSES), dtype=np.float64)
        val[y_train[i]] = 1.0
        batch_ys.append(val)
    batch_ys = np.asarray(batch_ys)
    return batch_xs, batch_ys


# output mini_batch data
def get_stoc_batch_data(x_train, y_train, size=None):
    if size is None:
        size = len(x_train)
    batch_xs = x_train
    batch_ys = []
    # convert to 1-of-N vector
    for i in range(len(y_train)):
        val = np.zeros((CLASSES), dtype=np.float64)
        val[y_train[i]] = 1.0
        batch_ys.append(val)
    batch_ys = np.asarray(batch_ys)
    stochastic_indexes = sorted(random.sample(range(len(y_train)), size))
    mini_batch_xs = [batch_xs[i] for i in stochastic_indexes]
    mini_batch_ys = [batch_ys[i] for i in stochastic_indexes]
    return mini_batch_xs, mini_batch_ys


# output test data
def get_test_data(x_test, y_test):
    batch_ys = []

    # convert to 1-of-N vector
    for i in range(len(y_test)):
        val = np.zeros((CLASSES), dtype=np.float64)
        val[y_test[i]] = 1.0
        batch_ys.append(val)
    return x_test, np.asarray(batch_ys)



