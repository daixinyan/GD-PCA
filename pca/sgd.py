from tool.numpy import *
import random
import time


def sgd_pca(X, eigen_vector, k, epochs, batch_size=1, log=None, eta=0.005, w_run=None):
    (d, n) = np.shape(X)
    start = time.clock()
    start = time.clock()
    oracle = 0

    if w_run is None:
        w_run = np.random.normal(size=[d, k])

    for i in range(epochs):

        for j in range(n):
            def update_deviation(size):
                nonlocal oracle
                result = None
                for i in range(size):
                    x = X[:, random.randint(0, n - 1)]
                    multiply = np.dot(np.transpose(x), w_run)
                    deviation = np.multiply(x.reshape([-1, 1]), multiply.reshape([1, -1]))
                    result = deviation if result is None else deviation + result
                    oracle = oracle + 1
                return result / size

            w_run = w_run + eta * update_deviation(batch_size)
            w_run = gram_schmidt_columns(w_run)
            # w_run = np.dot(w_run, square_matrix_power(norm(w_run), -0.5))

        w_snapshot = w_run

        def x_dot_vector(vector):
            return np.linalg.norm(np.matmul(np.transpose(X), vector))

        percent = (x_dot_vector(w_snapshot) / x_dot_vector(eigen_vector))
        loss = 0 if percent >= 1.0 else np.log10(1 - percent ** 2)
        if loss != 0:
            elapsed = (time.clock() - start)
            log(elapsed, oracle, loss)
        if i % int(epochs/5+1) == 1:
            print('epoch %d with loss %f' % (i+1, loss))
            print(np.diag(norm(np.dot(np.transpose(X), w_snapshot))))

    print(np.diag(norm(np.dot(np.transpose(X), w_snapshot))))

    return w_run