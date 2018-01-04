from tool.numpy import *
import random
import time


def scsg_pca(X, eigen_vector, k, out_epochs, batch_size, mini_batch, log=None, eta=0.005):
    (features, numbers) = np.shape(X)
    start = time.clock()
    start = time.clock()
    oracle = 0

    def x_dot_vector(vector):
        return np.linalg.norm(np.matmul(np.transpose(X), vector))
    index = 0

    def next_batch(size):
        return X[:, np.random.randint(0, numbers - 1, size)]
        # nonlocal index
        # result = X[:, index:index + size]
        # if index + size >= numbers:
        #     index = index + size - numbers
        #     if index != 0:
        #         result = np.concatenate((result, X[:, 0:index]), axis=1)
        # else:
        #     index = index + size
        # return result

    w_snapshot = np.random.normal(size=[features, k])
    for i in range(out_epochs):
        w_snapshot, calculate_time = _scsg_pca(
            X=next_batch(batch_size),
            mini_batch=mini_batch,
            eta=eta,
            w_snapshot=w_snapshot
        )
        percent = (x_dot_vector(w_snapshot) / x_dot_vector(eigen_vector))
        loss = 0 if percent >= 1.0 else np.log10(1 - percent ** 2)
        oracle = oracle + calculate_time
        if loss != 0:
            elapsed = (time.clock() - start)
            log(elapsed, oracle, loss)
        if i % int(out_epochs/5+1) == 1:
            print('epoch %d with loss %f' % (i, loss))
            print(np.diag(norm(np.dot(np.transpose(X), w_snapshot))))
    print(np.diag(norm(np.dot(np.transpose(X), w_snapshot))))


def _scsg_pca(X, mini_batch, eta, w_snapshot):
    (d, batch_size) = np.shape(X)
    oracle = 0
    w_run = w_snapshot

    u = (1.0 / batch_size) * np.dot(X, np.dot(np.transpose(X), w_snapshot))
    oracle += batch_size
    m = np.random.geometric(mini_batch / (mini_batch + batch_size), 1)[0]
    # m = int((mini_batch+batch_size)/mini_batch)

    for j in range(m):
        U_w, S_w, V_w = np.linalg.svd(np.matmul(np.transpose(w_run), w_snapshot))
        B = np.dot(V_w, np.transpose(U_w))

        def update_deviation(size):
            nonlocal oracle
            result = None
            for i in range(size):
                x = X[:, random.randint(0, batch_size - 1)]
                multiply = np.dot(np.transpose(x), w_run - np.dot(w_snapshot, B))
                deviation = np.multiply(x.reshape([-1, 1]), multiply.reshape([1, -1]))
                result = deviation if result is None else deviation + result
                oracle += 1
            return result/size

        w_run = w_run + eta * (update_deviation(mini_batch) + np.dot(u, B))
        w_run = gram_schmidt_columns(w_run)
        # w_run = np.dot(w_run, square_matrix_power(norm(w_run), -0.5))

    w_snapshot = w_run

    return w_snapshot, oracle




