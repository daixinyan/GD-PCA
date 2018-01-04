from tool.numpy import *
import random
import time


def vr_pca(X, eigen_vector, k, out_epochs, inner_epochs, log=None, eta=0.005, w_snapshot=None):
    (d, n) = np.shape(X)
    m = inner_epochs*n
    start = time.clock()
    start = time.clock()
    oracle = 0

    if w_snapshot is None:
        print('===========initial w_run============')
        w_snapshot = np.random.normal(size=[d, k])
    w_run = w_snapshot

    for i in range(out_epochs):
        u = (1.0/n)*np.dot(X, np.dot(np.transpose(X), w_snapshot))
        oracle += n
        for j in range(m):
            U_w, S_w, V_w = np.linalg.svd(np.matmul(np.transpose(w_run), w_snapshot))
            B = np.dot(V_w, np.transpose(U_w))

            x = X[:, random.randint(0, n-1)]
            multiply = np.dot(np.transpose(x), w_run - np.dot(w_snapshot, B))
            deviation = np.multiply(x.reshape([-1, 1]), multiply.reshape([1, -1]))
            w_run = w_run + eta * (deviation + np.dot(u, B))
            w_run = gram_schmidt_columns(w_run)
            oracle += 2
            # w_run = np.dot(w_run, square_matrix_power(norm(w_run), -0.5))

        w_snapshot = w_run

        def x_dot_vector(vector):
            return np.linalg.norm(np.matmul(np.transpose(X), vector))

        percent = (x_dot_vector(w_snapshot) / x_dot_vector(eigen_vector))
        loss = 0 if percent >= 1.0 else np.log10(1 - percent ** 2)

        if loss != 0:
            elapsed = (time.clock() - start)
            log(elapsed, oracle, loss)
        if i % (out_epochs/5+1) == 1:
            print('epoch %d with loss %f' % (i+1, loss))
            print(np.diag(norm(np.dot(np.transpose(X), w_snapshot))))

    print(np.diag(norm(np.dot(np.transpose(X), w_snapshot))))

    return w_run




