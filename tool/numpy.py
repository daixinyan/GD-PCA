import numpy as np


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def norm(vec):
    return np.dot(np.transpose(vec), vec)


def square_matrix_power(X, power):
    P, S, invert_P = np.linalg.svd(X)
    result = np.dot(np.dot(P, np.diag(S**power)), np.transpose(invert_P))
    return result
