from pca.sgd import sgd_pca
from pca.vr import vr_pca
from pca.scsg import scsg_pca
from sklearn import datasets
from pylab import plot
from pylab import show
from pylab import legend
from pylab import title
from pylab import grid
import numpy as np
import pickle
import os
from tool import mnist


# data = datasets.load_diabetes().data
data, _ = mnist.load_data(mnist.TRAINING_SET, 9216)
data = np.array(data).transpose()

U, S, V = np.linalg.svd(data)
features, numbers = np.shape(data)
print('features, numbers ', np.shape(data))
print('singular value: ', S**2)

k = 3
out_epochs = 200
inner_epochs = 2
log_file = 'log_file.txt'
eta = 0.001


def log(time, oracle, loss, time_line, oracle_line, loss_line):
    time_line.append(time)
    oracle_line.append(oracle)
    loss_line.append(loss)


def svrg(X):
    print('---------------------------------------------------')
    print('----------------------svrg-------------------------')
    time_line, oracle_line, loss_line = [], [], []

    def log_svrg(time, oracle, loss):
        log(time, oracle, loss, time_line, oracle_line, loss_line)
    vr_pca(X, U[:, 0:k], k, out_epochs, inner_epochs, log_svrg, eta=eta)
    return time_line, oracle_line, loss_line


def scsg(X):
    batch_size = int(2048)
    mini_batch = 64
    print('---------------------------------------------------')
    print('-----------------------scsg-------------------------')
    time_line, oracle_line, loss_line = [], [], []

    def log_scsg(time, oracle, loss):
        log(time, oracle, loss, time_line, oracle_line, loss_line)
    scsg_pca(
        X=X,
        eigen_vector=U[:, 0:k],
        k=k,
        out_epochs=int(out_epochs*inner_epochs*numbers/batch_size),
        batch_size=batch_size,
        mini_batch=mini_batch,
        log=log_scsg,
        eta=eta
    )
    return time_line, oracle_line, loss_line


def sgd(X):
    print('---------------------------------------------------')
    print('-----------------------sgd-------------------------')
    time_line, oracle_line, loss_line = [], [], []

    def log_sgd(time, oracle, loss):
        log(time, oracle, loss, time_line, oracle_line, loss_line)
    batch_size = 1
    sgd_pca(X, U[:, 0:k], k, int(out_epochs*inner_epochs/batch_size), batch_size, log_sgd, eta=eta)
    return time_line, oracle_line, loss_line


def scsg_draw(log_data, oracle_x):
    time_line, oracle_line, loss_line = log_data
    if oracle_x:
        plot(oracle_line, loss_line, color='blue', linewidth=2.5, linestyle="-", label='scsg_oracle')
    else:
        plot(time_line, loss_line, color='blue', linewidth=3, linestyle="-", label='scsg_time')


def svrg_draw(log_data, oracle_x):
    time_line, oracle_line, loss_line = log_data
    if oracle_x:
        plot(oracle_line, loss_line, color='yellow', linewidth=2.5, linestyle="-", label='svrg_oracel')
    else:
        plot(time_line, loss_line, color='yellow', linewidth=3, linestyle="-", label='svrg_time')


def sgd_draw(log_data, oracle_x):
    time_line, oracle_line, loss_line = log_data
    if oracle_x:
        plot(oracle_line, loss_line, color='red', linewidth=2.5, linestyle="-", label='sgd_oracle')
    else:
        plot(time_line, loss_line, color='red', linewidth=3, linestyle="-", label='sgd_time')


def draw(file, oracle_x):
    f = open(file, 'rb')
    sgd_data, svrg_data, scsg_data = pickle.load(f)

    sgd_draw(sgd_data, oracle_x)
    svrg_draw(svrg_data, oracle_x)
    scsg_draw(scsg_data, oracle_x)
    title('loss wrt oracle' if oracle_x else 'loss wrt time')
    legend(loc='upper right')
    grid(True)
    show()


def start_pca():
    svrg_log = svrg(data)
    scsg_log = scsg(data)
    sgd_log = sgd(data)
    f = open(log_file, 'wb')
    pickle.dump((sgd_log, svrg_log, scsg_log), f)


if not os.path.isfile(log_file):
    start_pca()

draw(log_file, True)
draw(log_file, False)


