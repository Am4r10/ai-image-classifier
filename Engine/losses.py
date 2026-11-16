import numpy as np


# ---------- Plain functions ----------

def one_hot(labels, num_classes):
    m = labels.shape[0]                    # number of examples
    Y = np.zeros((num_classes, m))         # all zeros, one column per example
    Y[labels, np.arange(m)] = 1            # put a 1 in each column at its label's row
    return Y


def mse(A, Y):
    m = Y.shape[1]                         # number of examples (columns)
    return np.sum((A - Y) ** 2) / m        # squared errors added up, averaged per example


def mse_derivative(A, Y):
    m = Y.shape[1]
    return 2 * (A - Y) / m                 # power rule on (a - y)^2, then the 1/m average


def cross_entropy(A, Y, eps=1e-12):
    m = Y.shape[1]                         # number of examples (columns)
    A = np.clip(A, eps, 1 - eps)           # keep probabilities away from 0 so log never breaks
    return -np.sum(Y * np.log(A)) / m      # -log(prob of correct answer), averaged


def cross_entropy_derivative(A, Y, eps=1e-12):
    m = Y.shape[1]
    A = np.clip(A, eps, 1 - eps)           # same clipping, so we never divide by 0
    return -Y / A / m                      # d/da of -y ln(a) is -y / a


# ---------- Classes used by NeuralNetwork ----------
# forward(A, Y) gives the loss (one number).  backward(A, Y) gives dL/dA, same shape as A.

class MSE:
    name = "mse"

    def forward(self, A, Y):
        return mse(A, Y)

    def backward(self, A, Y):
        return mse_derivative(A, Y)


class CrossEntropy:
    name = "cross_entropy"

    def __init__(self, eps=1e-12):
        self.eps = eps

    def forward(self, A, Y):
        return cross_entropy(A, Y, self.eps)

    def backward(self, A, Y):
        return cross_entropy_derivative(A, Y, self.eps)


LOSSES = {"mse": MSE, "cross_entropy": CrossEntropy}


def get_loss(name):
    return LOSSES[name]()                  # look up a loss by name
