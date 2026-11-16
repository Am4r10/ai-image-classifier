import numpy as np


# ---------- Plain functions ----------

def relu(z):
    return np.maximum(0, z)                          # negatives become 0


def relu_derivative(z):
    return (z > 0).astype(float)                     # slope 1 where z > 0, slope 0 on the flat part


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))             # safe: exp of a negative number never overflows
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1 + exp_z)                  # same formula rearranged for negative z
    return out


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)                               # sigma'(z) = sigma(z)(1 - sigma(z))


def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True) # subtract each column's max (stops overflow)
    exp_z = np.exp(z_shifted)                        # e to the power of each score
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)  # divide by column total so each column sums to 1


def softmax_backward(A, dA):
    # Softmax mixes a whole column, so its derivative is a Jacobian matrix J = diag(a) - a a^T.
    # J @ dA simplifies to a * (dA - sum(a * dA)), computed per column without building J.
    return A * (dA - np.sum(A * dA, axis=0, keepdims=True))


# ---------- Classes used by Layer ----------
# forward(Z) gives A.  backward(Z, A, dA) turns dL/dA into dL/dZ (the chain rule step).

class ReLU:
    name = "relu"

    def forward(self, Z):
        return relu(Z)

    def backward(self, Z, A, dA):
        return dA * relu_derivative(Z)               # element-wise: each z only affects its own a


class Sigmoid:
    name = "sigmoid"

    def forward(self, Z):
        return sigmoid(Z)

    def backward(self, Z, A, dA):
        return dA * A * (1 - A)                      # reuse A instead of recomputing sigmoid(Z)


class Softmax:
    name = "softmax"

    def forward(self, Z):
        return softmax(Z)

    def backward(self, Z, A, dA):
        return softmax_backward(A, dA)


class Linear:
    name = "linear"

    def forward(self, Z):
        return Z                                     # no activation (used for regression outputs)

    def backward(self, Z, A, dA):
        return dA


ACTIVATIONS = {"relu": ReLU, "sigmoid": Sigmoid, "softmax": Softmax, "linear": Linear}


def get_activation(name):
    return ACTIVATIONS[name]()                       # look up an activation by name
