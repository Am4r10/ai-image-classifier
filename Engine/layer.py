import numpy as np

from Engine.activations import get_activation


class Layer:
    """A fully connected (dense) layer: Z = W @ A_prev + b, then A = activation(Z).

    Convention: every example is a COLUMN, so a batch of m examples has shape (features, m).
    """

    def __init__(self, n_in, n_out, activation="relu", rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        self.activation = get_activation(activation)

        # He initialisation for ReLU, Xavier for the rest: keeps the size of the
        # signal roughly constant from layer to layer so training starts smoothly
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = rng.standard_normal((n_out, n_in)) * scale   # (n_out, n_in): one row per neuron
        self.b = np.zeros((n_out, 1))                          # (n_out, 1): broadcast across the batch

        self.dW = np.zeros_like(self.W)                        # gradients, filled in by backward()
        self.db = np.zeros_like(self.b)

    def forward(self, A_prev):
        self.A_prev = A_prev                     # (n_in, m), saved because backward needs it
        self.Z = self.W @ A_prev + self.b        # (n_out, n_in) @ (n_in, m) -> (n_out, m)
        self.A = self.activation.forward(self.Z) # (n_out, m)
        return self.A

    def backward(self, dA):
        """Takes dL/dA for this layer's output, returns dL/dA for the previous layer."""
        dZ = self.activation.backward(self.Z, self.A, dA)   # chain rule through the activation
        return self.backward_from_dZ(dZ)

    def backward_from_dZ(self, dZ):
        """Same as backward(), but starting from dL/dZ (used by the softmax + cross-entropy shortcut)."""
        # dW must match W's shape (n_out, n_in).
        # dZ is (n_out, m) and A_prev is (n_in, m), so A_prev has to be transposed to (m, n_in):
        # (n_out, m) @ (m, n_in) -> (n_out, n_in). The matrix product also sums over the batch.
        self.dW = dZ @ self.A_prev.T

        # db must match b's shape (n_out, 1): add dZ across the batch (axis=1 = across columns)
        self.db = np.sum(dZ, axis=1, keepdims=True)

        # dA_prev must match A_prev's shape (n_in, m).
        # W is (n_out, n_in), so W.T is (n_in, n_out): (n_in, n_out) @ (n_out, m) -> (n_in, m)
        return self.W.T @ dZ

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]        # used by the optimizers
