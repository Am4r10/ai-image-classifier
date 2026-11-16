import time

import numpy as np

from Engine.layer import Layer
from Engine.losses import get_loss


class NeuralNetwork:
    """A multi-layer perceptron built from Layer objects.

    layer_sizes = [784, 128, 64, 10] makes 3 layers: 784->128, 128->64, 64->10.
    """

    def __init__(self, layer_sizes, hidden_activation="relu", output_activation="softmax",
                 loss="cross_entropy", seed=None):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            is_last = i == len(layer_sizes) - 2
            activation = output_activation if is_last else hidden_activation
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation, rng))
        self.loss = get_loss(loss)

        # Softmax followed by cross-entropy simplifies to dL/dZ = (A - Y) / m.
        # Using it directly is faster and avoids dividing by tiny probabilities.
        self.use_softmax_ce_shortcut = output_activation == "softmax" and loss == "cross_entropy"

    def forward(self, X):
        A = X                                        # (n_0, m)
        for layer in self.layers:
            A = layer.forward(A)                     # each layer's output feeds the next
        return A                                     # (n_L, m)

    def backward(self, A_out, Y):
        last = self.layers[-1]
        if self.use_softmax_ce_shortcut:
            m = Y.shape[1]
            dA = last.backward_from_dZ((A_out - Y) / m)
        else:
            dA = last.backward(self.loss.backward(A_out, Y))
        for layer in reversed(self.layers[:-1]):     # walk backwards, passing dA to the layer before
            dA = layer.backward(dA)

    def train_step(self, X, Y, optimizer):
        A_out = self.forward(X)
        loss = self.loss.forward(A_out, Y)
        self.backward(A_out, Y)
        optimizer.step(self.layers)
        return loss

    def fit(self, X, Y, optimizer, epochs=10, batch_size=64, X_val=None, Y_val=None,
            seed=None, verbose=True):
        """Mini-batch training. Any batch size works; the last batch can be smaller."""
        rng = np.random.default_rng(seed)
        m = X.shape[1]
        history = []
        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            order = rng.permutation(m)                       # shuffle every epoch
            total_loss = 0.0
            for start in range(0, m, batch_size):
                idx = order[start:start + batch_size]
                batch_loss = self.train_step(X[:, idx], Y[:, idx], optimizer)
                total_loss += batch_loss * len(idx)          # weight by batch size for a fair average
            seconds = time.perf_counter() - start_time

            record = {"epoch": epoch, "train_loss": total_loss / m, "seconds": seconds}
            if X_val is not None:
                record["val_loss"], record["val_acc"] = self.evaluate(X_val, Y_val)
            history.append(record)

            if verbose:
                line = f"epoch {epoch:2d}  train loss {record['train_loss']:.4f}"
                if X_val is not None:
                    line += f"  val loss {record['val_loss']:.4f}  val acc {record['val_acc'] * 100:.2f}%"
                print(line + f"  ({seconds:.2f}s)")
        return history

    def predict_proba(self, X, chunk=10000):
        # Forward pass in chunks so big datasets don't need one huge matrix
        return np.concatenate([self.forward(X[:, i:i + chunk]) for i in range(0, X.shape[1], chunk)], axis=1)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)      # most likely class for each column

    def evaluate(self, X, Y):
        A_out = self.predict_proba(X)
        loss = self.loss.forward(A_out, Y)
        accuracy = np.mean(np.argmax(A_out, axis=0) == np.argmax(Y, axis=0))
        return float(loss), float(accuracy)

    def num_parameters(self):
        return sum(layer.W.size + layer.b.size for layer in self.layers)
