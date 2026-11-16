import numpy as np


class SGD:
    """Stochastic gradient descent with momentum.

    v = momentum * v - lr * grad      (a running "velocity" that remembers past steps)
    p = p + v
    momentum=0 gives plain gradient descent: p = p - lr * grad.
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}                                   # one velocity array per parameter

    def step(self, layers):
        for i, layer in enumerate(layers):
            for j, (param, grad) in enumerate(layer.params_and_grads()):
                key = (i, j)
                v = self.velocity.get(key, np.zeros_like(param))
                v = self.momentum * v - self.lr * grad
                self.velocity[key] = v
                param += v                                   # in place, so the layer's array changes


class Adam:
    """Adam: momentum on the gradient (m) plus a per-parameter step size from the squared gradient (v).

    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad^2
    m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)       (bias correction: m and v start at 0)
    p = p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0                                           # number of steps taken
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            for j, (param, grad) in enumerate(layer.params_and_grads()):
                key = (i, j)
                m = self.m.get(key, np.zeros_like(param))
                v = self.v.get(key, np.zeros_like(param))
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * grad ** 2
                self.m[key], self.v[key] = m, v
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
