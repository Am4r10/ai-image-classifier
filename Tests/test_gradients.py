"""Gradient check: compares backprop's gradients with slow "nudge" (finite difference) estimates.

For every weight w:  dL/dw ~ (L(w + h) - L(w - h)) / 2h
If backprop is correct, the two agree to about 1e-7 relative error or better.

Run with:  python Tests/test_gradients.py     (or: python -m pytest Tests)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # so "Engine" imports work

import numpy as np

from Engine.losses import one_hot
from Engine.network import NeuralNetwork

TOLERANCE = 1e-6

# (hidden activation, output activation, loss): every combination the engine supports
CONFIGS = [
    ("relu", "softmax", "cross_entropy"),     # the MNIST setup (uses the softmax + CE shortcut)
    ("sigmoid", "softmax", "cross_entropy"),
    ("relu", "softmax", "mse"),               # exercises the full softmax Jacobian
    ("sigmoid", "sigmoid", "mse"),
    ("relu", "linear", "mse"),                # regression-style output
    ("sigmoid", "sigmoid", "cross_entropy"),
]


def make_problem(hidden, output, loss, seed=0):
    rng = np.random.default_rng(seed)
    net = NeuralNetwork([5, 4, 3, 3], hidden_activation=hidden, output_activation=output,
                        loss=loss, seed=seed)
    X = rng.standard_normal((5, 7))                       # 7 examples, 5 features each
    if output == "linear":
        Y = rng.standard_normal((3, 7))                   # any real-valued targets
    else:
        Y = one_hot(rng.integers(0, 3, size=7), 3)
    return net, X, Y


def numerical_gradient(net, X, Y, param, h=1e-6):
    grad = np.zeros_like(param)
    for idx in np.ndindex(param.shape):                   # nudge one entry at a time
        original = param[idx]
        param[idx] = original + h
        loss_plus = net.loss.forward(net.forward(X), Y)
        param[idx] = original - h
        loss_minus = net.loss.forward(net.forward(X), Y)
        param[idx] = original                             # put it back
        grad[idx] = (loss_plus - loss_minus) / (2 * h)
    return grad


def relative_error(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


def check(hidden, output, loss):
    net, X, Y = make_problem(hidden, output, loss)
    net.backward(net.forward(X), Y)                       # fills in every layer's dW and db
    errors = []
    for layer in net.layers:
        for param, grad in [(layer.W, layer.dW.copy()), (layer.b, layer.db.copy())]:
            errors.append(relative_error(grad, numerical_gradient(net, X, Y, param)))
    return max(errors)


def test_all_configs():
    for config in CONFIGS:
        assert check(*config) < TOLERANCE, config


def test_softmax_ce_shortcut_matches_full_chain_rule():
    # (A - Y)/m should give exactly the same gradients as loss derivative -> softmax Jacobian
    net, X, Y = make_problem("relu", "softmax", "cross_entropy")
    net.backward(net.forward(X), Y)
    shortcut = [layer.dW.copy() for layer in net.layers]
    net.use_softmax_ce_shortcut = False
    net.backward(net.forward(X), Y)
    full = [layer.dW.copy() for layer in net.layers]
    for a, b in zip(shortcut, full):
        assert relative_error(a, b) < 1e-10


if __name__ == "__main__":
    print(f"{'hidden':<9}{'output':<9}{'loss':<15}{'max relative error':>20}")
    for config in CONFIGS:
        err = check(*config)
        status = "PASS" if err < TOLERANCE else "FAIL"
        print(f"{config[0]:<9}{config[1]:<9}{config[2]:<15}{err:>20.2e}  {status}")
    test_softmax_ce_shortcut_matches_full_chain_rule()
    print("softmax + cross-entropy shortcut matches the full chain rule: PASS")
