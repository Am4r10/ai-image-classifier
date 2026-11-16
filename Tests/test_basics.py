"""Sanity tests for shapes, numerical stability and the optimizers.

Run with:  python Tests/test_basics.py     (or: python -m pytest Tests)
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from Engine.activations import sigmoid, softmax
from Engine.losses import cross_entropy, one_hot
from Engine.network import NeuralNetwork
from Engine.optimizers import SGD, Adam


def test_shapes_for_any_batch_size():
    net = NeuralNetwork([784, 128, 64, 10], seed=0)
    expected_W = [(128, 784), (64, 128), (10, 64)]
    for m in [1, 32, 1000]:                                    # dynamic batching: any m works
        X = np.random.default_rng(0).random((784, m))
        Y = one_hot(np.zeros(m, dtype=int), 10)
        A = net.forward(X)
        assert A.shape == (10, m)
        net.backward(A, Y)
        for layer, shape in zip(net.layers, expected_W):
            assert layer.W.shape == layer.dW.shape == shape     # gradient has the same shape as W
            assert layer.b.shape == layer.db.shape == (shape[0], 1)
    assert net.num_parameters() == 109_386


def test_softmax_columns_sum_to_one_and_never_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                         # any overflow warning fails the test
        p = softmax(np.array([[1000.0, 1.0], [1001.0, 2.0], [999.0, 3.0]]))
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=0), 1.0)


def test_sigmoid_is_stable_for_huge_inputs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.allclose(s, [0.0, 0.5, 1.0])


def test_cross_entropy_clipping_avoids_log_zero():
    loss = cross_entropy(np.array([[0.0], [1.0]]), one_hot(np.array([0]), 2))
    assert np.isfinite(loss) and loss > 20                     # huge, but not infinite


def test_optimizers_reduce_loss():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 64))
    Y = one_hot((X[0] > 0).astype(int), 2)                     # simple rule the network can learn
    for optimizer in [SGD(lr=0.1, momentum=0.9), Adam(lr=0.01)]:
        net = NeuralNetwork([4, 8, 2], seed=1)
        first = net.train_step(X, Y, optimizer)
        for _ in range(200):
            last = net.train_step(X, Y, optimizer)
        assert last < first * 0.2, type(optimizer).__name__


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
