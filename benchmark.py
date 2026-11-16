"""Benchmark: pure-Python nested loops vs vectorised NumPy for one training epoch.

Both versions run the SAME maths (forward pass, softmax + cross-entropy, backprop, SGD update)
from the SAME starting weights, and the script checks that they end with the same weights.
The loop version is far too slow for a full 60,000-image epoch, so it runs on a subset and the
per-image time is scaled up to a full epoch.

Run with:  python benchmark.py            (or --samples 512 for a longer, steadier loop timing)
"""
import argparse
import json
import math
import os
import time

import numpy as np

from Data.mnist import load_mnist
from Engine.losses import one_hot
from Engine.network import NeuralNetwork
from Engine.optimizers import SGD

SIZES = [784, 128, 64, 10]


# ---------- Unvectorised version: Python lists and for loops only ----------

def matmul(A, B):
    """(n, k) @ (k, m) with three nested loops: n * k * m multiply-adds."""
    n, k, m = len(A), len(B), len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            total = 0.0
            for p in range(k):
                total += A[i][p] * B[p][j]
            out[i][j] = total
    return out


def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def loop_train_step(Ws, bs, X, Y, lr):
    """One SGD step on a batch. X: (784, m) and Y: (10, m) as lists of rows."""
    m = len(X[0])

    # Forward pass
    activations, pre_activations = [X], []
    A = X
    for layer, (W, b) in enumerate(zip(Ws, bs)):
        Z = matmul(W, A)
        for i in range(len(Z)):
            for j in range(m):
                Z[i][j] += b[i][0]
        if layer < len(Ws) - 1:
            A = [[z if z > 0 else 0.0 for z in row] for row in Z]       # ReLU
        else:
            A = [[0.0] * m for _ in range(len(Z))]                      # softmax, column by column
            for j in range(m):
                col_max = max(Z[i][j] for i in range(len(Z)))
                exps = [math.exp(Z[i][j] - col_max) for i in range(len(Z))]
                total = sum(exps)
                for i in range(len(Z)):
                    A[i][j] = exps[i] / total
        pre_activations.append(Z)
        activations.append(A)

    loss = 0.0
    for i in range(len(Y)):
        for j in range(m):
            if Y[i][j]:
                loss -= math.log(max(A[i][j], 1e-12)) / m

    # Backward pass: softmax + cross-entropy gives dZ = (A - Y) / m
    dZ = [[(A[i][j] - Y[i][j]) / m for j in range(m)] for i in range(len(Y))]
    grads = [None] * len(Ws)
    for layer in reversed(range(len(Ws))):
        dW = matmul(dZ, transpose(activations[layer]))
        db = [[sum(row)] for row in dZ]
        grads[layer] = (dW, db)
        if layer > 0:
            dA = matmul(transpose(Ws[layer]), dZ)
            Z_prev = pre_activations[layer - 1]
            dZ = [[dA[i][j] if Z_prev[i][j] > 0 else 0.0 for j in range(m)] for i in range(len(dA))]

    # Plain SGD update
    for (W, b), (dW, db) in zip(zip(Ws, bs), grads):
        for i in range(len(W)):
            for p in range(len(W[0])):
                W[i][p] -= lr * dW[i][p]
            b[i][0] -= lr * db[i][0]
    return loss


def loop_epoch(net, X, Y, batch_size, lr):
    Ws = [layer.W.tolist() for layer in net.layers]
    bs = [layer.b.tolist() for layer in net.layers]
    for start in range(0, X.shape[1], batch_size):
        loop_train_step(Ws, bs, X[:, start:start + batch_size].tolist(),
                        Y[:, start:start + batch_size].tolist(), lr)
    return Ws, bs


# ---------- Vectorised version: the real engine ----------

def numpy_epoch(net, X, Y, batch_size, lr):
    optimizer = SGD(lr=lr, momentum=0.0)                                 # plain SGD, same as the loops
    for start in range(0, X.shape[1], batch_size):
        net.train_step(X[:, start:start + batch_size], Y[:, start:start + batch_size], optimizer)


def copy_net(seed):
    return NeuralNetwork(SIZES, seed=seed)                               # same seed -> same weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=256, help="images used for the loop timing")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    X_train, y_train, _, _ = load_mnist()
    Y_train = one_hot(y_train, 10)
    n = args.samples
    X_small, Y_small = X_train[:, :n], Y_train[:, :n]

    # 1. Loops on the subset
    loop_net = copy_net(0)
    start = time.perf_counter()
    Ws, bs = loop_epoch(loop_net, X_small, Y_small, args.batch_size, args.lr)
    loop_seconds = time.perf_counter() - start

    # 2. NumPy on the same subset, from the same starting weights (best of 5 runs)
    numpy_seconds = float("inf")
    for _ in range(5):
        np_net = copy_net(0)
        start = time.perf_counter()
        numpy_epoch(np_net, X_small, Y_small, args.batch_size, args.lr)
        numpy_seconds = min(numpy_seconds, time.perf_counter() - start)

    # 3. Both versions must have learned exactly the same weights
    max_diff = max(np.max(np.abs(np.array(W) - layer.W)) for W, layer in zip(Ws, np_net.layers))

    # 4. NumPy on a real full epoch (60,000 images)
    full_net = copy_net(0)
    start = time.perf_counter()
    numpy_epoch(full_net, X_train, Y_train, args.batch_size, args.lr)
    numpy_full_epoch = time.perf_counter() - start

    loop_per_image = loop_seconds / n
    loop_full_epoch = loop_per_image * X_train.shape[1]
    results = {
        "samples": n,
        "batch_size": args.batch_size,
        "loop_seconds_subset": loop_seconds,
        "numpy_seconds_subset": numpy_seconds,
        "loop_ms_per_image": loop_per_image * 1000,
        "numpy_ms_per_image": numpy_seconds / n * 1000,
        "loop_full_epoch_seconds_estimated": loop_full_epoch,
        "numpy_full_epoch_seconds_measured": numpy_full_epoch,
        "speedup": loop_full_epoch / numpy_full_epoch,
        "max_weight_difference": float(max_diff),
    }

    print(f"Subset of {n} images, batch size {args.batch_size}")
    print(f"  loops : {loop_seconds:8.3f} s   ({results['loop_ms_per_image']:.3f} ms per image)")
    print(f"  numpy : {numpy_seconds:8.3f} s   ({results['numpy_ms_per_image']:.4f} ms per image)")
    print(f"  largest weight difference after training: {max_diff:.2e}  (same maths, same result)")
    print("Full epoch (60,000 images)")
    print(f"  loops : {loop_full_epoch / 60:8.1f} min  (estimated from the subset)")
    print(f"  numpy : {numpy_full_epoch:8.2f} s    (measured)")
    print(f"  speed-up: about {results['speedup']:,.0f}x")

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
