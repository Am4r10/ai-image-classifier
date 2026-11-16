"""Train a [784, 128, 64, 10] network on MNIST.

Examples:
    python train_mnist.py --optimizer adam
    python train_mnist.py --optimizer sgd --lr 0.01 --momentum 0.9
"""
import argparse
import json
import os

import numpy as np

from Data.mnist import load_mnist
from Engine.losses import one_hot
from Engine.network import NeuralNetwork
from Engine.optimizers import SGD, Adam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="default: 0.001 for adam, 0.01 for sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_mnist()

    # Hold back the last 5,000 training images as a validation set for checking progress.
    # The test set is only used once, at the very end, so it stays an honest measure.
    X_val, y_val = X_train[:, 55000:], y_train[55000:]
    X_train, y_train = X_train[:, :55000], y_train[:55000]
    Y_train, Y_val, Y_test = one_hot(y_train, 10), one_hot(y_val, 10), one_hot(y_test, 10)

    net = NeuralNetwork([784, 128, 64, 10], seed=args.seed)
    if args.optimizer == "adam":
        optimizer = Adam(lr=args.lr or 0.001)
    else:
        optimizer = SGD(lr=args.lr or 0.01, momentum=args.momentum)

    print(f"Parameters: {net.num_parameters():,}   optimizer: {args.optimizer}   "
          f"batch size: {args.batch_size}")
    history = net.fit(X_train, Y_train, optimizer, epochs=args.epochs, batch_size=args.batch_size,
                      X_val=X_val, Y_val=Y_val, seed=args.seed)

    test_loss, test_acc = net.evaluate(X_test, Y_test)
    print(f"\nTest loss {test_loss:.4f}   test accuracy {test_acc * 100:.2f}%")

    os.makedirs("results", exist_ok=True)
    with open(f"results/train_{args.optimizer}.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "test_loss": test_loss, "test_acc": test_acc}, f, indent=2)
    np.savez(f"results/model_{args.optimizer}.npz",
             **{f"W{i + 1}": layer.W for i, layer in enumerate(net.layers)},
             **{f"b{i + 1}": layer.b for i, layer in enumerate(net.layers)})


if __name__ == "__main__":
    main()
