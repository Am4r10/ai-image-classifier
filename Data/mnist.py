"""Downloads and loads MNIST using only the standard library and NumPy."""
import gzip
import os
import urllib.request

import numpy as np

URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"      # Google-hosted copy of the original files
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def _download(filename, folder):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):                                     # only download once
        os.makedirs(folder, exist_ok=True)
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(URL + filename, path)
    return path


def _read_idx(path, expected_magic, header_bytes):
    # IDX format: a big-endian header (magic number + sizes), then raw unsigned bytes
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != expected_magic:
        raise ValueError(f"{path} looks corrupted (magic number {magic})")
    return np.frombuffer(data, dtype=np.uint8, offset=header_bytes)


def load_mnist(folder=RAW_DIR):
    """Returns X_train (784, 60000), y_train (60000,), X_test (784, 10000), y_test (10000,).

    Pixels are scaled from 0..255 to 0..1. Each image is one COLUMN, matching the engine.
    """
    paths = {key: _download(name, folder) for key, name in FILES.items()}
    X_train = _read_idx(paths["train_images"], 2051, 16).reshape(-1, 784)   # (60000, 784): one row per image
    y_train = _read_idx(paths["train_labels"], 2049, 8)
    X_test = _read_idx(paths["test_images"], 2051, 16).reshape(-1, 784)
    y_test = _read_idx(paths["test_labels"], 2049, 8)

    # .T flips (examples, 784) into (784, examples) so every image is a column
    X_train = X_train.T.astype(np.float64) / 255.0
    X_test = X_test.T.astype(np.float64) / 255.0
    return X_train, y_train.astype(np.int64), X_test, y_test.astype(np.int64)
