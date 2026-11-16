import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # so "Engine" imports work

import numpy as np
from Engine.losses import one_hot, mse, cross_entropy

labels = np.array([0, 2])                  # two images: a "0" and a "2"
Y = one_hot(labels, 3)
print("Y =")
print(Y)

good = np.array([[0.7, 0.1],               # each column is one image's guess
                 [0.2, 0.1],
                 [0.1, 0.8]])
print("good  CE:", round(cross_entropy(good, Y), 4), " MSE:", round(mse(good, Y), 4))

bad = np.array([[0.1, 0.8],                # confident in the wrong answers
                [0.2, 0.1],
                [0.7, 0.1]])
print("bad   CE:", round(cross_entropy(bad, Y), 4), " MSE:", round(mse(bad, Y), 4))

zero = np.array([[0.0], [1.0], [0.0]])     # gives the correct answer probability 0
print("zero  CE:", round(cross_entropy(zero, one_hot(np.array([0]), 3)), 4))