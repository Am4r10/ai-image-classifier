def relu(z):
    return max(0, z)                        # negatives become 0


def relu_slope(z):
    if z > 0:
        return 1                            # right side of ReLU: slope 1
    return 0                                # flat left side: slope 0


def forward(x, w1, b1, w2, b2):
    z1 = w1 * x + b1                        # neuron 1: weighted input plus bias
    a1 = relu(z1)                           # neuron 1's activation
    z2 = w2 * a1 + b2                       # neuron 2 (the output): uses neuron 1's output
    return z1, a1, z2


def loss(z2, y):
    return (z2 - y) ** 2                    # squared error


def backward(x, y, z1, a1, z2, w2):
    dz2 = 2 * (z2 - y)                      # slope of loss w.r.t. z2 (power rule)
    dw2 = dz2 * a1                          # chain rule: dL/dz2 times dz2/dw2
    db2 = dz2                               # dz2/db2 is 1
    da1 = dz2 * w2                          # pass the slope back to neuron 1
    dz1 = da1 * relu_slope(z1)              # through ReLU
    dw1 = dz1 * x                           # dz1/dw1 is x
    db1 = dz1                               # dz1/db1 is 1
    return dw1, db1, dw2, db2


def total_loss(w1, b1, w2, b2):
    z1, a1, z2 = forward(x, w1, b1, w2, b2)
    return loss(z2, y)


x, y = 2, 3                                 # one training example: input 2, correct answer 3
w1, b1, w2, b2 = 0.5, 0.1, 1.5, 0.2         # starting weights and biases

# --- one forward and backward pass ---
z1, a1, z2 = forward(x, w1, b1, w2, b2)
L = loss(z2, y)
print("z1 =", round(z1, 4), " a1 =", round(a1, 4), " z2 =", round(z2, 4), " loss =", round(L, 4))

dw1, db1, dw2, db2 = backward(x, y, z1, a1, z2, w2)
print("backprop:  dw1 =", round(dw1, 4), " db1 =", round(db1, 4), " dw2 =", round(dw2, 4), " db2 =", round(db2, 4))

# --- check every slope by nudging ---
h = 0.00001
print("nudging:   dw1 =", round((total_loss(w1 + h, b1, w2, b2) - L) / h, 4),
      " db1 =", round((total_loss(w1, b1 + h, w2, b2) - L) / h, 4),
      " dw2 =", round((total_loss(w1, b1, w2 + h, b2) - L) / h, 4),
      " db2 =", round((total_loss(w1, b1, w2, b2 + h) - L) / h, 4))
print()

# --- train with gradient descent ---
lr = 0.01
for step in range(15):
    z1, a1, z2 = forward(x, w1, b1, w2, b2)
    dw1, db1, dw2, db2 = backward(x, y, z1, a1, z2, w2)
    w1 = w1 - lr * dw1                      # step every dial downhill
    b1 = b1 - lr * db1
    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2
    print("step", step, " output =", round(z2, 4), " loss =", round(loss(z2, y), 4))