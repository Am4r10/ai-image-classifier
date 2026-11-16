def f(w):
    return (w - 3) ** 2                     # the loss: smallest when w = 3


def slope_formula(w):
    return 2 * w - 6                        # derivative from the power rule


def slope_nudge(w, h=0.00001):
    return (f(w + h) - f(w)) / h            # nudge w a tiny bit and measure the change in loss


w = 5
print("slope by formula:", slope_formula(w))
print("slope by nudging:", slope_nudge(w))
print()                                     # blank line

w = 5.0                                     # starting guess
lr = 0.1                                    # learning rate (step size)
for step in range(20):
    w = w - lr * slope_formula(w)           # step downhill
    print("step", step, " w =", round(w, 4), " loss =", round(f(w), 4))