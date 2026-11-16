# AI Image Classifier: a Neural Network Built from Scratch

An AI that recognises handwritten digits, built from first principles with **only NumPy**: no PyTorch, TensorFlow or scikit-learn. Every part is implemented by hand: the forward pass, activation functions and their derivatives, loss functions, backpropagation, and the SGD-with-momentum and Adam optimizers. Every gradient is **verified numerically** against finite differences.

On MNIST handwritten digits, a `[784, 128, 64, 10]` network reaches **97.6% test accuracy** with Adam in 10 epochs (about 2–4 s per epoch on a laptop CPU).

| | |
|---|---|
| Trainable parameters | 109,386 |
| Test accuracy (Adam, 10 epochs) | 97.61% |
| Test accuracy (SGD + momentum, 10 epochs) | 97.59% |
| Largest gradient-check error | 1.3e-8 (relative) |
| Speed-up of NumPy over nested Python loops | about 270× |

---

## How it works

**1. An image becomes numbers.** Each picture is a 28×28 grid of pixels. Every pixel is a brightness value, so one image is a list of 784 numbers.

**2. The numbers flow through layers of "neurons".** Each neuron multiplies its inputs by its own set of **weights**, adds them up, and passes the result on. Neurons are grouped into layers: 784 inputs → 128 neurons → 64 neurons → 10 outputs. Altogether, the network has 109,386 adjustable weights.

**3. The network makes a guess.** The 10 outputs are turned into probabilities, one for each digit 0–9, such as "92% sure it's a 7". The highest one is the answer.

**4. It measures how wrong it was.** The guess is compared with the correct label, giving one number called the **loss**. A confident correct answer gives a loss near 0. A confident wrong answer gives a large loss.

**5. It works out how to improve.** Using the chain rule from calculus, **backpropagation** works backwards through the network and calculates, for every one of the 109,386 weights, whether nudging it up or down would reduce the loss, and by how much.

**6. It adjusts and repeats.** An **optimizer** nudges every weight a small step in the helpful direction. Repeating this over 60,000 training images, several times over, gradually turns random weights into a network that recognises digits it has never seen before, with 97.6% accuracy.

The sections below go through the maths behind each step.

---

## Quick start

```bash
pip install -r requirements.txt          # just NumPy
python Tests/test_gradients.py           # prove backprop is correct
python Tests/test_basics.py              # shapes, stability and optimizer tests
python train_mnist.py --optimizer adam   # downloads MNIST (~11 MB) on first run
python train_mnist.py --optimizer sgd
python benchmark.py                      # loops vs NumPy
```

## Project structure

```
Engine/
  activations.py   ReLU, Sigmoid, Softmax, Linear + exact derivatives
  losses.py        Cross-entropy and MSE + derivatives, one-hot encoding
  layer.py         Layer: one dense layer's forward and backward pass
  network.py       NeuralNetwork: stacks layers, mini-batch training loop
  optimizers.py    SGD with momentum, Adam
Data/mnist.py      Downloads and parses the raw MNIST files with NumPy
Tests/             Gradient check + shape/stability/optimizer tests
Learning/          The small experiments I built the project up from
train_mnist.py     Trains the network on MNIST
benchmark.py       Nested loops vs vectorised NumPy
results/           JSON output of the training runs and the benchmark
```

---

## 1. Notation and matrix shapes

Every example is a **column**. A batch of $m$ examples is a matrix with one column per example, so the same code works for any batch size. Layer $l$ has $n_l$ neurons.

| Quantity | Shape | For `[784, 128, 64, 10]` |
|---|---|---|
| $A^{[0]} = X$ (input batch) | $(n_0, m)$ | $(784, m)$ |
| $W^{[l]}$ | $(n_l, n_{l-1})$ | $(128, 784)$, $(64, 128)$, $(10, 64)$ |
| $b^{[l]}$ | $(n_l, 1)$, broadcast across the $m$ columns | $(128, 1)$, $(64, 1)$, $(10, 1)$ |
| $Z^{[l]}, A^{[l]}$ | $(n_l, m)$ | $(128, m)$, $(64, m)$, $(10, m)$ |
| $Y$ (one-hot labels) | $(n_L, m)$ | $(10, m)$ |

Parameter count: $(128 \cdot 784 + 128) + (64 \cdot 128 + 64) + (10 \cdot 64 + 10) = 100{,}480 + 8{,}256 + 650 = 109{,}386$.

## 2. Forward propagation

```math
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \qquad A^{[l]} = g^{[l]}\left(Z^{[l]}\right)
```

Shapes: $(n_l, n_{l-1}) \cdot (n_{l-1}, m) \to (n_l, m)$. Hidden layers use ReLU; the output layer uses softmax.

## 3. Activation functions and their derivatives

| Name | $g(z)$ | $g'(z)$ |
|---|---|---|
| ReLU | $\max(0, z)$ | $1$ if $z > 0$, else $0$ |
| Sigmoid | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | $\sigma(z)\,\big(1 - \sigma(z)\big)$ |
| Softmax | $a_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | $\dfrac{\partial a_i}{\partial z_k} = a_i(\delta_{ik} - a_k)$ |

Softmax mixes a whole column, so its derivative is a Jacobian matrix $J = \text{diag}(a) - a a^{\top}$. Its product with an upstream gradient simplifies without ever building $J$:

```math
J\,\frac{\partial L}{\partial a} = a \odot \left(\frac{\partial L}{\partial a} - \Big(a^{\top} \frac{\partial L}{\partial a}\Big)\mathbf{1}\right)
```

**Numerical stability:**
- Softmax subtracts each column's maximum before exponentiating. This doesn't change the result, since $\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$, but it stops $e^{z}$ overflowing.
- Sigmoid uses $\frac{e^{z}}{1 + e^{z}}$ for negative $z$, so `exp` is never given a large positive number.

## 4. Loss functions

Both losses are averaged over the $m$ examples in the batch.

**Categorical cross-entropy**, with probabilities clipped to $[\varepsilon, 1 - \varepsilon]$, $\varepsilon = 10^{-12}$, so $\log 0$ never happens:

```math
L_{\text{CE}} = -\frac{1}{m} \sum_{j=1}^{m} \sum_{i=1}^{n_L} Y_{ij} \log A_{ij},
\qquad
\frac{\partial L_{\text{CE}}}{\partial A} = -\frac{1}{m}\,\frac{Y}{A}
```

**Mean squared error:**

```math
L_{\text{MSE}} = \frac{1}{m} \sum_{j=1}^{m} \sum_{i=1}^{n_L} \left(A_{ij} - Y_{ij}\right)^2,
\qquad
\frac{\partial L_{\text{MSE}}}{\partial A} = \frac{2}{m}\,(A - Y)
```

## 5. Backpropagation

Backpropagation applies the chain rule from the loss back to the input, reusing each layer's result for the layer before it.

### Deriving the weight gradient

For one entry of $Z = WA + b$ (dropping the layer index), $Z_{ij} = \sum_k W_{ik} A_{kj} + b_i$. A weight $W_{ik}$ affects every example $j$ in row $i$, so by the chain rule:

```math
\frac{\partial L}{\partial W_{ik}} = \sum_{j=1}^{m} \frac{\partial L}{\partial Z_{ij}} \, A_{kj} = \left(dZ \, A^{\top}\right)_{ik}
```

That is why $A$ is transposed: the matrix product sums over the batch dimension $j$. The same reasoning gives $\frac{\partial L}{\partial A_{kj}} = \sum_i W_{ik}\,\frac{\partial L}{\partial Z_{ij}} = \left(W^{\top} dZ\right)_{kj}$.

### The four backprop equations

Writing $dX$ for $\partial L / \partial X$, and starting from $dA^{[L]} = \partial L / \partial A^{[L]}$:

```math
\begin{aligned}
dZ^{[l]} &= dA^{[l]} \odot g'^{[l]}\left(Z^{[l]}\right) && (n_l, m) \\
dW^{[l]} &= dZ^{[l]} \, A^{[l-1]\top} && (n_l, m)\cdot(m, n_{l-1}) \to (n_l, n_{l-1}) \\
db^{[l]} &= \textstyle\sum_{j=1}^{m} dZ^{[l]}_{:,j} && (n_l, 1) \\
dA^{[l-1]} &= W^{[l]\top} \, dZ^{[l]} && (n_{l-1}, n_l)\cdot(n_l, m) \to (n_{l-1}, m)
\end{aligned}
```

Every gradient has the same shape as the quantity it differentiates, which is a useful check on where each transpose goes. The $\frac{1}{m}$ from the averaged loss is already inside $dA^{[L]}$, so it doesn't appear again here.

### Softmax + cross-entropy simplifies

For one example, with $a = \text{softmax}(z)$ and one-hot $y$ (so $\sum_k y_k = 1$):

```math
\frac{\partial L}{\partial z_i}
= \sum_k \frac{\partial L}{\partial a_k}\frac{\partial a_k}{\partial z_i}
= \sum_k \left(-\frac{y_k}{a_k}\right) a_k\left(\delta_{ki} - a_i\right)
= -y_i + a_i \sum_k y_k
= a_i - y_i
```

Over a batch, $dZ^{[L]} = \frac{1}{m}\left(A^{[L]} - Y\right)$. The engine uses this shortcut for softmax + cross-entropy, which is faster and avoids dividing by tiny probabilities. A test confirms it matches the full Jacobian route to within $10^{-10}$.

## 6. Optimizers

**SGD with momentum** ($\mu = 0.9$):

```math
v \leftarrow \mu v - \eta \, \nabla_\theta L, \qquad \theta \leftarrow \theta + v
```

**Adam** ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, step $t$):

```math
\begin{aligned}
m &\leftarrow \beta_1 m + (1 - \beta_1)\, g, &\quad v &\leftarrow \beta_2 v + (1 - \beta_2)\, g^2 \\
\hat m &= \frac{m}{1 - \beta_1^{\,t}}, &\quad \hat v &= \frac{v}{1 - \beta_2^{\,t}} \\
\theta &\leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat v} + \epsilon}
\end{aligned}
```

The bias correction matters early on: $m$ and $v$ start at zero, so without it the first steps would be too small.

Weights use He initialisation for ReLU layers, $W \sim \mathcal{N}(0, 2/n_{l-1})$, and Xavier initialisation, $\mathcal{N}(0, 1/n_{l-1})$, otherwise. This keeps the size of the activations roughly constant from layer to layer.

## 7. Gradient check

`Tests/test_gradients.py` compares every backprop gradient with a central finite difference:

```math
\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + h) - L(\theta - h)}{2h}, \qquad h = 10^{-6},
\qquad \text{error} = \frac{\lVert g_{\text{backprop}} - g_{\text{numerical}} \rVert}{\lVert g_{\text{backprop}} \rVert + \lVert g_{\text{numerical}} \rVert}
```

It checks every activation/loss combination on a small `[5, 4, 3, 3]` network:

| Hidden | Output | Loss | Max relative error |
|---|---|---|---|
| ReLU | Softmax | Cross-entropy | 2.6e-9 |
| Sigmoid | Softmax | Cross-entropy | 1.1e-8 |
| ReLU | Softmax | MSE | 2.9e-9 |
| Sigmoid | Sigmoid | MSE | 1.3e-8 |
| ReLU | Linear | MSE | 5.3e-9 |
| Sigmoid | Sigmoid | Cross-entropy | 1.1e-8 |

The central difference has $O(h^2)$ error, so agreement to around $10^{-8}$ means the analytic gradients are correct.

## 8. Results on MNIST

Training on 55,000 images, with 5,000 held out for validation and the 10,000-image test set used only once at the end. Batch size 64, 10 epochs, seed 0.

| Optimizer | Learning rate | Val accuracy (epoch 10) | **Test accuracy** | Test loss | Time per epoch |
|---|---|---|---|---|---|
| SGD + momentum 0.9 | 0.01 | 98.04% | **97.59%** | 0.079 | ~2 s |
| Adam | 0.001 | 97.76% | **97.61%** | 0.095 | ~4 s |

Full per-epoch logs are in `results/train_sgd.json` and `results/train_adam.json`.

With Adam, the training loss keeps falling, but the validation loss bottoms out around epoch 5 (0.075) and then creeps up. That's overfitting: the network starts memorising the training images. Early stopping or L2 regularisation would be the natural next step.

## 9. Benchmark: nested loops vs vectorised NumPy

`benchmark.py` trains the same network with the same maths two ways:
- **Loops:** pure Python lists, with a triple `for` loop for every matrix product.
- **Vectorised:** the NumPy engine.

Both start from identical weights, and the script checks they finish with the same weights (largest difference 5.6e-17). This proves they compute the same thing.

| | Nested loops | Vectorised NumPy |
|---|---|---|
| Time per image (256-image subset) | 8.63 ms | 0.021 ms |
| One full epoch (60,000 images) | ~8.6 min (estimated from the subset) | 1.93 s (measured) |
| **Speed-up** | | **~270×** for a full epoch (~400× on the subset) |

Measured on a Windows laptop CPU with batch size 64. Raw numbers are in `results/benchmark.json`.

## 10. Runtime complexity

Let $P = \sum_{l} n_l \, n_{l-1}$ be the number of weights ($\approx 109{,}000$ here) and $N$ the number of training examples.

| Operation | Cost per batch of $m$ |
|---|---|
| Forward: $W^{[l]} A^{[l-1]}$ for every layer | $\mathcal{O}(m \sum_l n_l n_{l-1}) = \mathcal{O}(mP)$ |
| Backward: $dZ\,A^{\top}$ and $W^{\top} dZ$ for every layer | $\mathcal{O}(2mP)$ |
| Activations, bias, loss | $\mathcal{O}(m \sum_l n_l)$, small next to $mP$ |
| Optimizer update | $\mathcal{O}(P)$ |

One epoch is therefore $\mathcal{O}(N P)$: about $3NP \approx 2 \times 10^{10}$ multiply-adds for MNIST. Memory is $\mathcal{O}(P + m \sum_l n_l)$ for the weights plus the activations cached for backprop.

**Vectorisation doesn't change the big-O.** The loop and NumPy versions do exactly the same number of multiply-adds. The speed-up comes from the constant factor. A Python loop pays interpreter overhead on every single multiply-add. NumPy hands the whole matrix product to compiled BLAS code, which uses SIMD instructions, cache-friendly blocking and multiple CPU cores.

## 11. How I built it

`Learning/` contains the small experiments the engine grew out of:
- `gradient_descent.py`: gradient descent on $f(w) = (w-3)^2$, comparing the power-rule derivative with a numerical "nudge".
- `backprop_by_hand.py`: a two-neuron network with the chain rule written out step by step, then checked by nudging every weight.
- `losses_demo.py`: cross-entropy and MSE on hand-computed examples.
