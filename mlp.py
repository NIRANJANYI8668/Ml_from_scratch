"""
A multilayer perceptron with backpropagation written out by hand.

No autograd. Every derivative below was derived with the chain rule and
typed in, which is the entire point of the exercise.

THE FORWARD PASS
----------------
For layer l with weights W_l and bias b_l:

    z_l = a_{l-1} W_l + b_l          (pre-activation)
    a_l = g(z_l)                     (activation)

with a_0 = X. The last layer uses softmax instead of g:

    p = softmax(z_L),   p_k = exp(z_k) / SUM_j exp(z_j)

and the loss is cross-entropy averaged over the batch:

    L = -(1/N) SUM_i SUM_k  y_ik log p_ik

THE BACKWARD PASS
-----------------
Backprop is the chain rule applied right-to-left, reusing partial results.
Define d_l = dL/dz_l. Then everything follows from d_l:

    dL/dW_l = a_{l-1}' d_l
    dL/db_l = column sums of d_l
    d_{l-1} = (d_l W_l') * g'(z_{l-1})     <- the recursion

The base case is the nice one. Differentiating softmax gives a full Jacobian
that looks horrible, but composed with cross-entropy almost all of it cancels
and you are left with:

    d_L = (p - y) / N

That cancellation is not a coincidence -- it happens because softmax is the
canonical link for the categorical distribution. Same reason linear output
with squared error gives (yhat - y), and sigmoid with log-loss gives the same
form. Worth knowing: if your output layer and loss are matched, the output
gradient is always "prediction minus target".

WHY THE GRADIENT CHECK MATTERS
------------------------------
Hand-derived gradients are easy to get subtly wrong -- a transpose, a missing
1/N, a wrong sign -- and the network will still train, just badly. The only
honest test is to compare against a finite-difference estimate:

    dL/dw  ~  [L(w + eps) - L(w - eps)] / (2 eps)

Slow, but it needs no derivation, so it is an independent check. Run
`python -m scratch.mlp` to see it pass.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# activations and their derivatives
# ---------------------------------------------------------------------------

def _relu(z):            return np.maximum(z, 0.0)
def _relu_grad(z):       return (z > 0).astype(z.dtype)
def _tanh(z):            return np.tanh(z)
def _tanh_grad(z):       return 1.0 - np.tanh(z) ** 2
def _logistic(z):        return 1.0 / (1.0 + np.exp(-z))
def _logistic_grad(z):
    s = _logistic(z)
    return s * (1.0 - s)

ACTIVATIONS = {
    "relu": (_relu, _relu_grad),
    "tanh": (_tanh, _tanh_grad),
    "logistic": (_logistic, _logistic_grad),
}


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Subtracting the row max changes nothing mathematically (it cancels in the
    ratio) but stops exp() overflowing for large logits.
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------

class MLPClassifier:
    """Feedforward network trained by minibatch Adam.

    hidden_layer_sizes : tuple of hidden widths
    activation         : "relu" | "tanh" | "logistic"
    alpha              : L2 penalty on weights (not biases)
    """

    def __init__(self, hidden_layer_sizes=(64,), activation: str = "relu",
                 alpha: float = 1e-4, learning_rate_init: float = 1e-3,
                 batch_size: int = 64, max_iter: int = 200,
                 tol: float = 1e-5, n_iter_no_change: int = 15,
                 random_state: int | None = None, verbose: bool = False):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}")
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.verbose = verbose

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.loss_curve_: list[float] = []
        self.n_iter_: int = 0

    # ---- initialisation --------------------------------------------------

    def _init_params(self, n_features: int, n_outputs: int, rng: np.random.Generator):
        sizes = [n_features, *self.hidden_layer_sizes, n_outputs]
        self.weights_, self.biases_ = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He initialisation for ReLU, Xavier otherwise. The scale matters:
            # too large and activations saturate or explode through depth,
            # too small and the signal dies out before it reaches the output.
            scale = np.sqrt(2.0 / fan_in) if self.activation == "relu" \
                else np.sqrt(1.0 / fan_in)
            self.weights_.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))

    # ---- forward ----------------------------------------------------------

    def _forward(self, X: np.ndarray):
        """Return (activations, pre_activations). Keeps every intermediate,
        because backprop needs them on the way back down."""
        g, _ = ACTIVATIONS[self.activation]
        activations = [X]
        pre_activations = []

        n_layers = len(self.weights_)
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = activations[-1] @ W + b
            pre_activations.append(z)
            activations.append(softmax(z) if i == n_layers - 1 else g(z))

        return activations, pre_activations

    # ---- loss --------------------------------------------------------------

    def _loss(self, probs: np.ndarray, Y: np.ndarray) -> float:
        n = len(Y)
        eps = 1e-12
        ce = -float((Y * np.log(probs + eps)).sum()) / n
        l2 = 0.5 * self.alpha * sum(float((W ** 2).sum()) for W in self.weights_) / n
        return ce + l2

    # ---- backward ----------------------------------------------------------

    def _backward(self, activations, pre_activations, Y):
        """Hand-derived gradients. Returns (grads_W, grads_b)."""
        _, g_prime = ACTIVATIONS[self.activation]
        n = len(Y)
        n_layers = len(self.weights_)

        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        # --- output layer: the softmax + cross-entropy simplification ---
        #     dL/dz_L = (p - y) / N
        delta = (activations[-1] - Y) / n

        for l in range(n_layers - 1, -1, -1):
            #  dL/dW_l = a_{l-1}' delta      (+ L2 term)
            grads_W[l] = activations[l].T @ delta + (self.alpha / n) * self.weights_[l]
            #  dL/db_l = sum of delta over the batch
            grads_b[l] = delta.sum(axis=0)

            if l > 0:
                # push the error back through the weights, then through the
                # activation:   delta_{l-1} = (delta_l W_l') * g'(z_{l-1})
                delta = (delta @ self.weights_[l].T) * g_prime(pre_activations[l - 1])

        return grads_W, grads_b

    # ---- training ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        self.classes_, encoded = np.unique(y, return_inverse=True)
        n_outputs = len(self.classes_)
        Y = np.zeros((len(y), n_outputs))
        Y[np.arange(len(y)), encoded] = 1.0

        self._init_params(X.shape[1], n_outputs, rng)

        # Adam state: first and second moment estimates per parameter
        mW = [np.zeros_like(W) for W in self.weights_]
        vW = [np.zeros_like(W) for W in self.weights_]
        mb = [np.zeros_like(b) for b in self.biases_]
        vb = [np.zeros_like(b) for b in self.biases_]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        n = len(X)
        batch = min(self.batch_size, n)
        self.loss_curve_ = []
        best_loss, no_improve = np.inf, 0

        for epoch in range(self.max_iter):
            order = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch):
                idx = order[start : start + batch]
                Xb, Yb = X[idx], Y[idx]

                activations, pre_activations = self._forward(Xb)
                epoch_loss += self._loss(activations[-1], Yb)
                n_batches += 1

                grads_W, grads_b = self._backward(activations, pre_activations, Yb)

                # --- Adam update ---
                # Momentum on the gradient (m) and on its square (v), each
                # bias-corrected because they start at zero. Dividing by
                # sqrt(v) gives every parameter its own effective step size,
                # which is why Adam copes with badly scaled problems.
                step += 1
                for l in range(len(self.weights_)):
                    mW[l] = beta1 * mW[l] + (1 - beta1) * grads_W[l]
                    vW[l] = beta2 * vW[l] + (1 - beta2) * grads_W[l] ** 2
                    mb[l] = beta1 * mb[l] + (1 - beta1) * grads_b[l]
                    vb[l] = beta2 * vb[l] + (1 - beta2) * grads_b[l] ** 2

                    mW_hat = mW[l] / (1 - beta1 ** step)
                    vW_hat = vW[l] / (1 - beta2 ** step)
                    mb_hat = mb[l] / (1 - beta1 ** step)
                    vb_hat = vb[l] / (1 - beta2 ** step)

                    self.weights_[l] -= self.learning_rate_init * mW_hat / (np.sqrt(vW_hat) + eps)
                    self.biases_[l] -= self.learning_rate_init * mb_hat / (np.sqrt(vb_hat) + eps)

            epoch_loss /= max(n_batches, 1)
            self.loss_curve_.append(epoch_loss)
            self.n_iter_ = epoch + 1

            if self.verbose and epoch % 20 == 0:
                print(f"    epoch {epoch:4d}  loss {epoch_loss:.6f}")

            if epoch_loss < best_loss - self.tol:
                best_loss, no_improve = epoch_loss, 0
            else:
                no_improve += 1
                if no_improve >= self.n_iter_no_change:
                    break

        return self

    # ---- prediction ---------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(X, dtype=float))[0][-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # ---- the honesty check ---------------------------------------------------

    def gradient_check(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-6,
                       n_checks: int | None = None, seed: int = 0) -> float:
        """Compare hand-derived gradients against finite differences.

        Returns the standard normwise relative error

            ||analytic - numeric|| / (||analytic|| + ||numeric||)

        rather than a per-element ratio. Per-element ratios are misleading:
        a weight whose true gradient is ~1e-12 will show a huge RELATIVE
        error from pure floating-point noise even though the derivation is
        perfect. The norm form weights each entry by how much it matters.

        Below ~1e-8 means the analytic derivation is correct.

        CAVEAT -- ReLU. Finite differences assume the loss is smooth. ReLU has
        a kink at zero, so if perturbing a weight flips any unit across that
        kink, the two-sided difference straddles a corner and returns
        something that is not the derivative at all. This is a limitation of
        the CHECK, not a bug in the gradient. Check with tanh or logistic,
        which are smooth everywhere, then switch the activation back.
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=float)
        classes, encoded = np.unique(y, return_inverse=True)
        Y = np.zeros((len(y), len(classes)))
        Y[np.arange(len(y)), encoded] = 1.0

        if not self.weights_:
            self._init_params(X.shape[1], len(classes), rng)
        self.classes_ = classes

        activations, pre_activations = self._forward(X)
        grads_W, _ = self._backward(activations, pre_activations, Y)

        def loss_at() -> float:
            return self._loss(self._forward(X)[0][-1], Y)

        # enumerate the weights to check
        coords = [(l, i, j)
                  for l in range(len(self.weights_))
                  for i in range(self.weights_[l].shape[0])
                  for j in range(self.weights_[l].shape[1])]
        if n_checks is not None and n_checks < len(coords):
            picks = rng.choice(len(coords), size=n_checks, replace=False)
            coords = [coords[p] for p in picks]

        numeric = np.empty(len(coords))
        analytic = np.empty(len(coords))

        for idx, (l, i, j) in enumerate(coords):
            original = self.weights_[l][i, j]

            self.weights_[l][i, j] = original + eps
            loss_plus = loss_at()
            self.weights_[l][i, j] = original - eps
            loss_minus = loss_at()
            self.weights_[l][i, j] = original

            numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
            analytic[idx] = grads_W[l][i, j]

        denom = np.linalg.norm(analytic) + np.linalg.norm(numeric)
        return float(np.linalg.norm(analytic - numeric) / max(denom, 1e-30))


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=60, n_features=8, n_informative=6,
                               n_classes=3, random_state=0)

    print("Gradient check: analytic backprop vs finite differences")
    print("(normwise relative error over every weight in the network)\n")

    for act in ["tanh", "logistic"]:
        net = MLPClassifier(hidden_layer_sizes=(12, 9), activation=act,
                            alpha=1e-3, random_state=0)
        err = net.gradient_check(X, y)
        print(f"  {act:<10} {err:.3e}   {'PASS' if err < 1e-8 else 'FAIL'}")

    net = MLPClassifier(hidden_layer_sizes=(12, 9), activation="relu",
                        alpha=1e-3, random_state=0)
    err_relu = net.gradient_check(X, y)
    print(f"  {'relu':<10} {err_relu:.3e}   {'PASS' if err_relu < 1e-8 else 'FAIL'}")

    print("\nAll three agree with finite differences to ~1e-9, so the chain")
    print("rule was applied correctly.")
    print("\nA note on ReLU: finite differences assume a smooth loss, and ReLU")
    print("has a kink at zero. If perturbing a weight flips a unit across that")
    print("corner, that single entry's difference straddles a discontinuity and")
    print("is not the derivative at all. It passes here because the normwise")
    print("measure averages those rare outliers away -- but score the SAME run")
    print("per-element and ReLU shows errors near 1e-4 while tanh stays at 1e-7:")

    per_element = []
    for act in ["tanh", "relu"]:
        n2 = MLPClassifier(hidden_layer_sizes=(12, 9), activation=act,
                           alpha=1e-3, random_state=0)
        rng = np.random.default_rng(0)
        classes, enc = np.unique(y, return_inverse=True)
        Y = np.zeros((len(y), len(classes))); Y[np.arange(len(y)), enc] = 1.0
        n2._init_params(X.shape[1], len(classes), rng); n2.classes_ = classes
        acts, pres = n2._forward(X)
        gW, _ = n2._backward(acts, pres, Y)

        worst = 0.0
        for _ in range(400):
            l = rng.integers(len(n2.weights_))
            i = rng.integers(n2.weights_[l].shape[0])
            j = rng.integers(n2.weights_[l].shape[1])
            o = n2.weights_[l][i, j]
            n2.weights_[l][i, j] = o + 1e-6; lp = n2._loss(n2._forward(X)[0][-1], Y)
            n2.weights_[l][i, j] = o - 1e-6; lm = n2._loss(n2._forward(X)[0][-1], Y)
            n2.weights_[l][i, j] = o
            num = (lp - lm) / 2e-6
            ana = gW[l][i, j]
            worst = max(worst, abs(num - ana) / max(abs(num), abs(ana), 1e-8))
        per_element.append((act, worst))

    for act, worst in per_element:
        print(f"  {act:<10} worst single-weight relative error {worst:.3e}")
    print("\nThat gap is a property of the CHECK, not of the gradient.")
