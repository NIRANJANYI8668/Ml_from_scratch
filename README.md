# ML from scratch, benchmarked against scikit-learn

<!-- Replace YOUR-USERNAME with your GitHub username once the repo is pushed. -->
![tests](https://github.com/YOUR-USERNAME/ml-from-scratch/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Seven machine learning algorithms implemented in pure NumPy, then checked against
scikit-learn on the same data — not just "does it roughly work", but *does it
produce the same numbers*, and *how much slower is it*.

Everything here runs in about 30 seconds on a laptop.

```bash
pip install -r requirements.txt

python benchmark.py        # the full comparison table + figures
python -m scratch.mlp      # gradient check: backprop vs finite differences
pytest -q                  # 27 correctness tests
```

---

## Results

**Deterministic algorithms — one right answer, must match to precision**

| algorithm | dataset | ours | sklearn | disagreement | speed vs sklearn |
|---|---|---:|---:|---:|---:|
| LinearRegression (SVD) | diabetes | R² 0.5177 | R² 0.5177 | **0.0** | 0.2× (faster) |
| LinearRegression (normal equations) | diabetes | R² 0.5177 | R² 0.5177 | 2.5e-11 | 0.1× (faster) |
| Ridge (α=0.1) | diabetes | R² 0.5126 | R² 0.5126 | 2.8e-13 | 0.1× (faster) |
| Ridge (α=10) | diabetes | R² 0.1889 | R² 0.1889 | 2.8e-14 | 0.1× (faster) |
| PCA (k=5) | breast_cancer | 84.73% var | 84.73% var | **0.0** | 0.8× (faster) |

**Randomised / greedy algorithms — models differ, performance must not**

| algorithm | dataset | ours | sklearn | gap | speed vs sklearn |
|---|---|---:|---:|---:|---:|
| LinearRegression (gradient descent) | diabetes | R² 0.5177 | R² 0.5177 | 7e-14 | 192× slower |
| KMeans (k=6) | blobs 3000×8 | inertia 52966.20 | inertia 52966.20 | 3e-16 | 0.3× (faster) |
| DecisionTreeClassifier (depth 6) | breast_cancer | 91.81% | 90.64% | +1.2pp | 3.9× slower |
| DecisionTreeRegressor (depth 4) | diabetes | R² 0.117 | R² 0.139 | −2.2pp | 5.3× slower |
| RandomForestClassifier (100 trees) | wine | 100% | 100% | 0.0 | 1.4× slower |
| RandomForestRegressor (50 trees) | diabetes | R² 0.298 | R² 0.292 | +0.6pp | 22× slower |
| MLPClassifier (64-32, Adam) | digits | 97.22% | 96.85% | +0.4pp | 1.5× slower |

**12/12 checks pass. Worst disagreement on a deterministic algorithm: 2.5e-11**
(float64 machine epsilon is 2.2e-16).

Backprop verified against finite differences at **1.4e-9** normwise relative error.

![benchmark](results/benchmark.png)

---

## The four things this actually taught me

**1. "Exact agreement" and "performance parity" are different standards, and
knowing which applies is half of understanding an algorithm.**

Least squares, ridge, and PCA have exactly one correct answer, so matching
sklearn to 1e-11 is a meaningful test — any real bug would show up as a much
larger error. But a decision tree breaks ties arbitrarily, k-means starts from
a random seed, and a neural net's initialisation is random. Two *correct*
implementations of those will produce different models. Demanding identical
output there would mean misunderstanding what the algorithm is. So the test
suite asserts precision for the first group and predictive quality for the
second.

**2. Being slower than sklearn is not uniform, and the pattern is informative.**

- **Faster on linear algebra (0.1–0.3×).** Not because the maths is better —
  it's identical LAPACK underneath. sklearn spends its time on input
  validation, dtype coercion, and sparse-matrix handling that a teaching
  implementation skips. That overhead is the price of being a library people
  can't easily misuse.
- **3–22× slower on trees.** This is real. sklearn's splitter is compiled
  Cython with an incrementally updated impurity; mine is vectorised NumPy that
  re-sorts each feature per node. Vectorisation gets you within an order of
  magnitude of C, not to it.
- **Only 1.5× slower on the neural network.** The most interesting result
  here. A neural net is almost entirely large matrix multiplications, and both
  implementations hand those to the same BLAS. Once your inner loop is a
  `matmul`, writing it yourself costs you almost nothing. This is exactly why
  the deep learning ecosystem could be built in Python.

**3. Gradient descent needs 192× the time to reach an answer that has a closed
form — and the step size is not a free parameter.**

For a quadratic loss with Hessian `H`, gradient descent converges if and only
if `lr < 2/L`, where `L` is the largest eigenvalue of `H`. Above that it
amplifies error along the steepest direction and diverges to `nan` in a few
dozen steps. My first run did exactly that. `LinearRegressionGD` now computes
`L` and picks `lr = 1/L`, and warns if you pass something unstable. Every
"my training diverged" story is a version of this.

**4. Hand-derived gradients need an independent check, and finite differences
have their own failure mode.**

`gradient_check()` compares analytic backprop against
`[L(w+ε) − L(w−ε)]/2ε`. All three activations agree to ~1e-9.

But finite differences assume a *smooth* loss, and ReLU has a kink at zero. If
perturbing a weight flips a unit across that corner, the difference straddles
a discontinuity and isn't the derivative at all. Scored per-element, ReLU shows
errors near 1e-4 while tanh stays at 1e-7 — on gradients that are equally
correct. The fix is to score normwise (which averages those rare outliers away)
and to validate with a smooth activation. **The discrepancy is a property of
the check, not the gradient.** `python -m scratch.mlp` demonstrates both.

---

## What's in each file

| file | what it implements | the idea worth carrying away |
|---|---|---|
| `scratch/linear.py` | OLS via SVD and normal equations, ridge, gradient descent | Three routes to one answer. Normal equations square the condition number; SVD doesn't. GD's step size is bounded by curvature. |
| `scratch/pca.py` | PCA via SVD of the centred data | Not "eigenvectors of the covariance matrix" — that squares the condition number too. By Eckart–Young, PCA is provably the *optimal* rank-k approximation, which the tests verify against 20 random projections. |
| `scratch/kmeans.py` | Lloyd's algorithm + k-means++ | The algorithm is two lines; the initialisation is what determines the answer. Lloyd provably never increases inertia — tested directly. |
| `scratch/tree.py` | CART, Gini and MSE | The only clever part: sort each feature once, then sweep with running sums so all thresholds cost O(n) instead of O(n²). Same idea as sklearn's Cython splitter. |
| `scratch/forest.py` | Bagging + feature subsetting, OOB scoring | Averaging only reduces variance if errors are independent, which is why per-split feature subsetting matters more than bootstrapping. OOB gives you cross-validation-quality error for free. |
| `scratch/mlp.py` | MLP, hand-derived backprop, Adam | Softmax + cross-entropy collapses to `(p − y)/N`. That cancellation isn't luck — matched output layer and loss always give "prediction minus target". |

## Test coverage

27 tests, split into three kinds:

- **Agreement with sklearn** — coefficients, components, explained variance.
- **Performance parity** — accuracy and R² within tolerance for stochastic methods.
- **Mathematical invariants that hold regardless of implementation** —
  PCA components orthonormal and optimal at rank k; k-means inertia monotonically
  decreasing; ridge coefficients shrinking as α grows; an unpruned tree reaching
  zero training error; `min_samples_leaf` and `max_depth` actually respected;
  feature importances summing to 1; a forest beating a single tree; OOB score
  tracking the held-out score; softmax stable at logits of ±50.

That third group is the most useful. Agreement tests tell you that you match
sklearn; invariant tests tell you that you match the *mathematics*, which is
the thing you were actually trying to learn.

## Where this could go next

- Gradient boosting (the natural sequel to the tree code — the residual-fitting
  idea is about 60 lines on top of `DecisionTreeRegressor`).
- Gaussian process regression, which is the right tool for surrogate modelling
  over expensive simulations.
- Swapping the tree splitter for histogram-based binning, which is how LightGBM
  and sklearn's `HistGradientBoosting` get their speed — and would close most of
  the 22× gap.
