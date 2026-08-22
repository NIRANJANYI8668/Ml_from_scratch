"""
Linear and ridge regression, from scratch.

Three routes to the same answer, which is the point:

  1. Normal equations      -- solve (X'X) w = X'y directly.  Fast, but
                              squares the condition number, so it loses
                              precision on badly scaled problems.
  2. SVD / least squares   -- what sklearn actually does.  Numerically
                              stable, handles rank-deficient X.
  3. Gradient descent      -- no linear algebra shortcut, just walk downhill.
                              Converges to the same w, slowly.

Seeing all three land on the same coefficients is a good first lesson: the
"model" is a mathematical object, and fitting is just a way of finding it.
"""

from __future__ import annotations

import numpy as np


class LinearRegression:
    """Ordinary least squares.

    Minimises  ||Xw + b - y||^2

    solver : "svd"    -- least squares via SVD (matches sklearn)
             "normal" -- normal equations, (X'X)^-1 X'y
    """

    def __init__(self, fit_intercept: bool = True, solver: str = "svd"):
        if solver not in {"svd", "normal"}:
            raise ValueError("solver must be 'svd' or 'normal'")
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.coef_: np.ndarray | None = None
        self.intercept_: float | np.ndarray = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.fit_intercept:
            # Centring the data is exactly equivalent to fitting an intercept,
            # and keeps the intercept out of the linear system.
            x_mean = X.mean(axis=0)
            y_mean = y.mean(axis=0)
            Xc, yc = X - x_mean, y - y_mean
        else:
            x_mean = np.zeros(X.shape[1])
            y_mean = 0.0
            Xc, yc = X, y

        if self.solver == "svd":
            # Minimum-norm least squares solution. Stable, and correct even
            # when X'X is singular.
            self.coef_, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        else:
            gram = Xc.T @ Xc
            self.coef_ = np.linalg.solve(gram, Xc.T @ yc)

        self.intercept_ = y_mean - x_mean @ self.coef_ if self.fit_intercept else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class Ridge:
    """L2-regularised least squares.

    Minimises  ||Xw + b - y||^2 + alpha * ||w||^2

    The intercept is NOT penalised -- shrinking it would make the fit depend
    on where you put the origin, which is meaningless. sklearn does the same,
    by centring the data first.

    Solved as  (X'X + alpha I) w = X'y.  Adding alpha to the diagonal is why
    ridge fixes near-singular problems: it lifts the small eigenvalues away
    from zero.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float | np.ndarray = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), y.mean(axis=0)
            Xc, yc = X - x_mean, y - y_mean
        else:
            x_mean, y_mean = np.zeros(X.shape[1]), 0.0
            Xc, yc = X, y

        n_features = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = y_mean - x_mean @ self.coef_ if self.fit_intercept else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class LinearRegressionGD:
    """Least squares by gradient descent -- the slow, general route.

    No closed form.  Just repeatedly step against the gradient:

        L(w)      = (1/n) ||Xw + b - y||^2
        dL/dw     = (2/n) X' (Xw + b - y)
        dL/db     = (2/n) sum(Xw + b - y)

    Included to show that the iterative machinery every neural network uses
    reproduces the exact algebraic answer on a problem where we know it.

    Features are standardised internally, because gradient descent on raw
    features with wildly different scales converges terribly -- itself a
    lesson worth seeing.

    Step size is not a free parameter.  For a quadratic loss with Hessian H,
    gradient descent converges if and only if

        lr < 2 / L,      L = largest eigenvalue of H

    Above that the iteration amplifies the error along the steepest direction
    and diverges to inf (then nan) within a few dozen steps.  Here
    H = (2/n) X'X, so L is computable, and lr="auto" picks 1/L -- comfortably
    inside the stable region and near the optimal rate.  Try passing
    lr=2.5/L's worth by hand and watch it blow up; that is the fastest way to
    understand every "my training diverged" post ever written.
    """

    def __init__(self, lr: float | str = "auto", n_iter: int = 5000, tol: float = 1e-12):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape

        x_mean, x_std = X.mean(axis=0), X.std(axis=0)
        x_std[x_std == 0] = 1.0
        Xs = (X - x_mean) / x_std

        # largest curvature of the loss, so we can pick a stable step
        hessian_scale = 2.0 * np.linalg.eigvalsh(Xs.T @ Xs / n).max()
        self.max_stable_lr_ = 2.0 / hessian_scale
        lr = (1.0 / hessian_scale) if self.lr == "auto" else float(self.lr)
        self.lr_used_ = lr
        if lr >= self.max_stable_lr_:
            import warnings
            warnings.warn(
                f"lr={lr:.4g} is at or above the stability limit "
                f"2/L={self.max_stable_lr_:.4g}; gradient descent will diverge.",
                RuntimeWarning, stacklevel=2)

        w = np.zeros(d)
        b = float(y.mean())
        self.loss_history_ = []

        for _ in range(self.n_iter):
            resid = Xs @ w + b - y
            loss = float((resid ** 2).mean())
            self.loss_history_.append(loss)

            grad_w = (2.0 / n) * (Xs.T @ resid)
            grad_b = (2.0 / n) * resid.sum()
            w -= lr * grad_w
            b -= lr * grad_b

            if len(self.loss_history_) > 1 and \
               abs(self.loss_history_[-2] - loss) < self.tol:
                break

        # undo the standardisation so coefficients are in original units
        self.coef_ = w / x_std
        self.intercept_ = b - float(x_mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
