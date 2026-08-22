"""
Principal Component Analysis, from scratch.

PCA is often taught as "eigenvectors of the covariance matrix", which is
true but is the wrong way to compute it.  The right way is the SVD of the
centred data matrix:

    X_centred = U S V'

    - rows of V'          are the principal directions (components)
    - S^2 / (n - 1)       are the variances along those directions
    - U S                 are the coordinates of each sample (the scores)

Why SVD instead of eig(X'X)?  Forming X'X squares the condition number, so
you lose half your precision before you start.  The SVD works on X directly.
sklearn does exactly this.

One wrinkle: the SVD is only unique up to sign.  Flipping a component and its
scores together gives an equally valid decomposition, so implementations pick
a convention.  We copy sklearn's (`svd_flip`) so the outputs are directly
comparable, sign and all.
"""

from __future__ import annotations

import numpy as np


def svd_flip(u: np.ndarray, vt: np.ndarray, v_based: bool = True):
    """Deterministic sign convention, matching sklearn.utils.extmath.svd_flip.

    The SVD is only unique up to sign: negating a column of U and the matching
    row of V' leaves U S V' unchanged. So implementations need a rule.

    v_based=True  -- for each component, find the entry of the RIGHT singular
                     vector with the largest absolute value and force it
                     positive. This is what sklearn's PCA uses
                     (svd_flip(..., u_based_decision=False)).
    v_based=False -- same idea but keyed on the left singular vectors, which
                     is svd_flip's default elsewhere in sklearn.

    Getting this wrong does not make the decomposition wrong -- it makes
    components come out negated, which looks alarming but is the same
    subspace. It matters here only because we want to compare numbers
    directly with sklearn.
    """
    if v_based:
        max_abs_cols = np.argmax(np.abs(vt), axis=1)
        signs = np.sign(vt[range(vt.shape[0]), max_abs_cols])
    else:
        max_abs_rows = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_rows, range(u.shape[1])])
    return u * signs, vt * signs[:, np.newaxis]


class PCA:
    """Principal component analysis via SVD.

    Attributes after fit (all named to match sklearn):
        components_                 (n_components, n_features)
        explained_variance_         variance along each component
        explained_variance_ratio_   fraction of total variance
        singular_values_            the S from the SVD
        mean_                       feature means removed before decomposing
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # Centring is not optional. PCA finds directions of maximum VARIANCE,
        # and variance is defined about the mean. Skip this and the first
        # component just points at the data's centre of mass.
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        U, Vt = svd_flip(U, Vt)

        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()

        k = self.n_components if self.n_components is not None else min(n_samples, n_features)
        self.components_ = Vt[:k]
        self.explained_variance_ = explained_variance[:k]
        self.explained_variance_ratio_ = explained_variance[:k] / total_variance
        self.singular_values_ = S[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto the principal components."""
        if self.components_ is None:
            raise RuntimeError("call fit() first")
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map back to the original space -- lossy unless all components kept."""
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared error of the rank-k reconstruction.

        By the Eckart-Young theorem this is the smallest error achievable by
        ANY rank-k approximation. PCA is not a heuristic -- it is optimal.
        """
        X = np.asarray(X, dtype=float)
        return float(((X - self.inverse_transform(self.transform(X))) ** 2).mean())
