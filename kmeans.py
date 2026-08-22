"""
k-means clustering, from scratch.

The algorithm (Lloyd, 1957) is two lines:

    repeat:
        assign each point to its nearest centroid
        move each centroid to the mean of its assigned points

It provably decreases the objective

    inertia = SUM_i  || x_i - mu_{c(i)} ||^2

at every step, so it always converges -- but only to a LOCAL minimum.  Which
one you land in depends entirely on where you start, which is why the
initialisation matters more than the algorithm.

k-means++ (Arthur & Vassilvitskii, 2007) fixes this: pick the first centre at
random, then pick each subsequent centre with probability proportional to its
squared distance from the nearest existing centre.  Spread-out starts, far
better local minima.  It comes with an O(log k) approximation guarantee.

We also copy sklearn's trick of running the whole thing n_init times and
keeping the best, plus its "greedy k-means++" variant which samples several
candidate centres per step and keeps whichever lowers inertia most.
"""

from __future__ import annotations

import numpy as np


def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Pairwise squared euclidean distances, (n_samples, n_centers).

    Uses ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2 so the work is one matmul
    rather than an explicit loop. The clip guards against tiny negatives
    from floating point cancellation.
    """
    x_sq = (X ** 2).sum(axis=1)[:, np.newaxis]
    c_sq = (centers ** 2).sum(axis=1)[np.newaxis, :]
    d = x_sq - 2.0 * (X @ centers.T) + c_sq
    return np.maximum(d, 0.0)


class KMeans:
    """Lloyd's algorithm with k-means++ initialisation.

    Attributes after fit (named to match sklearn):
        cluster_centers_  (n_clusters, n_features)
        labels_           cluster index for each training point
        inertia_          sum of squared distances to assigned centre
        n_iter_           iterations used by the best run
    """

    def __init__(self, n_clusters: int = 8, n_init: int = 10, max_iter: int = 300,
                 tol: float = 1e-4, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    # ---- initialisation -------------------------------------------------

    def _kmeans_plusplus(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples = X.shape[0]
        k = self.n_clusters
        centers = np.empty((k, X.shape[1]))

        # first centre: uniformly at random
        centers[0] = X[rng.integers(n_samples)]
        closest_sq = _squared_distances(X, centers[:1]).ravel()

        # sklearn tries this many candidates per step and keeps the best
        n_candidates = 2 + int(np.log(k))

        for c in range(1, k):
            # sample candidates with probability proportional to D(x)^2
            total = closest_sq.sum()
            if total <= 0:
                candidate_ids = rng.integers(n_samples, size=n_candidates)
            else:
                probs = closest_sq / total
                candidate_ids = rng.choice(n_samples, size=n_candidates, p=probs)

            cand_sq = _squared_distances(X, X[candidate_ids])
            # for each candidate, what would the total inertia become?
            new_closest = np.minimum(closest_sq[:, np.newaxis], cand_sq)
            best = int(np.argmin(new_closest.sum(axis=0)))

            centers[c] = X[candidate_ids[best]]
            closest_sq = new_closest[:, best]

        return centers

    # ---- Lloyd iterations ------------------------------------------------

    def _lloyd(self, X: np.ndarray, centers: np.ndarray, rng: np.random.Generator):
        labels = np.zeros(X.shape[0], dtype=int)
        n_iter = 0

        for n_iter in range(1, self.max_iter + 1):
            # E step: assign to nearest centre
            distances = _squared_distances(X, centers)
            labels = np.argmin(distances, axis=1)

            # M step: move centres to cluster means
            new_centers = centers.copy()
            for j in range(self.n_clusters):
                members = X[labels == j]
                if len(members):
                    new_centers[j] = members.mean(axis=0)
                else:
                    # empty cluster: re-seed it at the worst-fitting point,
                    # which is what sklearn effectively does
                    worst = int(np.argmax(distances[np.arange(len(X)), labels]))
                    new_centers[j] = X[worst]

            shift = float(((new_centers - centers) ** 2).sum())
            centers = new_centers
            if shift <= self.tol:
                break

        distances = _squared_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        inertia = float(distances[np.arange(len(X)), labels].sum())
        return centers, labels, inertia, n_iter

    # ---- public API -------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best = (None, None, np.inf, 0)
        for _ in range(self.n_init):
            init_centers = self._kmeans_plusplus(X, rng)
            centers, labels, inertia, n_iter = self._lloyd(X, init_centers, rng)
            if inertia < best[2]:
                best = (centers, labels, inertia, n_iter)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        d = _squared_distances(np.asarray(X, dtype=float), self.cluster_centers_)
        return np.argmin(d, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
