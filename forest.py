"""
Random forests, from scratch.

A single deep decision tree has very low bias and very high variance -- it
will fit the training set perfectly and change completely if you perturb the
data. The fix is to average many trees. But averaging only reduces variance
if the trees make INDEPENDENT errors, and trees grown on the same data are
nearly identical. So a random forest injects randomness twice:

  1. Bagging          -- each tree sees a bootstrap resample (n draws with
                         replacement). About 1 - 1/e ~ 63% of rows appear in
                         any given tree; the rest are "out-of-bag" and can be
                         used as a free validation set.
  2. Feature subsetting -- at EVERY split, only a random subset of features is
                         considered (sqrt(p) for classification, p for
                         regression, by convention). This is Breiman's key
                         addition over plain bagging: without it, one strong
                         feature dominates the top split of every tree and
                         they stay correlated.

Prediction averages the trees' predicted PROBABILITIES rather than taking a
majority vote of hard labels -- which is what sklearn does, and is slightly
better because it keeps the confidence information.

The OOB score is worth noticing: it gives you cross-validation-quality error
estimates for free, as a side effect of bootstrapping.
"""

from __future__ import annotations

import numpy as np

from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class _BaseForest:
    tree_class = None
    is_classifier = False

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: int | str | None = "sqrt", bootstrap: bool = True,
                 oob_score: bool = False, random_state: int | None = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.estimators_: list = []
        self.classes_: np.ndarray | None = None
        self.n_features_: int = 0
        self.oob_score_: float | None = None

    def _make_tree(self, seed: int):
        return self.tree_class(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples = len(y)
        self.n_features_ = X.shape[1]
        if self.is_classifier:
            self.classes_ = np.unique(y)

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2 ** 31 - 1, size=self.n_estimators)

        self.estimators_ = []
        oob_sum = None
        oob_count = np.zeros(n_samples)

        for seed in seeds:
            tree = self._make_tree(int(seed))

            if self.bootstrap:
                boot_rng = np.random.default_rng(int(seed))
                idx = boot_rng.integers(0, n_samples, size=n_samples)
                tree.fit(X[idx], y[idx])

                if self.oob_score:
                    oob_mask = np.ones(n_samples, dtype=bool)
                    oob_mask[np.unique(idx)] = False
                    if oob_mask.any():
                        pred = (tree.predict_proba(X[oob_mask]) if self.is_classifier
                                else tree.predict(X[oob_mask]))
                        if oob_sum is None:
                            shape = ((n_samples, pred.shape[1]) if self.is_classifier
                                     else (n_samples,))
                            oob_sum = np.zeros(shape)
                        oob_sum[oob_mask] += pred
                        oob_count[oob_mask] += 1
            else:
                tree.fit(X, y)

            self.estimators_.append(tree)

        if self.oob_score and oob_sum is not None:
            self.oob_score_ = self._compute_oob(y, oob_sum, oob_count)

        return self

    def _compute_oob(self, y, oob_sum, oob_count):
        seen = oob_count > 0
        if self.is_classifier:
            avg = oob_sum[seen] / oob_count[seen, None]
            pred = self.classes_[np.argmax(avg, axis=1)]
            return float((pred == y[seen]).mean())
        avg = oob_sum[seen] / oob_count[seen]
        ss_res = float(((y[seen] - avg) ** 2).sum())
        ss_tot = float(((y[seen] - y[seen].mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean impurity decrease across the ensemble."""
        imps = np.array([t.feature_importances_ for t in self.estimators_])
        mean = imps.mean(axis=0)
        s = mean.sum()
        return mean / s if s > 0 else mean


class RandomForestClassifier(_BaseForest):
    tree_class = DecisionTreeClassifier
    is_classifier = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        total = np.zeros((len(X), len(self.classes_)))
        for tree in self.estimators_:
            # each tree may have seen a subset of classes; map onto the
            # full class list before averaging
            proba = tree.predict_proba(X)
            if len(tree.classes_) == len(self.classes_):
                total += proba
            else:
                cols = np.searchsorted(self.classes_, tree.classes_)
                total[:, cols] += proba
        return total / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class RandomForestRegressor(_BaseForest):
    tree_class = DecisionTreeRegressor
    is_classifier = False

    def __init__(self, *args, max_features=1.0, **kwargs):
        # regression forests use all features per split by default
        super().__init__(*args, max_features=max_features, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.mean([t.predict(X) for t in self.estimators_], axis=0)
