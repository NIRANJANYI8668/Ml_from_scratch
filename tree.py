"""
CART decision trees, from scratch.

A decision tree is a greedy search: at every node, look at every feature and
every possible threshold, and pick the split that most reduces impurity.
Recurse until a stopping rule fires.

Impurity measures how mixed a node is:

    classification   Gini = 1 - SUM_k p_k^2      (0 when the node is pure)
    regression       MSE  = variance of y in the node

and a split is scored by the WEIGHTED impurity of its children:

    score = (n_left * imp_left + n_right * imp_right) / n
    gain  = imp_parent - score          <- maximise this

The naive implementation re-scans the node for every candidate threshold and
costs O(n^2) per feature.  The trick that makes trees fast, and the only
non-obvious part of this file, is to sort the feature once and sweep:

    sort samples by the feature value, then accumulate class counts (or sums
    of y and y^2) from left to right.  Every prefix of the sorted order IS a
    candidate left child, and its impurity follows from the running totals in
    O(1).  So all thresholds for a feature cost one sort, O(n log n), and the
    sweep itself is vectorised over every threshold at once.

That is the same idea sklearn's Cython splitter uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Node:
    """A node. Leaves have feature = -1 and carry a value."""
    feature: int = -1
    threshold: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None
    value: np.ndarray | float = field(default=0.0)
    n_samples: int = 0
    impurity: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0


class BaseDecisionTree:
    """Shared machinery. Subclasses supply the impurity and leaf value."""

    is_classifier = False

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: int | str | None = None,
                 random_state: int | None = None):
        self.max_depth = max_depth if max_depth is not None else 2 ** 31
        self.min_samples_split = max(2, min_samples_split)
        self.min_samples_leaf = max(1, min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Node | None = None
        self.n_features_: int = 0
        self.classes_: np.ndarray | None = None
        self._rng: np.random.Generator | None = None
        self._importances: np.ndarray | None = None

    # ---- to be provided by subclasses -----------------------------------

    def _prepare_target(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _node_impurity(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def _leaf_value(self, y: np.ndarray):
        raise NotImplementedError

    def _sweep_scores(self, y_sorted: np.ndarray, n: int) -> np.ndarray:
        """Weighted child impurity for every split position 1..n-1."""
        raise NotImplementedError

    # ---- fitting ---------------------------------------------------------

    def _resolve_max_features(self) -> int:
        mf = self.max_features
        if mf is None:
            return self.n_features_
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(self.n_features_)))
            if mf == "log2":
                return max(1, int(np.log2(self.n_features_)))
            raise ValueError(f"unknown max_features {mf!r}")
        if isinstance(mf, float):
            return max(1, int(mf * self.n_features_))
        return max(1, min(int(mf), self.n_features_))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Search features for the split with the largest impurity gain."""
        n = len(y)
        n_try = self._resolve_max_features()
        features = (self._rng.permutation(self.n_features_)[:n_try]
                    if n_try < self.n_features_ else np.arange(self.n_features_))

        best_feature, best_threshold, best_score = -1, 0.0, np.inf

        for feature in features:
            column = X[:, feature]
            order = np.argsort(column, kind="mergesort")
            x_sorted = column[order]
            y_sorted = y[order]

            # a split is only possible between two DIFFERENT values
            valid = x_sorted[:-1] < x_sorted[1:]
            # respect min_samples_leaf on both sides
            positions = np.arange(1, n)
            valid &= (positions >= self.min_samples_leaf)
            valid &= ((n - positions) >= self.min_samples_leaf)
            if not valid.any():
                continue

            scores = self._sweep_scores(y_sorted, n)
            scores = np.where(valid, scores, np.inf)

            k = int(np.argmin(scores))
            if scores[k] < best_score:
                best_score = float(scores[k])
                best_feature = int(feature)
                # threshold midway between the two bracketing values
                best_threshold = float((x_sorted[k] + x_sorted[k + 1]) / 2.0)

        return best_feature, best_threshold, best_score

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n = len(y)
        impurity = self._node_impurity(y)
        node = Node(value=self._leaf_value(y), n_samples=n, impurity=impurity)

        if depth >= self.max_depth or n < self.min_samples_split or impurity <= 1e-16:
            return node

        feature, threshold, score = self._best_split(X, y)
        if feature < 0 or not (impurity - score > 1e-16):
            return node                       # no split improves anything

        mask = X[:, feature] <= threshold
        if mask.all() or (~mask).all():
            return node                       # degenerate split, keep the leaf

        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        y_prepared = self._prepare_target(np.asarray(y))
        self.root_ = self._build(X, y_prepared, depth=0)
        self._importances = None
        return self

    # ---- prediction -------------------------------------------------------

    def _traverse(self, X: np.ndarray, node: Node, indices: np.ndarray, out: np.ndarray):
        """Push whole blocks of rows down the tree instead of one at a time."""
        if node.is_leaf:
            out[indices] = node.value
            return
        going_left = X[indices, node.feature] <= node.threshold
        if going_left.any():
            self._traverse(X, node.left, indices[going_left], out)
        if (~going_left).any():
            self._traverse(X, node.right, indices[~going_left], out)

    # ---- feature importance ----------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        """Total impurity decrease contributed by each feature, normalised.

        Exactly what sklearn reports: for every internal node, the impurity
        it removed weighted by how many samples passed through it.
        """
        if self._importances is not None:
            return self._importances

        importances = np.zeros(self.n_features_)
        total = self.root_.n_samples

        def walk(node: Node):
            if node.is_leaf:
                return
            left, right = node.left, node.right
            decrease = (
                node.n_samples * node.impurity
                - left.n_samples * left.impurity
                - right.n_samples * right.impurity
            ) / total
            importances[node.feature] += decrease
            walk(left)
            walk(right)

        walk(self.root_)
        s = importances.sum()
        self._importances = importances / s if s > 0 else importances
        return self._importances

    def get_depth(self) -> int:
        def depth_of(node: Node) -> int:
            return 0 if node.is_leaf else 1 + max(depth_of(node.left), depth_of(node.right))
        return depth_of(self.root_)

    def get_n_leaves(self) -> int:
        def count(node: Node) -> int:
            return 1 if node.is_leaf else count(node.left) + count(node.right)
        return count(self.root_)


class DecisionTreeClassifier(BaseDecisionTree):
    """CART classifier using the Gini index."""

    is_classifier = True

    def _prepare_target(self, y: np.ndarray) -> np.ndarray:
        self.classes_, encoded = np.unique(y, return_inverse=True)
        # one-hot, so the sweep can accumulate class counts with a cumsum
        onehot = np.zeros((len(encoded), len(self.classes_)))
        onehot[np.arange(len(encoded)), encoded] = 1.0
        return onehot

    def _node_impurity(self, y: np.ndarray) -> float:
        p = y.sum(axis=0) / len(y)
        return float(1.0 - (p ** 2).sum())

    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        return y.sum(axis=0) / len(y)        # class probabilities

    def _sweep_scores(self, y_sorted: np.ndarray, n: int) -> np.ndarray:
        cum = np.cumsum(y_sorted, axis=0)[:-1]          # (n-1, n_classes)
        total = cum[-1] + y_sorted[-1]
        n_left = np.arange(1, n, dtype=float)[:, None]
        n_right = n - n_left

        gini_left = 1.0 - ((cum / n_left) ** 2).sum(axis=1)
        gini_right = 1.0 - (((total - cum) / n_right) ** 2).sum(axis=1)
        return (n_left.ravel() * gini_left + n_right.ravel() * gini_right) / n

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros((len(X), len(self.classes_)))
        self._traverse(X, self.root_, np.arange(len(X)), out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class DecisionTreeRegressor(BaseDecisionTree):
    """CART regressor using mean squared error."""

    def _prepare_target(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float).ravel()

    def _node_impurity(self, y: np.ndarray) -> float:
        return float(y.var())

    def _leaf_value(self, y: np.ndarray) -> float:
        return float(y.mean())

    def _sweep_scores(self, y_sorted: np.ndarray, n: int) -> np.ndarray:
        # variance from running sums:  Var = E[y^2] - E[y]^2
        cs = np.cumsum(y_sorted)[:-1]
        cs2 = np.cumsum(y_sorted ** 2)[:-1]
        total, total2 = cs[-1] + y_sorted[-1], cs2[-1] + y_sorted[-1] ** 2

        n_left = np.arange(1, n, dtype=float)
        n_right = n - n_left

        var_left = cs2 / n_left - (cs / n_left) ** 2
        var_right = (total2 - cs2) / n_right - ((total - cs) / n_right) ** 2
        return (n_left * var_left + n_right * var_right) / n

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros(len(X))
        self._traverse(X, self.root_, np.arange(len(X)), out)
        return out
