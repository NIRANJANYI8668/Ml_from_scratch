"""
Automated agreement tests.

    pip install pytest && pytest -v

The tolerances encode a real distinction:

  * Deterministic algorithms (least squares, ridge, PCA) have exactly one
    correct answer. We assert agreement with sklearn near machine precision.
    If one of these ever fails, there is a bug.

  * Randomised or greedy algorithms (k-means, trees, forests, MLP) do not
    have a unique answer. Two correct implementations disagree on the model
    but must agree on quality. We assert performance parity instead.

  * Some properties are guaranteed by the mathematics regardless of
    implementation -- k-means inertia must decrease monotonically, PCA must
    be the optimal rank-k approximation, a tree must fit its training data
    perfectly if grown deep enough. Those are tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import Ridge as SkRidge
from sklearn.decomposition import PCA as SkPCA
from sklearn.cluster import KMeans as SkKMeans
from sklearn.tree import DecisionTreeClassifier as SkDTC
from sklearn.ensemble import RandomForestClassifier as SkRFC

from scratch.linear import LinearRegression, LinearRegressionGD, Ridge
from scratch.pca import PCA
from scratch.kmeans import KMeans, _squared_distances
from scratch.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scratch.forest import RandomForestClassifier
from scratch.mlp import MLPClassifier

SEED = 0


@pytest.fixture(scope="module")
def regression_data():
    return datasets.load_diabetes(return_X_y=True)


@pytest.fixture(scope="module")
def classification_data():
    return datasets.load_breast_cancer(return_X_y=True)


# ---------------------------------------------------------------------------
# exact: deterministic linear algebra
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ["svd", "normal"])
def test_linear_regression_matches_sklearn(regression_data, solver):
    X, y = regression_data
    ours = LinearRegression(solver=solver).fit(X, y)
    theirs = SkLinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
def test_ridge_matches_sklearn(regression_data, alpha):
    X, y = regression_data
    ours = Ridge(alpha=alpha).fit(X, y)
    theirs = SkRidge(alpha=alpha, solver="cholesky").fit(X, y)
    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-8, atol=1e-8)


def test_ridge_shrinks_toward_zero(regression_data):
    """Larger alpha must give smaller coefficients. This is what ridge IS."""
    X, y = regression_data
    norms = [np.linalg.norm(Ridge(alpha=a).fit(X, y).coef_)
             for a in [0.01, 1.0, 100.0, 10_000.0]]
    assert all(a > b for a, b in zip(norms, norms[1:])), norms


def test_gradient_descent_reaches_closed_form(regression_data):
    """Three routes, one answer."""
    X, y = regression_data
    exact = LinearRegression().fit(X, y)
    iterative = LinearRegressionGD(lr="auto", n_iter=50_000).fit(X, y)
    np.testing.assert_allclose(iterative.coef_, exact.coef_, rtol=1e-4, atol=1e-4)


def test_gradient_descent_warns_above_stability_limit(regression_data):
    """lr >= 2/L must diverge, and we should say so before it does."""
    X, y = regression_data
    probe = LinearRegressionGD(lr="auto", n_iter=10).fit(X, y)
    unstable = 1.01 * probe.max_stable_lr_
    with pytest.warns(RuntimeWarning, match="diverge"):
        LinearRegressionGD(lr=unstable, n_iter=200).fit(X, y)


def test_pca_matches_sklearn(classification_data):
    X, _ = classification_data
    X = StandardScaler().fit_transform(X)
    ours = PCA(n_components=5).fit(X)
    theirs = SkPCA(n_components=5, svd_solver="full").fit(X)
    np.testing.assert_allclose(ours.components_, theirs.components_,
                               rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(ours.explained_variance_ratio_,
                               theirs.explained_variance_ratio_, rtol=1e-8, atol=1e-8)


def test_pca_components_are_orthonormal(classification_data):
    X, _ = classification_data
    comps = PCA(n_components=5).fit(StandardScaler().fit_transform(X)).components_
    np.testing.assert_allclose(comps @ comps.T, np.eye(5), atol=1e-10)


def test_pca_is_optimal_rank_k(classification_data):
    """Eckart-Young: no rank-k projection can beat PCA's reconstruction error."""
    X, _ = classification_data
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=4).fit(X)
    best = pca.reconstruction_error(X)

    rng = np.random.default_rng(SEED)
    for _ in range(20):
        # a random orthonormal 4-dimensional basis
        Q, _ = np.linalg.qr(rng.normal(size=(X.shape[1], 4)))
        Xc = X - X.mean(axis=0)
        recon = Xc @ Q @ Q.T
        assert ((Xc - recon) ** 2).mean() >= best - 1e-12


# ---------------------------------------------------------------------------
# statistical: randomised and greedy algorithms
# ---------------------------------------------------------------------------

def test_kmeans_inertia_competitive_with_sklearn():
    X, _ = datasets.make_blobs(n_samples=2000, centers=5, n_features=6,
                               cluster_std=1.2, random_state=SEED)
    ours = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit(X)
    theirs = SkKMeans(n_clusters=5, n_init=10, random_state=SEED).fit(X)
    assert ours.inertia_ <= theirs.inertia_ * 1.02


def test_kmeans_inertia_decreases_monotonically():
    """Lloyd's algorithm cannot increase the objective. If it does, it's wrong."""
    X, _ = datasets.make_blobs(n_samples=800, centers=4, n_features=3,
                               random_state=SEED)
    km = KMeans(n_clusters=4, n_init=1, random_state=SEED)
    rng = np.random.default_rng(SEED)
    centers = km._kmeans_plusplus(X, rng)

    previous = np.inf
    for _ in range(25):
        d = _squared_distances(X, centers)
        labels = np.argmin(d, axis=1)
        inertia = d[np.arange(len(X)), labels].sum()
        assert inertia <= previous + 1e-9, f"inertia rose: {previous} -> {inertia}"
        previous = inertia
        for j in range(4):
            members = X[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)


def test_decision_tree_parity(classification_data):
    X, y = classification_data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    ours = DecisionTreeClassifier(max_depth=6, random_state=SEED).fit(Xtr, ytr)
    theirs = SkDTC(max_depth=6, random_state=SEED).fit(Xtr, ytr)
    acc_ours = (ours.predict(Xte) == yte).mean()
    acc_theirs = (theirs.predict(Xte) == yte).mean()
    assert acc_ours >= acc_theirs - 0.05


def test_unpruned_tree_fits_training_data(classification_data):
    """A tree with no depth limit must reach zero training error on
    non-contradictory data. If it does not, the splitter is broken."""
    X, y = classification_data
    tree = DecisionTreeClassifier(random_state=SEED).fit(X, y)
    assert (tree.predict(X) == y).mean() == 1.0


def test_tree_respects_max_depth(classification_data):
    X, y = classification_data
    for depth in (1, 3, 5):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X, y)
        assert tree.get_depth() <= depth


def test_tree_respects_min_samples_leaf(regression_data):
    X, y = regression_data
    tree = DecisionTreeRegressor(min_samples_leaf=25, random_state=SEED).fit(X, y)

    def check(node):
        if node.is_leaf:
            assert node.n_samples >= 25
            return
        check(node.left)
        check(node.right)

    check(tree.root_)


def test_forest_beats_single_tree(classification_data):
    """Averaging must reduce variance. If the forest is not better than one
    tree, the randomisation is not doing its job."""
    X, y = classification_data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    tree = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
    forest = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(Xtr, ytr)
    assert (forest.predict(Xte) == yte).mean() >= (tree.predict(Xte) == yte).mean()


def test_forest_parity(classification_data):
    X, y = classification_data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    ours = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(Xtr, ytr)
    theirs = SkRFC(n_estimators=100, random_state=SEED).fit(Xtr, ytr)
    assert (ours.predict(Xte) == yte).mean() >= (theirs.predict(Xte) == yte).mean() - 0.03


def test_oob_score_approximates_test_score(classification_data):
    """Out-of-bag error is a free validation estimate. It should land close
    to the real held-out score."""
    X, y = classification_data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    forest = RandomForestClassifier(n_estimators=200, oob_score=True,
                                    random_state=SEED).fit(Xtr, ytr)
    test_score = (forest.predict(Xte) == yte).mean()
    assert abs(forest.oob_score_ - test_score) < 0.08


def test_feature_importances_sum_to_one(classification_data):
    X, y = classification_data
    tree = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(X, y)
    forest = RandomForestClassifier(n_estimators=20, random_state=SEED).fit(X, y)
    assert abs(tree.feature_importances_.sum() - 1.0) < 1e-10
    assert abs(forest.feature_importances_.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# the neural network
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("activation", ["tanh", "logistic", "relu"])
def test_backprop_matches_finite_differences(activation):
    """The one test that actually validates the calculus."""
    X, y = datasets.make_classification(n_samples=50, n_features=8,
                                        n_informative=6, n_classes=3,
                                        random_state=SEED)
    net = MLPClassifier(hidden_layer_sizes=(10, 7), activation=activation,
                        alpha=1e-3, random_state=SEED)
    assert net.gradient_check(X, y) < 1e-8


def test_mlp_parity_on_digits():
    X, y = datasets.load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    net = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150,
                        random_state=SEED).fit(Xtr, ytr)
    assert (net.predict(Xte) == yte).mean() > 0.93


def test_mlp_loss_decreases():
    X, y = datasets.load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    net = MLPClassifier(hidden_layer_sizes=(32,), max_iter=40,
                        random_state=SEED).fit(X, y)
    assert net.loss_curve_[-1] < net.loss_curve_[0] * 0.5


def test_softmax_rows_sum_to_one():
    from scratch.mlp import softmax
    rng = np.random.default_rng(SEED)
    z = rng.normal(scale=50.0, size=(100, 7))     # large values: overflow check
    p = softmax(z)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert np.isfinite(p).all()
