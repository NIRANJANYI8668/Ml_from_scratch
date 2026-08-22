#!/usr/bin/env python3
"""
Benchmark every from-scratch implementation against scikit-learn.

Two kinds of comparison, and the distinction matters:

  EXACT      Deterministic linear algebra -- least squares, ridge, PCA. There
             is one right answer, so our numbers must match sklearn's to
             floating-point precision. Anything above ~1e-8 is a bug.

  STATISTICAL  Algorithms with randomness or greedy tie-breaking -- k-means,
             trees, forests, neural nets. Two correct implementations will
             NOT produce identical models. What must match is predictive
             performance. Demanding bit-identical output here would be a
             misunderstanding of the algorithm.

Run:  python benchmark.py
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# our implementations
from scratch.linear import LinearRegression, LinearRegressionGD, Ridge
from scratch.pca import PCA
from scratch.kmeans import KMeans
from scratch.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scratch.forest import RandomForestClassifier, RandomForestRegressor
from scratch.mlp import MLPClassifier

# sklearn's
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import Ridge as SkRidge
from sklearn.decomposition import PCA as SkPCA
from sklearn.cluster import KMeans as SkKMeans
from sklearn.tree import DecisionTreeClassifier as SkDTC
from sklearn.tree import DecisionTreeRegressor as SkDTR
from sklearn.ensemble import RandomForestClassifier as SkRFC
from sklearn.ensemble import RandomForestRegressor as SkRFR
from sklearn.neural_network import MLPClassifier as SkMLP

SEED = 0
EXACT_TOL = 1e-8          # deterministic algorithms must match this closely
PARITY_TOL = 0.05         # stochastic ones: metric within 5 percentage points


@dataclass
class Result:
    name: str
    dataset: str
    kind: str                     # "exact" or "statistical"
    metric_name: str
    ours: float
    sklearn: float
    agreement: float              # max abs difference, or metric gap
    agreement_label: str
    t_ours: float
    t_sklearn: float
    passed: bool = field(init=False)

    def __post_init__(self):
        tol = EXACT_TOL if self.kind == "exact" else PARITY_TOL
        self.passed = bool(self.agreement <= tol)

    @property
    def speed_ratio(self) -> float:
        return self.t_ours / max(self.t_sklearn, 1e-9)


def timed(fn, repeats: int = 1):
    """Run fn, return (result, best wall time)."""
    best, out = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def r2(y_true, y_pred) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


# ---------------------------------------------------------------------------
# individual benchmarks
# ---------------------------------------------------------------------------

def bench_linear_regression() -> list[Result]:
    X, y = datasets.load_diabetes(return_X_y=True)
    out = []

    for solver in ("svd", "normal"):
        ours, t_ours = timed(lambda: LinearRegression(solver=solver).fit(X, y), repeats=5)
        theirs, t_sk = timed(lambda: SkLinearRegression().fit(X, y), repeats=5)
        coef_diff = float(np.abs(ours.coef_ - theirs.coef_).max())
        int_diff = abs(float(ours.intercept_) - float(theirs.intercept_))
        out.append(Result(
            f"LinearRegression ({solver})", "diabetes", "exact", "R2",
            r2(y, ours.predict(X)), r2(y, theirs.predict(X)),
            max(coef_diff, int_diff), "max |coef| diff", t_ours, t_sk,
        ))

    # gradient descent should reach the same optimum, just slowly
    ours, t_ours = timed(lambda: LinearRegressionGD(lr="auto", n_iter=50000).fit(X, y))
    theirs, t_sk = timed(lambda: SkLinearRegression().fit(X, y), repeats=5)
    out.append(Result(
        "LinearRegression (gradient descent)", "diabetes", "statistical", "R2",
        r2(y, ours.predict(X)), r2(y, theirs.predict(X)),
        abs(r2(y, ours.predict(X)) - r2(y, theirs.predict(X))), "R2 gap",
        t_ours, t_sk,
    ))
    return out


def bench_ridge() -> list[Result]:
    X, y = datasets.load_diabetes(return_X_y=True)
    out = []
    for alpha in (0.1, 10.0):
        ours, t_ours = timed(lambda: Ridge(alpha=alpha).fit(X, y), repeats=5)
        theirs, t_sk = timed(lambda: SkRidge(alpha=alpha, solver="cholesky").fit(X, y),
                             repeats=5)
        diff = float(np.abs(ours.coef_ - theirs.coef_).max())
        out.append(Result(
            f"Ridge (alpha={alpha})", "diabetes", "exact", "R2",
            r2(y, ours.predict(X)), r2(y, theirs.predict(X)),
            diff, "max |coef| diff", t_ours, t_sk,
        ))
    return out


def bench_pca() -> list[Result]:
    X, _ = datasets.load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    k = 5

    ours, t_ours = timed(lambda: PCA(n_components=k).fit(X), repeats=5)
    theirs, t_sk = timed(lambda: SkPCA(n_components=k, svd_solver="full").fit(X),
                         repeats=5)

    comp_diff = float(np.abs(ours.components_ - theirs.components_).max())
    var_diff = float(np.abs(ours.explained_variance_ratio_
                            - theirs.explained_variance_ratio_).max())
    return [Result(
        f"PCA (k={k})", "breast_cancer", "exact", "explained var",
        float(ours.explained_variance_ratio_.sum()),
        float(theirs.explained_variance_ratio_.sum()),
        max(comp_diff, var_diff), "max |component| diff", t_ours, t_sk,
    )]


def bench_kmeans() -> list[Result]:
    X, _ = datasets.make_blobs(n_samples=3000, centers=6, n_features=8,
                               cluster_std=1.5, random_state=SEED)
    k = 6
    ours, t_ours = timed(lambda: KMeans(n_clusters=k, n_init=10,
                                        random_state=SEED).fit(X))
    theirs, t_sk = timed(lambda: SkKMeans(n_clusters=k, n_init=10,
                                          random_state=SEED).fit(X))
    # inertia is the objective both are minimising -- compare it relatively
    rel_gap = abs(ours.inertia_ - theirs.inertia_) / theirs.inertia_
    return [Result(
        f"KMeans (k={k})", "blobs 3000x8", "statistical", "inertia (lower=better)",
        ours.inertia_, float(theirs.inertia_), rel_gap, "relative inertia gap",
        t_ours, t_sk,
    )]


def bench_decision_tree() -> list[Result]:
    out = []

    # classification
    X, y = datasets.load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)
    ours, t_ours = timed(lambda: DecisionTreeClassifier(max_depth=6,
                                                        random_state=SEED).fit(Xtr, ytr))
    theirs, t_sk = timed(lambda: SkDTC(max_depth=6, random_state=SEED).fit(Xtr, ytr))
    a_ours, a_sk = accuracy(yte, ours.predict(Xte)), accuracy(yte, theirs.predict(Xte))
    out.append(Result(
        "DecisionTreeClassifier (depth 6)", "breast_cancer", "statistical", "accuracy",
        a_ours, a_sk, abs(a_ours - a_sk), "accuracy gap", t_ours, t_sk,
    ))

    # regression
    Xr, yr = datasets.load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.3, random_state=SEED)
    ours, t_ours = timed(lambda: DecisionTreeRegressor(max_depth=4,
                                                       random_state=SEED).fit(Xtr, ytr))
    theirs, t_sk = timed(lambda: SkDTR(max_depth=4, random_state=SEED).fit(Xtr, ytr))
    r_ours, r_sk = r2(yte, ours.predict(Xte)), r2(yte, theirs.predict(Xte))
    out.append(Result(
        "DecisionTreeRegressor (depth 4)", "diabetes", "statistical", "R2",
        r_ours, r_sk, abs(r_ours - r_sk), "R2 gap", t_ours, t_sk,
    ))
    return out


def bench_random_forest() -> list[Result]:
    out = []

    X, y = datasets.load_wine(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)
    ours, t_ours = timed(lambda: RandomForestClassifier(
        n_estimators=100, max_features="sqrt", oob_score=True,
        random_state=SEED).fit(Xtr, ytr))
    theirs, t_sk = timed(lambda: SkRFC(n_estimators=100, max_features="sqrt",
                                       oob_score=True, random_state=SEED).fit(Xtr, ytr))
    a_ours, a_sk = accuracy(yte, ours.predict(Xte)), accuracy(yte, theirs.predict(Xte))
    out.append(Result(
        "RandomForestClassifier (100 trees)", "wine", "statistical", "accuracy",
        a_ours, a_sk, abs(a_ours - a_sk), "accuracy gap", t_ours, t_sk,
    ))

    Xr, yr = datasets.load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.3, random_state=SEED)
    ours, t_ours = timed(lambda: RandomForestRegressor(
        n_estimators=50, max_depth=8, random_state=SEED).fit(Xtr, ytr))
    theirs, t_sk = timed(lambda: SkRFR(n_estimators=50, max_depth=8,
                                       random_state=SEED).fit(Xtr, ytr))
    r_ours, r_sk = r2(yte, ours.predict(Xte)), r2(yte, theirs.predict(Xte))
    out.append(Result(
        "RandomForestRegressor (50 trees)", "diabetes", "statistical", "R2",
        r_ours, r_sk, abs(r_ours - r_sk), "R2 gap", t_ours, t_sk,
    ))
    return out


def bench_mlp() -> list[Result]:
    X, y = datasets.load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)

    ours, t_ours = timed(lambda: MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
        learning_rate_init=1e-3, batch_size=64, max_iter=150,
        random_state=SEED).fit(Xtr, ytr))
    theirs, t_sk = timed(lambda: SkMLP(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
        learning_rate_init=1e-3, batch_size=64, max_iter=150,
        solver="adam", random_state=SEED).fit(Xtr, ytr))

    a_ours, a_sk = accuracy(yte, ours.predict(Xte)), accuracy(yte, theirs.predict(Xte))
    return [Result(
        "MLPClassifier (64-32, Adam)", "digits", "statistical", "accuracy",
        a_ours, a_sk, abs(a_ours - a_sk), "accuracy gap", t_ours, t_sk,
    )]


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_table(results: list[Result]):
    for kind, title, blurb in [
        ("exact", "EXACT AGREEMENT REQUIRED",
         "Deterministic linear algebra: one right answer, must match to precision."),
        ("statistical", "PERFORMANCE PARITY REQUIRED",
         "Randomised or greedy: models differ, predictive quality must not."),
    ]:
        subset = [r for r in results if r.kind == kind]
        if not subset:
            continue
        print(f"\n{'=' * 100}\n{title}\n{blurb}\n{'=' * 100}")
        print(f"{'algorithm':<38}{'dataset':<15}{'ours':>13}{'sklearn':>13}"
              f"{'agreement':>13}{'speed':>9}  ")
        print("-" * 106)
        for r in subset:
            mark = "ok " if r.passed else "FAIL"
            speed = f"{r.speed_ratio:.1f}x"
            # large metrics (inertia) need different formatting from ratios
            fmt = "{:>13.4g}" if max(abs(r.ours), abs(r.sklearn)) >= 1e4 else "{:>13.4f}"
            print(f"{r.name:<38}{r.dataset:<15}"
                  f"{fmt.format(r.ours)}{fmt.format(r.sklearn)}"
                  f"{r.agreement:>13.2e}{speed:>9}  {mark}")


def plot_results(results: list[Result], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = [r.name for r in results]
    ypos = np.arange(len(names))

    ratios = [r.speed_ratio for r in results]
    colours = ["tab:red" if x > 1 else "tab:green" for x in ratios]
    ax1.barh(ypos, ratios, color=colours, alpha=0.8)
    ax1.axvline(1.0, color="black", linewidth=1.2)
    ax1.set_yticks(ypos, names, fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("time ours / time sklearn   (log scale, 1.0 = equal)")
    ax1.set_title("Speed: pure numpy vs compiled sklearn", fontsize=11)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    # Only metrics on a comparable 0-1 scale go on the parity plot. k-means
    # inertia is in the tens of thousands and would flatten everything else
    # into a corner; it is reported in the table instead.
    scatter = [r for r in results if "inertia" not in r.metric_name]

    ax2.scatter([r.sklearn for r in scatter], [r.ours for r in scatter],
                s=80, c="tab:blue", zorder=3, edgecolors="white", linewidths=0.8)
    for r in scatter:
        short = r.name.split(" (")[0]
        ax2.annotate(short, (r.sklearn, r.ours), fontsize=7,
                     xytext=(5, -3), textcoords="offset points", alpha=0.85)

    lo = min(min(r.ours, r.sklearn) for r in scatter)
    hi = max(max(r.ours, r.sklearn) for r in scatter)
    pad = 0.08 * (hi - lo) + 0.02
    lims = [lo - pad, hi + pad]
    ax2.plot(lims, lims, "k--", linewidth=1, label="perfect agreement")
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_xlabel("sklearn metric (R2 or accuracy)")
    ax2.set_ylabel("our metric")
    ax2.set_title("Quality parity -- on the line means identical performance",
                  fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(results: list[Result], path: str):
    lines = ["# Benchmark: from scratch vs scikit-learn\n",
             "| algorithm | dataset | metric | ours | sklearn | agreement | ours/sklearn time | pass |",
             "|---|---|---|---:|---:|---:|---:|:--:|"]
    for r in results:
        lines.append(
            f"| {r.name} | {r.dataset} | {r.metric_name} | {r.ours:.4f} | "
            f"{r.sklearn:.4f} | {r.agreement:.2e} ({r.agreement_label}) | "
            f"{r.speed_ratio:.1f}x | {'yes' if r.passed else 'NO'} |"
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    from pathlib import Path
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("Benchmarking from-scratch implementations against scikit-learn")
    print(f"numpy {np.__version__}\n")

    results: list[Result] = []
    for label, fn in [
        ("linear regression", bench_linear_regression),
        ("ridge", bench_ridge),
        ("PCA", bench_pca),
        ("k-means", bench_kmeans),
        ("decision trees", bench_decision_tree),
        ("random forests", bench_random_forest),
        ("neural network", bench_mlp),
    ]:
        print(f"  running {label} ...", flush=True)
        results.extend(fn())

    print_table(results)

    n_pass = sum(r.passed for r in results)
    print(f"\n{n_pass}/{len(results)} checks passed")

    exact = [r for r in results if r.kind == "exact"]
    if exact:
        worst = max(r.agreement for r in exact)
        print(f"Worst disagreement on a deterministic algorithm: {worst:.2e} "
              f"(float64 epsilon is {np.finfo(float).eps:.1e})")

    slowest = max(results, key=lambda r: r.speed_ratio)
    fastest = min(results, key=lambda r: r.speed_ratio)
    print(f"Slowest relative to sklearn: {slowest.name} ({slowest.speed_ratio:.0f}x)")
    print(f"Closest to sklearn:          {fastest.name} ({fastest.speed_ratio:.1f}x)")

    plot_results(results, str(results_dir / "benchmark.png"))
    write_markdown(results, str(results_dir / "benchmark.md"))
    print(f"\nWrote {results_dir}/benchmark.png and benchmark.md")


if __name__ == "__main__":
    main()
