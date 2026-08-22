# Benchmark: from scratch vs scikit-learn

| algorithm | dataset | metric | ours | sklearn | agreement | ours/sklearn time | pass |
|---|---|---|---:|---:|---:|---:|:--:|
| LinearRegression (svd) | diabetes | R2 | 0.5177 | 0.5177 | 0.00e+00 (max |coef| diff) | 0.2x | yes |
| LinearRegression (normal) | diabetes | R2 | 0.5177 | 0.5177 | 2.50e-11 (max |coef| diff) | 0.1x | yes |
| LinearRegression (gradient descent) | diabetes | R2 | 0.5177 | 0.5177 | 7.15e-14 (R2 gap) | 188.8x | yes |
| Ridge (alpha=0.1) | diabetes | R2 | 0.5126 | 0.5126 | 2.84e-13 (max |coef| diff) | 0.1x | yes |
| Ridge (alpha=10.0) | diabetes | R2 | 0.1889 | 0.1889 | 2.84e-14 (max |coef| diff) | 0.1x | yes |
| PCA (k=5) | breast_cancer | explained var | 0.8473 | 0.8473 | 0.00e+00 (max |component| diff) | 0.7x | yes |
| KMeans (k=6) | blobs 3000x8 | inertia (lower=better) | 52966.1981 | 52966.1981 | 2.75e-16 (relative inertia gap) | 0.3x | yes |
| DecisionTreeClassifier (depth 6) | breast_cancer | accuracy | 0.9181 | 0.9064 | 1.17e-02 (accuracy gap) | 4.4x | yes |
| DecisionTreeRegressor (depth 4) | diabetes | R2 | 0.1165 | 0.1386 | 2.20e-02 (R2 gap) | 4.7x | yes |
| RandomForestClassifier (100 trees) | wine | accuracy | 1.0000 | 1.0000 | 0.00e+00 (accuracy gap) | 1.7x | yes |
| RandomForestRegressor (50 trees) | diabetes | R2 | 0.2983 | 0.2921 | 6.16e-03 (R2 gap) | 19.7x | yes |
| MLPClassifier (64-32, Adam) | digits | accuracy | 0.9722 | 0.9685 | 3.70e-03 (accuracy gap) | 1.5x | yes |
