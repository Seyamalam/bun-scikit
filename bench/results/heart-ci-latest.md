## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2418 | 0.0183 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6580 | 0.0832 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.721x
Bun predict speedup vs scikit-learn: 4.551x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5128 | 0.0396 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.0244 | 0.1271 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.338x
Bun predict speedup vs scikit-learn: 3.212x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.1424 | 0.0246 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8683 | 0.1338 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 37.0133 | 1.5791 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 109.9669 | 6.7795 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.635x
DecisionTree predict speedup vs scikit-learn: 5.436x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.971x
RandomForest predict speedup vs scikit-learn: 4.293x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.1424 | 0.0246 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5268 | 0.0534 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8683 | 0.1338 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 37.0133 | 1.5791 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 32.9153 | 0.9150 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 109.9669 | 6.7795 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.748x
DecisionTree zig/js predict speedup: 0.461x
RandomForest zig/js fit speedup: 1.125x
RandomForest zig/js predict speedup: 1.726x
Snapshot generated at: 2026-04-20T10:47:32.030Z
