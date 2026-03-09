## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.3243 | 0.0320 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6526 | 0.0846 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.012x
Bun predict speedup vs scikit-learn: 2.648x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5624 | 0.0454 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.0214 | 0.1282 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.294x
Bun predict speedup vs scikit-learn: 2.823x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.2726 | 0.0265 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8494 | 0.1459 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 38.3508 | 1.6108 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 109.5323 | 6.7058 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.453x
DecisionTree predict speedup vs scikit-learn: 5.504x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.856x
RandomForest predict speedup vs scikit-learn: 4.163x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.2726 | 0.0265 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5232 | 0.0497 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8494 | 0.1459 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 38.3508 | 1.6108 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 32.6420 | 0.9121 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 109.5323 | 6.7058 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.835x
DecisionTree zig/js predict speedup: 0.533x
RandomForest zig/js fit speedup: 1.175x
RandomForest zig/js predict speedup: 1.766x
Snapshot generated at: 2026-03-09T09:51:51.060Z
