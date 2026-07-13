## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.3561 | 0.0303 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.8232 | 0.1363 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.312x
Bun predict speedup vs scikit-learn: 4.495x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5317 | 0.0433 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.9048 | 0.2267 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.897x
Bun predict speedup vs scikit-learn: 5.231x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.1371 | 0.0216 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 2.0133 | 0.1881 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 37.3353 | 1.7685 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 118.6281 | 7.0250 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.770x
DecisionTree predict speedup vs scikit-learn: 8.692x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 3.177x
RandomForest predict speedup vs scikit-learn: 3.972x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.1371 | 0.0216 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5648 | 0.0539 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 2.0133 | 0.1881 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 37.3353 | 1.7685 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 33.8750 | 0.9897 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 118.6281 | 7.0250 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.727x
DecisionTree zig/js predict speedup: 0.401x
RandomForest zig/js fit speedup: 1.102x
RandomForest zig/js predict speedup: 1.787x
Snapshot generated at: 2026-07-13T12:02:14.724Z
