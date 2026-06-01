## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2581 | 0.0261 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6589 | 0.0843 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.553x
Bun predict speedup vs scikit-learn: 3.223x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5123 | 0.0385 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.0461 | 0.1279 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.353x
Bun predict speedup vs scikit-learn: 3.326x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.1630 | 0.0213 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8720 | 0.1382 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 36.9784 | 1.6292 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 111.6597 | 6.8653 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.610x
DecisionTree predict speedup vs scikit-learn: 6.500x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 3.020x
RandomForest predict speedup vs scikit-learn: 4.214x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.1630 | 0.0213 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5609 | 0.0530 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8720 | 0.1382 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 36.9784 | 1.6292 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 33.2292 | 0.9167 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 111.6597 | 6.8653 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.745x
DecisionTree zig/js predict speedup: 0.401x
RandomForest zig/js fit speedup: 1.113x
RandomForest zig/js predict speedup: 1.777x
Snapshot generated at: 2026-06-01T10:47:30.287Z
