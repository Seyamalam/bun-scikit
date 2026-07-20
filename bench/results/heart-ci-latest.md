## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2884 | 0.0200 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.8330 | 0.1388 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.888x
Bun predict speedup vs scikit-learn: 6.954x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5577 | 0.0462 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.9072 | 0.2266 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.866x
Bun predict speedup vs scikit-learn: 4.905x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.2972 | 0.0263 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 2.0245 | 0.1924 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 37.7564 | 1.7113 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 120.1967 | 6.9777 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.561x
DecisionTree predict speedup vs scikit-learn: 7.306x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 3.183x
RandomForest predict speedup vs scikit-learn: 4.077x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.2972 | 0.0263 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5803 | 0.0522 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 2.0245 | 0.1924 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 37.7564 | 1.7113 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 33.1788 | 0.9160 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 120.1967 | 6.9777 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.821x
DecisionTree zig/js predict speedup: 0.504x
RandomForest zig/js fit speedup: 1.138x
RandomForest zig/js predict speedup: 1.868x
Snapshot generated at: 2026-07-20T11:44:17.006Z
