## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.3633 | 0.0356 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6747 | 0.0863 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 1.857x
Bun predict speedup vs scikit-learn: 2.421x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5158 | 0.0432 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.0346 | 0.1282 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.342x
Bun predict speedup vs scikit-learn: 2.966x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.6688 | 0.0348 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8817 | 0.1397 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 38.6671 | 1.7740 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 112.1050 | 7.0481 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.128x
DecisionTree predict speedup vs scikit-learn: 4.017x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.899x
RandomForest predict speedup vs scikit-learn: 3.973x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.6688 | 0.0348 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5479 | 0.0531 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8817 | 0.1397 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 38.6671 | 1.7740 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 33.2867 | 0.9900 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 112.1050 | 7.0481 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 1.078x
DecisionTree zig/js predict speedup: 0.654x
RandomForest zig/js fit speedup: 1.162x
RandomForest zig/js predict speedup: 1.792x
Snapshot generated at: 2026-03-30T10:16:51.038Z
