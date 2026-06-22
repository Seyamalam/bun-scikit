## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2401 | 0.0156 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.7694 | 0.1177 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 3.204x
Bun predict speedup vs scikit-learn: 7.542x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.6184 | 0.0517 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.7551 | 0.2020 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.702x
Bun predict speedup vs scikit-learn: 3.904x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.1476 | 0.0214 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9713 | 0.1709 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 37.5771 | 1.6305 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 116.0787 | 7.0357 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.718x
DecisionTree predict speedup vs scikit-learn: 7.983x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 3.089x
RandomForest predict speedup vs scikit-learn: 4.315x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.1476 | 0.0214 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5558 | 0.0530 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9713 | 0.1709 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 37.5771 | 1.6305 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 33.3200 | 0.9300 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 116.0787 | 7.0357 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.738x
DecisionTree zig/js predict speedup: 0.404x
RandomForest zig/js fit speedup: 1.128x
RandomForest zig/js predict speedup: 1.753x
Snapshot generated at: 2026-06-22T10:58:36.662Z
