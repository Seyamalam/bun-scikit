## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.1903 | 0.0203 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3185 | 0.0381 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 1.674x
Bun predict speedup vs scikit-learn: 1.877x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 0.6098 | 0.0295 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.0860 | 0.0709 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.781x
Bun predict speedup vs scikit-learn: 2.402x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 0.9312 | 0.0226 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.4337 | 0.0986 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 28.1736 | 1.6744 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 72.9953 | 2.1550 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.540x
DecisionTree predict speedup vs scikit-learn: 4.375x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2
RandomForest fit speedup vs scikit-learn: 2.591x
RandomForest predict speedup vs scikit-learn: 1.287x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 0.9312 | 0.0226 | 0.946341 | 0.948837 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 0.8968 | 0.0153 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.4337 | 0.0986 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 28.1736 | 1.6744 | 0.990244 | 0.990566 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 26.0242 | 1.5464 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 72.9953 | 2.1550 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 1.038x
DecisionTree zig/js predict speedup: 1.474x
RandomForest zig/js fit speedup: 1.083x
RandomForest zig/js predict speedup: 1.083x
Snapshot generated at: 2026-02-23T17:21:32.297Z
