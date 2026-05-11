## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2486 | 0.0192 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.5373 | 0.0658 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.162x
Bun predict speedup vs scikit-learn: 3.418x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.4888 | 0.0455 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.7337 | 0.1187 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.165x
Bun predict speedup vs scikit-learn: 2.612x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.2112 | 0.0237 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9692 | 0.1303 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 40.4303 | 1.8530 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 106.8550 | 5.3996 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.626x
DecisionTree predict speedup vs scikit-learn: 5.488x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.643x
RandomForest predict speedup vs scikit-learn: 2.914x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.2112 | 0.0237 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.6346 | 0.0517 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9692 | 0.1303 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 40.4303 | 1.8530 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 34.9852 | 1.0226 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 106.8550 | 5.3996 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.741x
DecisionTree zig/js predict speedup: 0.459x
RandomForest zig/js fit speedup: 1.156x
RandomForest zig/js predict speedup: 1.812x
Snapshot generated at: 2026-05-11T12:10:32.751Z
