## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.4085 | 0.0352 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.5459 | 0.0681 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 1.336x
Bun predict speedup vs scikit-learn: 1.936x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.4752 | 0.0447 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.7534 | 0.1189 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.189x
Bun predict speedup vs scikit-learn: 2.661x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.2585 | 0.0241 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9844 | 0.1336 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 40.2303 | 1.8067 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 108.0600 | 5.4068 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.577x
DecisionTree predict speedup vs scikit-learn: 5.548x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.686x
RandomForest predict speedup vs scikit-learn: 2.993x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.2585 | 0.0241 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.6318 | 0.0535 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9844 | 0.1336 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 40.2303 | 1.8067 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 34.9185 | 1.0201 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 108.0600 | 5.4068 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.771x
DecisionTree zig/js predict speedup: 0.450x
RandomForest zig/js fit speedup: 1.152x
RandomForest zig/js predict speedup: 1.771x
Snapshot generated at: 2026-05-25T10:36:00.575Z
