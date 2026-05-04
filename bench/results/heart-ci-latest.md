## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.3270 | 0.0320 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6726 | 0.0876 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.057x
Bun predict speedup vs scikit-learn: 2.733x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5529 | 0.0413 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.0548 | 0.1289 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.323x
Bun predict speedup vs scikit-learn: 3.118x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.3394 | 0.0311 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8786 | 0.1359 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 37.5564 | 1.7013 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 111.8150 | 6.9574 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.403x
DecisionTree predict speedup vs scikit-learn: 4.370x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.977x
RandomForest predict speedup vs scikit-learn: 4.090x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.3394 | 0.0311 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5569 | 0.0558 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8786 | 0.1359 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 37.5564 | 1.7013 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 33.1039 | 0.9826 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 111.8150 | 6.9574 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.860x
DecisionTree zig/js predict speedup: 0.557x
RandomForest zig/js fit speedup: 1.135x
RandomForest zig/js predict speedup: 1.731x
Snapshot generated at: 2026-05-04T11:05:20.400Z
