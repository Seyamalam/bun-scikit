## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2531 | 0.0187 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6585 | 0.0843 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.602x
Bun predict speedup vs scikit-learn: 4.507x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5169 | 0.0393 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.0574 | 0.1283 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.356x
Bun predict speedup vs scikit-learn: 3.266x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.2111 | 0.0241 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8494 | 0.1387 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 38.3944 | 1.7000 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 110.8501 | 7.0371 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.527x
DecisionTree predict speedup vs scikit-learn: 5.745x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.887x
RandomForest predict speedup vs scikit-learn: 4.139x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.2111 | 0.0241 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5628 | 0.0522 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8494 | 0.1387 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 38.3944 | 1.7000 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 32.9799 | 1.0096 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 110.8501 | 7.0371 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.775x
DecisionTree zig/js predict speedup: 0.462x
RandomForest zig/js fit speedup: 1.164x
RandomForest zig/js predict speedup: 1.684x
Snapshot generated at: 2026-03-23T10:02:41.269Z
