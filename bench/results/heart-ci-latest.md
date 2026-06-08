## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2748 | 0.0277 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6105 | 0.0939 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.222x
Bun predict speedup vs scikit-learn: 3.383x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.4746 | 0.0393 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.3055 | 0.1855 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.564x
Bun predict speedup vs scikit-learn: 4.716x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.2197 | 0.0221 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 2.0457 | 0.1625 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 40.0082 | 1.8115 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 112.1379 | 5.4174 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.677x
DecisionTree predict speedup vs scikit-learn: 7.338x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.803x
RandomForest predict speedup vs scikit-learn: 2.991x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.2197 | 0.0221 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.6750 | 0.0555 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 2.0457 | 0.1625 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 40.0082 | 1.8115 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 34.9334 | 1.0086 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 112.1379 | 5.4174 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.728x
DecisionTree zig/js predict speedup: 0.399x
RandomForest zig/js fit speedup: 1.145x
RandomForest zig/js predict speedup: 1.796x
Snapshot generated at: 2026-06-08T10:38:56.781Z
