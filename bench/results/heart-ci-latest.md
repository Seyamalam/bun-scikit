## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2861 | 0.0290 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.7717 | 0.1162 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.698x
Bun predict speedup vs scikit-learn: 4.007x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5237 | 0.0413 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.7612 | 0.2018 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.812x
Bun predict speedup vs scikit-learn: 4.890x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.2235 | 0.0255 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9807 | 0.1716 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 37.6732 | 1.9435 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 118.4305 | 6.9652 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.619x
DecisionTree predict speedup vs scikit-learn: 6.737x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 3.144x
RandomForest predict speedup vs scikit-learn: 3.584x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.2235 | 0.0255 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5669 | 0.0537 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9807 | 0.1716 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 37.6732 | 1.9435 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 33.6108 | 0.9130 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 118.4305 | 6.9652 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.781x
DecisionTree zig/js predict speedup: 0.475x
RandomForest zig/js fit speedup: 1.121x
RandomForest zig/js predict speedup: 2.129x
Snapshot generated at: 2026-06-15T11:07:55.139Z
