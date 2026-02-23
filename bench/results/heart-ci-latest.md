## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.3331 | 0.0357 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6591 | 0.0841 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 1.979x
Bun predict speedup vs scikit-learn: 2.357x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 0.9534 | 0.0440 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.0643 | 0.1280 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 2.165x
Bun predict speedup vs scikit-learn: 2.906x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.3314 | 0.0314 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.8623 | 0.1368 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 35.4194 | 1.8532 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 110.2591 | 6.7844 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.399x
DecisionTree predict speedup vs scikit-learn: 4.354x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2
RandomForest fit speedup vs scikit-learn: 3.113x
RandomForest predict speedup vs scikit-learn: 3.661x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.3314 | 0.0314 | 0.946341 | 0.948837 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.2592 | 0.0178 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.8623 | 0.1368 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 35.4194 | 1.8532 | 0.990244 | 0.990566 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 35.3450 | 1.8522 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 110.2591 | 6.7844 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 1.057x
DecisionTree zig/js predict speedup: 1.761x
RandomForest zig/js fit speedup: 1.002x
RandomForest zig/js predict speedup: 1.001x
Snapshot generated at: 2026-02-23T17:29:07.160Z
