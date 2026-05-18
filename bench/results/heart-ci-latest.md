## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2480 | 0.0270 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6593 | 0.0834 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.658x
Bun predict speedup vs scikit-learn: 3.091x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5181 | 0.0396 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.0270 | 0.1264 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.335x
Bun predict speedup vs scikit-learn: 3.196x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.1599 | 0.0229 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8612 | 0.1322 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 37.0508 | 1.7254 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 109.9205 | 6.8775 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.605x
DecisionTree predict speedup vs scikit-learn: 5.770x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.967x
RandomForest predict speedup vs scikit-learn: 3.986x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.1599 | 0.0229 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5207 | 0.0517 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8612 | 0.1322 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 37.0508 | 1.7254 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 33.2852 | 0.9485 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 109.9205 | 6.8775 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.763x
DecisionTree zig/js predict speedup: 0.444x
RandomForest zig/js fit speedup: 1.113x
RandomForest zig/js predict speedup: 1.819x
Snapshot generated at: 2026-05-18T10:32:30.122Z
