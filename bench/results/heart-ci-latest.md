## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.4384 | 0.0343 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.5416 | 0.0677 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 1.235x
Bun predict speedup vs scikit-learn: 1.977x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5080 | 0.0408 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.7238 | 0.1170 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.143x
Bun predict speedup vs scikit-learn: 2.870x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.3074 | 0.0234 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9711 | 0.1285 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 41.0094 | 2.0022 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 107.0371 | 5.3645 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.508x
DecisionTree predict speedup vs scikit-learn: 5.484x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.610x
RandomForest predict speedup vs scikit-learn: 2.679x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.3074 | 0.0234 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.6545 | 0.0539 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.9711 | 0.1285 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 41.0094 | 2.0022 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 35.2326 | 1.0628 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 107.0371 | 5.3645 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.790x
DecisionTree zig/js predict speedup: 0.435x
RandomForest zig/js fit speedup: 1.164x
RandomForest zig/js predict speedup: 1.884x
Snapshot generated at: 2026-03-02T09:49:08.763Z
