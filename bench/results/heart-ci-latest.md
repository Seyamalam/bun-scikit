## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2527 | 0.0172 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6551 | 0.0852 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.592x
Bun predict speedup vs scikit-learn: 4.966x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.5353 | 0.0483 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.0196 | 0.1261 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.315x
Bun predict speedup vs scikit-learn: 2.609x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.2098 | 0.0269 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8578 | 0.1457 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 37.8928 | 1.5517 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 109.9588 | 6.7548 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.536x
DecisionTree predict speedup vs scikit-learn: 5.413x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.902x
RandomForest predict speedup vs scikit-learn: 4.353x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.2098 | 0.0269 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.5682 | 0.0502 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 1.8578 | 0.1457 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 37.8928 | 1.5517 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 33.0182 | 0.9063 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 109.9588 | 6.7548 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.771x
DecisionTree zig/js predict speedup: 0.536x
RandomForest zig/js fit speedup: 1.148x
RandomForest zig/js predict speedup: 1.712x
Snapshot generated at: 2026-03-16T10:06:31.670Z
