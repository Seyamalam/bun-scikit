## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2692 | 0.0270 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6110 | 0.0936 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.270x
Bun predict speedup vs scikit-learn: 3.462x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.4786 | 0.0405 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.3437 | 0.1861 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 1.585x
Bun predict speedup vs scikit-learn: 4.593x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 1.2269 | 0.0228 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 2.0465 | 0.1620 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 40.0777 | 1.7938 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 112.0529 | 5.3854 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.668x
DecisionTree predict speedup vs scikit-learn: 7.105x
DecisionTree accuracy delta (bun - sklearn): 4.878e-3
DecisionTree f1 delta (bun - sklearn): 3.837e-3
RandomForest fit speedup vs scikit-learn: 2.796x
RandomForest predict speedup vs scikit-learn: 3.002x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 1.2269 | 0.0228 | 0.936585 | 0.937799 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 1.6327 | 0.0536 | 0.936585 | 0.937799 |
| DecisionTreeClassifier | python-scikit-learn | 2.0465 | 0.1620 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 40.0777 | 1.7938 | 0.995122 | 0.995261 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 35.0182 | 1.0119 | 1.000000 | 1.000000 |
| RandomForestClassifier | python-scikit-learn | 112.0529 | 5.3854 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 0.751x
DecisionTree zig/js predict speedup: 0.425x
RandomForest zig/js fit speedup: 1.144x
RandomForest zig/js predict speedup: 1.773x
Snapshot generated at: 2026-06-29T10:38:33.077Z
