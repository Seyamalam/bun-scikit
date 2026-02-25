## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.1763 | 0.0186 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3886 | 0.0452 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 2.204x
Bun predict speedup vs scikit-learn: 2.430x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 0.5275 | 0.0320 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.2934 | 0.0833 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 2.452x
Bun predict speedup vs scikit-learn: 2.601x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 0.8338 | 0.0209 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.3712 | 0.0928 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 31.2166 | 1.7649 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 199.6324 | 6.9251 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.645x
DecisionTree predict speedup vs scikit-learn: 4.438x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2
RandomForest fit speedup vs scikit-learn: 6.395x
RandomForest predict speedup vs scikit-learn: 3.924x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 0.8338 | 0.0209 | 0.946341 | 0.948837 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 0.4583 | 0.0339 | 0.892683 | 0.899083 |
| DecisionTreeClassifier | python-scikit-learn | 1.3712 | 0.0928 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 31.2166 | 1.7649 | 0.990244 | 0.990566 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 11.7783 | 0.7824 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 199.6324 | 6.9251 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 1.819x
DecisionTree zig/js predict speedup: 0.617x
RandomForest zig/js fit speedup: 2.650x
RandomForest zig/js predict speedup: 2.256x
Snapshot generated at: 2026-02-25T19:47:51.136Z
