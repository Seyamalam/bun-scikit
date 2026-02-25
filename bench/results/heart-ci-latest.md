## Regression (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.1728 | 0.0179 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3290 | 0.0391 | 0.117545 | 0.529539 |
Bun fit speedup vs scikit-learn: 1.904x
Bun predict speedup vs scikit-learn: 2.187x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13
## Classification (Heart Dataset)
| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 0.5310 | 0.0280 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.0794 | 0.0713 | 0.863415 | 0.875000 |
Bun fit speedup vs scikit-learn: 2.033x
Bun predict speedup vs scikit-learn: 2.548x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3
## Tree Classification (Heart Dataset)
| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) [js-fast] | bun-scikit | 0.8700 | 0.0195 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.3154 | 0.0864 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) [js-fast] | bun-scikit | 31.3548 | 1.9602 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 65.2760 | 2.0886 | 0.995122 | 0.995261 |
DecisionTree fit speedup vs scikit-learn: 1.512x
DecisionTree predict speedup vs scikit-learn: 4.419x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2
RandomForest fit speedup vs scikit-learn: 2.082x
RandomForest predict speedup vs scikit-learn: 1.065x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3
## Tree Backend Modes (Bun vs Bun vs sklearn)
| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | js-fast | 0.8700 | 0.0195 | 0.946341 | 0.948837 |
| DecisionTreeClassifier(maxDepth=8) | zig-tree | 0.4444 | 0.0355 | 0.892683 | 0.899083 |
| DecisionTreeClassifier | python-scikit-learn | 1.3154 | 0.0864 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | 31.3548 | 1.9602 | 0.990244 | 0.990566 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | 11.0824 | 0.8194 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 65.2760 | 2.0886 | 0.995122 | 0.995261 |
DecisionTree zig/js fit speedup: 1.958x
DecisionTree zig/js predict speedup: 0.550x
RandomForest zig/js fit speedup: 2.829x
RandomForest zig/js predict speedup: 2.392x
Snapshot generated at: 2026-02-25T19:23:37.811Z
