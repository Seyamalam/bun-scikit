## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2073 | 0.0416 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3220 | 0.0377 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 1.553x
Bun predict speedup vs scikit-learn: 0.905x
MSE delta (bun - sklearn): 6.360e-14
R2 delta (bun - sklearn): -2.539e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.3293 | 0.0376 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.1626 | 0.0761 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.875x
Bun predict speedup vs scikit-learn: 2.028x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 1.1473 | 0.0196 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.4081 | 0.0923 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 36.0933 | 1.5500 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 70.8191 | 2.1874 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 1.227x
DecisionTree predict speedup vs scikit-learn: 4.707x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2

RandomForest fit speedup vs scikit-learn: 1.962x
RandomForest predict speedup vs scikit-learn: 1.411x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3

Snapshot generated at: 2026-02-22T17:45:48.958Z
