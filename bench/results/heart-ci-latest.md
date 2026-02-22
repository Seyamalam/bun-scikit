## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.9420 | 0.0194 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.7581 | 0.0906 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 0.805x
Bun predict speedup vs scikit-learn: 4.658x
MSE delta (bun - sklearn): 6.360e-14
R2 delta (bun - sklearn): -2.539e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd) | 17.1480 | 0.0596 | 0.863415 | 0.875000 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.2842 | 0.1486 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.133x
Bun predict speedup vs scikit-learn: 2.494x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): -1.110e-16

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 8.4473 | 0.0339 | 0.931707 | 0.935185 |
| DecisionTreeClassifier | python-scikit-learn | 2.3470 | 0.2268 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 170.6729 | 1.7940 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 73.0010 | 2.2587 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 0.278x
DecisionTree predict speedup vs scikit-learn: 6.702x
DecisionTree accuracy delta (bun - sklearn): 0.000e+0
DecisionTree f1 delta (bun - sklearn): 1.223e-3

RandomForest fit speedup vs scikit-learn: 0.428x
RandomForest predict speedup vs scikit-learn: 1.259x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16

Snapshot generated at: 2026-02-22T11:21:30.327Z
