## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.7298 | 0.0243 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 1.1323 | 0.1461 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 1.552x
Bun predict speedup vs scikit-learn: 6.005x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd) | 16.9612 | 0.0394 | 0.863415 | 0.875000 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 3.5231 | 0.1938 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.208x
Bun predict speedup vs scikit-learn: 4.915x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): -1.110e-16

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 13.8622 | 0.0300 | 0.931707 | 0.935185 |
| DecisionTreeClassifier | python-scikit-learn | 2.4583 | 0.1800 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 225.1871 | 1.7256 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 110.1258 | 6.8506 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 0.177x
DecisionTree predict speedup vs scikit-learn: 6.005x
DecisionTree accuracy delta (bun - sklearn): 0.000e+0
DecisionTree f1 delta (bun - sklearn): 1.223e-3

RandomForest fit speedup vs scikit-learn: 0.489x
RandomForest predict speedup vs scikit-learn: 3.970x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16

Snapshot generated at: 2026-02-22T12:17:31.022Z
