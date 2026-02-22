## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 1.4633 | 0.0431 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 1.1705 | 0.1497 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 0.800x
Bun predict speedup vs scikit-learn: 3.475x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.540e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd) | 21.8799 | 0.0408 | 0.863415 | 0.875000 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.9950 | 0.1786 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.137x
Bun predict speedup vs scikit-learn: 4.382x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): -1.110e-16

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 15.5636 | 0.0431 | 0.931707 | 0.935185 |
| DecisionTreeClassifier | python-scikit-learn | 2.4952 | 0.1854 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 222.1371 | 2.0011 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 112.4116 | 7.0331 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 0.160x
DecisionTree predict speedup vs scikit-learn: 4.299x
DecisionTree accuracy delta (bun - sklearn): 0.000e+0
DecisionTree f1 delta (bun - sklearn): 1.223e-3

RandomForest fit speedup vs scikit-learn: 0.506x
RandomForest predict speedup vs scikit-learn: 3.515x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16

Snapshot generated at: 2026-02-22T11:23:11.219Z
