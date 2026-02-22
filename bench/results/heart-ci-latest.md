## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.7707 | 0.0265 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.9458 | 0.1200 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 1.227x
Bun predict speedup vs scikit-learn: 4.529x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd) | 17.1836 | 0.0494 | 0.863415 | 0.875000 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 3.3536 | 0.1967 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.195x
Bun predict speedup vs scikit-learn: 3.980x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): -1.110e-16

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 11.0958 | 0.0240 | 0.931707 | 0.935185 |
| DecisionTreeClassifier | python-scikit-learn | 2.3914 | 0.1937 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 232.1108 | 1.7963 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 111.3027 | 7.0863 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 0.216x
DecisionTree predict speedup vs scikit-learn: 8.061x
DecisionTree accuracy delta (bun - sklearn): 0.000e+0
DecisionTree f1 delta (bun - sklearn): 1.223e-3

RandomForest fit speedup vs scikit-learn: 0.480x
RandomForest predict speedup vs scikit-learn: 3.945x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16

Snapshot generated at: 2026-02-22T11:32:04.342Z
