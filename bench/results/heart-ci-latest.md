## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.6821 | 0.0137 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3391 | 0.0410 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 0.497x
Bun predict speedup vs scikit-learn: 2.989x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd) | 15.5949 | 0.0709 | 0.863415 | 0.875000 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 2.4016 | 0.1820 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.154x
Bun predict speedup vs scikit-learn: 2.566x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): -1.110e-16

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 8.9542 | 0.0354 | 0.931707 | 0.935185 |
| DecisionTreeClassifier | python-scikit-learn | 2.3275 | 0.1965 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 170.0284 | 1.6933 | 0.995122 | 0.995261 |
| RandomForestClassifier | python-scikit-learn | 70.3500 | 2.2056 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 0.260x
DecisionTree predict speedup vs scikit-learn: 5.551x
DecisionTree accuracy delta (bun - sklearn): 0.000e+0
DecisionTree f1 delta (bun - sklearn): 1.223e-3

RandomForest fit speedup vs scikit-learn: 0.414x
RandomForest predict speedup vs scikit-learn: 1.303x
RandomForest accuracy delta (bun - sklearn): 0.000e+0
RandomForest f1 delta (bun - sklearn): 1.110e-16

Snapshot generated at: 2026-02-22T11:30:24.711Z
