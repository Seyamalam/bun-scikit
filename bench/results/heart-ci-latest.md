## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2751 | 0.0514 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.6437 | 0.0824 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 2.340x
Bun predict speedup vs scikit-learn: 1.603x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.540e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 2.0931 | 0.0544 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.9931 | 0.1261 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.952x
Bun predict speedup vs scikit-learn: 2.319x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 1.4751 | 0.0214 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.8485 | 0.1343 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 43.8214 | 1.6327 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 109.5692 | 6.8000 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 1.253x
DecisionTree predict speedup vs scikit-learn: 6.275x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2

RandomForest fit speedup vs scikit-learn: 2.500x
RandomForest predict speedup vs scikit-learn: 4.165x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3

Snapshot generated at: 2026-02-23T09:54:58.330Z
