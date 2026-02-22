## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.1624 | 0.0164 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3054 | 0.0377 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 1.881x
Bun predict speedup vs scikit-learn: 2.302x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 0.5255 | 0.0291 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.1601 | 0.0798 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 2.207x
Bun predict speedup vs scikit-learn: 2.747x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 0.8436 | 0.0209 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 2.6148 | 0.2761 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 29.6543 | 1.7566 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 203.3201 | 6.9707 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 3.100x
DecisionTree predict speedup vs scikit-learn: 13.213x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2

RandomForest fit speedup vs scikit-learn: 6.856x
RandomForest predict speedup vs scikit-learn: 3.968x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3

Snapshot generated at: 2026-02-22T18:24:11.645Z
