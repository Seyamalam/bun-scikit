## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2690 | 0.0147 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3004 | 0.0357 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 1.117x
Bun predict speedup vs scikit-learn: 2.425x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 1.0356 | 0.0195 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.0588 | 0.0702 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 1.023x
Bun predict speedup vs scikit-learn: 3.588x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 1.0874 | 0.0176 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.2841 | 0.0850 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 32.1913 | 1.4896 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 65.2060 | 2.0660 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 1.181x
DecisionTree predict speedup vs scikit-learn: 4.832x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2

RandomForest fit speedup vs scikit-learn: 2.026x
RandomForest predict speedup vs scikit-learn: 1.387x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3

Snapshot generated at: 2026-02-22T17:19:58.072Z
