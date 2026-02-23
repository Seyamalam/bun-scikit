## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.2103 | 0.0216 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3201 | 0.0365 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 1.522x
Bun predict speedup vs scikit-learn: 1.684x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 0.4868 | 0.0282 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.1246 | 0.0724 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 2.310x
Bun predict speedup vs scikit-learn: 2.574x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 0.8062 | 0.0190 | 0.946341 | 0.948837 |
| DecisionTreeClassifier | python-scikit-learn | 1.4781 | 0.0999 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 27.6225 | 1.8535 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 172.9585 | 6.4850 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 1.833x
DecisionTree predict speedup vs scikit-learn: 5.244x
DecisionTree accuracy delta (bun - sklearn): 1.463e-2
DecisionTree f1 delta (bun - sklearn): 1.487e-2

RandomForest fit speedup vs scikit-learn: 6.262x
RandomForest predict speedup vs scikit-learn: 3.499x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3

Snapshot generated at: 2026-02-23T14:55:51.251Z
