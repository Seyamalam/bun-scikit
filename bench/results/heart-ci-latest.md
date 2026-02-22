## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.1798 | 0.0188 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.4432 | 0.0543 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 2.465x
Bun predict speedup vs scikit-learn: 2.893x
MSE delta (bun - sklearn): 6.362e-14
R2 delta (bun - sklearn): -2.539e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd,zig) | 0.5098 | 0.0298 | 0.863415 | 0.876106 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.2075 | 0.0794 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 2.369x
Bun predict speedup vs scikit-learn: 2.664x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): 1.106e-3

## Tree Classification (Heart Dataset)

| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| DecisionTreeClassifier(maxDepth=8) | bun-scikit | 0.8007 | 0.0212 | 0.921951 | 0.923077 |
| DecisionTreeClassifier | python-scikit-learn | 1.4200 | 0.0953 | 0.931707 | 0.933962 |
| RandomForestClassifier(nEstimators=80,maxDepth=8) | bun-scikit | 27.8613 | 1.7048 | 0.990244 | 0.990566 |
| RandomForestClassifier | python-scikit-learn | 181.9663 | 6.8009 | 0.995122 | 0.995261 |

DecisionTree fit speedup vs scikit-learn: 1.773x
DecisionTree predict speedup vs scikit-learn: 4.508x
DecisionTree accuracy delta (bun - sklearn): -9.756e-3
DecisionTree f1 delta (bun - sklearn): -1.089e-2

RandomForest fit speedup vs scikit-learn: 6.531x
RandomForest predict speedup vs scikit-learn: 3.989x
RandomForest accuracy delta (bun - sklearn): -4.878e-3
RandomForest f1 delta (bun - sklearn): -4.695e-3

Snapshot generated at: 2026-02-22T18:18:30.681Z
