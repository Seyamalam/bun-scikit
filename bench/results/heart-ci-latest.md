## Regression (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 1.1130 | 0.0253 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.3210 | 0.0380 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 0.288x
Bun predict speedup vs scikit-learn: 1.502x
MSE delta (bun - sklearn): 6.360e-14
R2 delta (bun - sklearn): -2.539e-13

## Classification (Heart Dataset)

| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LogisticRegression(gd) | 15.3544 | 0.0539 | 0.863415 | 0.875000 |
| python-scikit-learn | StandardScaler + LogisticRegression(lbfgs) | 1.2240 | 0.0842 | 0.863415 | 0.875000 |

Bun fit speedup vs scikit-learn: 0.080x
Bun predict speedup vs scikit-learn: 1.561x
Accuracy delta (bun - sklearn): 0.000e+0
F1 delta (bun - sklearn): -1.110e-16

Snapshot generated at: 2026-02-22T11:10:54.561Z
