| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 0.7289 | 0.0167 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 0.7879 | 0.0927 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 1.081x
Bun predict speedup vs scikit-learn: 5.565x
MSE delta (bun - sklearn): 6.363e-14
R2 delta (bun - sklearn): -2.540e-13

Snapshot generated at: 2026-02-22T10:34:04.665Z
