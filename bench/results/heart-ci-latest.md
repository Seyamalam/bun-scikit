| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |
|---|---|---:|---:|---:|---:|
| bun-scikit | StandardScaler + LinearRegression(normal) | 1.0020 | 0.0154 | 0.117545 | 0.529539 |
| python-scikit-learn | StandardScaler + LinearRegression | 2.6626 | 0.0778 | 0.117545 | 0.529539 |

Bun fit speedup vs scikit-learn: 2.657x
Bun predict speedup vs scikit-learn: 5.052x
MSE delta (bun - sklearn): 6.360e-14
R2 delta (bun - sklearn): -2.539e-13

Snapshot generated at: 2026-02-22T10:31:55.653Z
