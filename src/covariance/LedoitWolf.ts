import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import {
  covarianceMatrix,
  featureMeans,
  mahalanobisDistanceSquared,
  matrixDeterminant,
  regularizedPrecision,
} from "./shared";

export class LedoitWolf {
  location_: Vector | null = null;
  covariance_: Matrix | null = null;
  precision_: Matrix | null = null;
  shrinkage_ = 0;
  nFeaturesIn_: number | null = null;

  private fitted = false;

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    const location = featureMeans(X);
    const empirical = covarianceMatrix(X, location);
    const p = empirical.length;
    const n = X.length;
    let trace = 0;
    for (let i = 0; i < p; i += 1) {
      trace += empirical[i][i];
    }
    const mu = trace / p;
    const shrinkage = Math.min(1, p / Math.max(1, n));

    const covariance: Matrix = empirical.map((row, i) =>
      row.map((value, j) =>
        (1 - shrinkage) * value + (i === j ? shrinkage * mu : 0),
      ),
    );
    const precision = regularizedPrecision(covariance);

    this.location_ = location;
    this.covariance_ = covariance;
    this.precision_ = precision;
    this.shrinkage_ = shrinkage;
    this.nFeaturesIn_ = X[0].length;
    this.fitted = true;
    return this;
  }

  scoreSamples(X: Matrix): Vector {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    const det = Math.max(1e-12, matrixDeterminant(this.covariance_!));
    const logDet = Math.log(det);
    const d = this.nFeaturesIn_!;
    return X.map((row) => {
      const mah = mahalanobisDistanceSquared(row, this.location_!, this.precision_!);
      return -0.5 * (d * Math.log(2 * Math.PI) + logDet + mah);
    });
  }

  score(X: Matrix): number {
    const scores = this.scoreSamples(X);
    let sum = 0;
    for (let i = 0; i < scores.length; i += 1) {
      sum += scores[i];
    }
    return sum / scores.length;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.location_ || !this.covariance_ || !this.precision_ || this.nFeaturesIn_ === null) {
      throw new Error("LedoitWolf has not been fitted.");
    }
  }
}
