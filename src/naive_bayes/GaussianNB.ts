import type { ClassificationModel, Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { accuracyScore } from "../metrics/classification";

const DEFAULT_VAR_SMOOTHING = 1e-9;

export interface GaussianNBOptions {
  varSmoothing?: number;
}

export class GaussianNB implements ClassificationModel {
  classes_: Vector = [0, 1];
  classPrior_: Vector | null = null;
  theta_: Matrix | null = null;
  var_: Matrix | null = null;

  private readonly varSmoothing: number;
  private fittedFeatureCount = 0;
  private epsilon = 0;

  constructor(options: GaussianNBOptions = {}) {
    this.varSmoothing = options.varSmoothing ?? DEFAULT_VAR_SMOOTHING;
    if (!Number.isFinite(this.varSmoothing) || this.varSmoothing < 0) {
      throw new Error(`varSmoothing must be finite and >= 0. Got ${this.varSmoothing}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);

    const sampleCount = X.length;
    const featureCount = X[0].length;
    this.fittedFeatureCount = featureCount;
    this.classes_ = [0, 1];

    let maxVariance = 0;
    for (let j = 0; j < featureCount; j += 1) {
      let sum = 0;
      let sumSquares = 0;
      for (let i = 0; i < sampleCount; i += 1) {
        const value = X[i][j];
        sum += value;
        sumSquares += value * value;
      }
      const mean = sum / sampleCount;
      const variance = sumSquares / sampleCount - mean * mean;
      if (variance > maxVariance) {
        maxVariance = variance;
      }
    }
    this.epsilon = this.varSmoothing * maxVariance;

    const priors = new Array<number>(2).fill(0);
    const means = Array.from({ length: 2 }, () => new Array<number>(featureCount).fill(0));
    const variances = Array.from({ length: 2 }, () => new Array<number>(featureCount).fill(0));
    const counts = new Array<number>(2).fill(0);

    for (let i = 0; i < sampleCount; i += 1) {
      const label = y[i];
      counts[label] += 1;
      for (let j = 0; j < featureCount; j += 1) {
        means[label][j] += X[i][j];
      }
    }

    for (let cls = 0; cls < 2; cls += 1) {
      if (counts[cls] === 0) {
        throw new Error(`GaussianNB requires both classes to be present. Missing class ${cls}.`);
      }
      priors[cls] = counts[cls] / sampleCount;
      for (let j = 0; j < featureCount; j += 1) {
        means[cls][j] /= counts[cls];
      }
    }

    for (let i = 0; i < sampleCount; i += 1) {
      const label = y[i];
      for (let j = 0; j < featureCount; j += 1) {
        const diff = X[i][j] - means[label][j];
        variances[label][j] += diff * diff;
      }
    }

    for (let cls = 0; cls < 2; cls += 1) {
      for (let j = 0; j < featureCount; j += 1) {
        variances[cls][j] = variances[cls][j] / counts[cls] + this.epsilon;
      }
    }

    this.classPrior_ = priors;
    this.theta_ = means;
    this.var_ = variances;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    if (!this.classPrior_ || !this.theta_ || !this.var_ || this.fittedFeatureCount === 0) {
      throw new Error("GaussianNB has not been fitted.");
    }

    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.fittedFeatureCount) {
      throw new Error(
        `Feature size mismatch. Expected ${this.fittedFeatureCount}, got ${X[0].length}.`,
      );
    }

    const outputs = new Array<number[]>(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = X[i];
      const logProb = new Array<number>(2).fill(0);
      for (let cls = 0; cls < 2; cls += 1) {
        let sum = Math.log(this.classPrior_[cls]);
        for (let j = 0; j < this.fittedFeatureCount; j += 1) {
          const variance = this.var_[cls][j];
          const mean = this.theta_[cls][j];
          const diff = row[j] - mean;
          sum += -0.5 * Math.log(2 * Math.PI * variance) - (diff * diff) / (2 * variance);
        }
        logProb[cls] = sum;
      }

      const maxLog = Math.max(logProb[0], logProb[1]);
      const exp0 = Math.exp(logProb[0] - maxLog);
      const exp1 = Math.exp(logProb[1] - maxLog);
      const denom = exp0 + exp1;
      outputs[i] = [exp0 / denom, exp1 / denom];
    }

    return outputs;
  }

  predict(X: Matrix): Vector {
    const probabilities = this.predictProba(X);
    return probabilities.map((pair) => (pair[1] >= 0.5 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }
}
