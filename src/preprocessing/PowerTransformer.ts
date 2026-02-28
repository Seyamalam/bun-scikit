import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";

export type PowerTransformerMethod = "yeo-johnson" | "box-cox";

export interface PowerTransformerOptions {
  method?: PowerTransformerMethod;
  standardize?: boolean;
}

function boxCox(value: number, lambda: number): number {
  if (lambda === 0) {
    return Math.log(value);
  }
  return (Math.pow(value, lambda) - 1) / lambda;
}

function boxCoxInverse(value: number, lambda: number): number {
  if (lambda === 0) {
    return Math.exp(value);
  }
  return Math.pow(lambda * value + 1, 1 / lambda);
}

function yeoJohnson(value: number, lambda: number): number {
  if (value >= 0) {
    if (lambda === 0) {
      return Math.log(value + 1);
    }
    return (Math.pow(value + 1, lambda) - 1) / lambda;
  }
  if (lambda === 2) {
    return -Math.log(1 - value);
  }
  return -((Math.pow(1 - value, 2 - lambda) - 1) / (2 - lambda));
}

function yeoJohnsonInverse(value: number, lambda: number): number {
  if (value >= 0) {
    if (lambda === 0) {
      return Math.exp(value) - 1;
    }
    return Math.pow(lambda * value + 1, 1 / lambda) - 1;
  }
  if (lambda === 2) {
    return 1 - Math.exp(-value);
  }
  return 1 - Math.pow(1 - (2 - lambda) * value, 1 / (2 - lambda));
}

export class PowerTransformer {
  lambdas_: Vector | null = null;
  nFeaturesIn_: number | null = null;

  private method: PowerTransformerMethod;
  private standardize: boolean;
  private mean_: Vector | null = null;
  private scale_: Vector | null = null;
  private fitted = false;

  constructor(options: PowerTransformerOptions = {}) {
    this.method = options.method ?? "yeo-johnson";
    this.standardize = options.standardize ?? true;
    if (!(this.method === "yeo-johnson" || this.method === "box-cox")) {
      throw new Error(`method must be 'yeo-johnson' or 'box-cox'. Got ${this.method}.`);
    }
  }

  fit(X: Matrix): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (this.method === "box-cox") {
      for (let i = 0; i < X.length; i += 1) {
        for (let j = 0; j < X[i].length; j += 1) {
          if (X[i][j] <= 0) {
            throw new Error("box-cox transformation requires strictly positive inputs.");
          }
        }
      }
    }

    const nFeatures = X[0].length;
    const lambdas = new Array<number>(nFeatures).fill(0);
    // Lightweight default estimator: lambda=0 per feature (log-like transform).
    this.lambdas_ = lambdas;
    this.nFeaturesIn_ = nFeatures;

    const transformed = this.transformWithLambdas(X, lambdas);
    if (this.standardize) {
      const mean = new Array<number>(nFeatures).fill(0);
      const scale = new Array<number>(nFeatures).fill(0);
      for (let i = 0; i < transformed.length; i += 1) {
        for (let j = 0; j < nFeatures; j += 1) {
          mean[j] += transformed[i][j];
        }
      }
      for (let j = 0; j < nFeatures; j += 1) {
        mean[j] /= transformed.length;
      }
      for (let i = 0; i < transformed.length; i += 1) {
        for (let j = 0; j < nFeatures; j += 1) {
          const d = transformed[i][j] - mean[j];
          scale[j] += d * d;
        }
      }
      for (let j = 0; j < nFeatures; j += 1) {
        scale[j] = Math.sqrt(scale[j] / transformed.length);
        if (scale[j] <= 1e-12) {
          scale[j] = 1;
        }
      }
      this.mean_ = mean;
      this.scale_ = scale;
    } else {
      this.mean_ = null;
      this.scale_ = null;
    }

    this.fitted = true;
    return this;
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    const out = this.transformWithLambdas(X, this.lambdas_!);
    if (this.standardize) {
      for (let i = 0; i < out.length; i += 1) {
        for (let j = 0; j < out[i].length; j += 1) {
          out[i][j] = (out[i][j] - this.mean_![j]) / this.scale_![j];
        }
      }
    }
    return out;
  }

  inverseTransform(X: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }

    const prepared: Matrix = X.map((row) => row.slice());
    if (this.standardize) {
      for (let i = 0; i < prepared.length; i += 1) {
        for (let j = 0; j < prepared[i].length; j += 1) {
          prepared[i][j] = prepared[i][j] * this.scale_![j] + this.mean_![j];
        }
      }
    }

    const out: Matrix = new Array(prepared.length);
    for (let i = 0; i < prepared.length; i += 1) {
      const row = new Array<number>(this.nFeaturesIn_!);
      for (let j = 0; j < this.nFeaturesIn_!; j += 1) {
        const lambda = this.lambdas_![j];
        if (this.method === "box-cox") {
          row[j] = boxCoxInverse(prepared[i][j], lambda);
        } else {
          row[j] = yeoJohnsonInverse(prepared[i][j], lambda);
        }
      }
      out[i] = row;
    }
    return out;
  }

  fitTransform(X: Matrix): Matrix {
    return this.fit(X).transform(X);
  }

  private transformWithLambdas(X: Matrix, lambdas: Vector): Matrix {
    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const row = new Array<number>(X[0].length);
      for (let j = 0; j < X[0].length; j += 1) {
        const lambda = lambdas[j];
        if (this.method === "box-cox") {
          row[j] = boxCox(X[i][j], lambda);
        } else {
          row[j] = yeoJohnson(X[i][j], lambda);
        }
      }
      out[i] = row;
    }
    return out;
  }

  private assertFitted(): void {
    if (!this.fitted || !this.lambdas_ || this.nFeaturesIn_ === null) {
      throw new Error("PowerTransformer has not been fitted.");
    }
  }
}
