import type { Matrix, RegressionModel, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateRegressionInputs,
} from "../utils/validation";
import { inverseMatrix, multiplyMatrixVector, mean } from "../utils/linalg";
import { kernelValue, resolveKernelConfig, type GammaOption, type KernelName } from "./kernelUtils";

export interface SVROptions {
  C?: number;
  epsilon?: number;
  kernel?: KernelName;
  gamma?: GammaOption;
  degree?: number;
  coef0?: number;
}

export class SVR implements RegressionModel {
  dualCoef_: Vector = [];
  intercept_ = 0;
  support_: number[] = [];
  supportVectors_: Matrix = [];

  private C: number;
  private epsilon: number;
  private kernel: KernelName;
  private gamma: GammaOption;
  private degree: number;
  private coef0: number;

  private XTrain: Matrix | null = null;
  private isFitted = false;

  constructor(options: SVROptions = {}) {
    this.C = options.C ?? 1.0;
    this.epsilon = options.epsilon ?? 0.1;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? "scale";
    this.degree = options.degree ?? 3;
    this.coef0 = options.coef0 ?? 0;
    this.validateOptions();
  }

  getParams(): SVROptions {
    return {
      C: this.C,
      epsilon: this.epsilon,
      kernel: this.kernel,
      gamma: this.gamma,
      degree: this.degree,
      coef0: this.coef0,
    };
  }

  setParams(params: Partial<SVROptions>): this {
    if (params.C !== undefined) this.C = params.C;
    if (params.epsilon !== undefined) this.epsilon = params.epsilon;
    if (params.kernel !== undefined) this.kernel = params.kernel;
    if (params.gamma !== undefined) this.gamma = params.gamma;
    if (params.degree !== undefined) this.degree = params.degree;
    if (params.coef0 !== undefined) this.coef0 = params.coef0;
    this.validateOptions();
    return this;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateRegressionInputs(X, y);
    const kernelConfig = resolveKernelConfig(X, this.getParams());
    const n = X.length;
    const gram: Matrix = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i += 1) {
      for (let j = i; j < n; j += 1) {
        const value = kernelValue(X[i], X[j], kernelConfig);
        gram[i][j] = value;
        gram[j][i] = value;
      }
    }

    const lambda = 1 / this.C;
    const system: Matrix = gram.map((row, i) => {
      const next = row.slice();
      next[i] += lambda;
      return next;
    });
    const inv = inverseMatrix(system);
    const alpha = multiplyMatrixVector(inv, y);
    for (let i = 0; i < alpha.length; i += 1) {
      if (alpha[i] > this.C) alpha[i] = this.C;
      if (alpha[i] < -this.C) alpha[i] = -this.C;
      if (Math.abs(alpha[i]) < this.epsilon * 0.1) alpha[i] = 0;
    }

    const residual = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
      let pred = 0;
      for (let j = 0; j < n; j += 1) {
        pred += alpha[j] * gram[j][i];
      }
      residual[i] = y[i] - pred;
    }

    this.intercept_ = mean(residual);
    this.dualCoef_ = alpha;
    this.XTrain = X.map((row) => row.slice());
    this.support_ = [];
    this.supportVectors_ = [];
    for (let i = 0; i < alpha.length; i += 1) {
      if (Math.abs(alpha[i]) > 1e-12) {
        this.support_.push(i);
        this.supportVectors_.push(this.XTrain[i]);
      }
    }

    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.XTrain![0].length) {
      throw new Error(`Feature size mismatch. Expected ${this.XTrain![0].length}, got ${X[0].length}.`);
    }

    const kernelConfig = resolveKernelConfig(this.XTrain!, this.getParams());
    const out = new Array<number>(X.length);
    for (let i = 0; i < X.length; i += 1) {
      let score = this.intercept_;
      for (let j = 0; j < this.XTrain!.length; j += 1) {
        const alpha = this.dualCoef_[j];
        if (alpha === 0) {
          continue;
        }
        score += alpha * kernelValue(this.XTrain![j], X[i], kernelConfig);
      }
      out[i] = score;
    }
    return out;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.XTrain === null) {
      throw new Error("SVR has not been fitted.");
    }
  }

  private validateOptions(): void {
    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new Error(`C must be finite and > 0. Got ${this.C}.`);
    }
    if (!Number.isFinite(this.epsilon) || this.epsilon < 0) {
      throw new Error(`epsilon must be finite and >= 0. Got ${this.epsilon}.`);
    }
  }
}

