import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import { argmax, normalizeProbabilitiesInPlace, uniqueSortedLabels } from "../utils/classification";
import { kernelValue, resolveKernelConfig, type GammaOption, type KernelName } from "./kernelUtils";

export interface SVCOptions {
  C?: number;
  kernel?: KernelName;
  gamma?: GammaOption;
  degree?: number;
  coef0?: number;
  maxIter?: number;
  tolerance?: number;
  learningRate?: number;
}

interface BinaryModel {
  alpha: Vector;
  ySigned: Vector;
  bias: number;
}

export class SVC implements ClassificationModel {
  classes_: Vector = [];
  supportVectors_: Matrix = [];
  support_: number[] = [];

  private C: number;
  private kernel: KernelName;
  private gamma: GammaOption;
  private degree: number;
  private coef0: number;
  private maxIter: number;
  private tolerance: number;
  private learningRate: number;

  private XTrain: Matrix | null = null;
  private classModels: BinaryModel[] = [];
  private isFitted = false;

  constructor(options: SVCOptions = {}) {
    this.C = options.C ?? 1.0;
    this.kernel = options.kernel ?? "rbf";
    this.gamma = options.gamma ?? "scale";
    this.degree = options.degree ?? 3;
    this.coef0 = options.coef0 ?? 0;
    this.maxIter = options.maxIter ?? 500;
    this.tolerance = options.tolerance ?? 1e-6;
    this.learningRate = options.learningRate ?? 0.05;
    this.validateOptions();
  }

  getParams(): SVCOptions {
    return {
      C: this.C,
      kernel: this.kernel,
      gamma: this.gamma,
      degree: this.degree,
      coef0: this.coef0,
      maxIter: this.maxIter,
      tolerance: this.tolerance,
      learningRate: this.learningRate,
    };
  }

  setParams(params: Partial<SVCOptions>): this {
    if (params.C !== undefined) this.C = params.C;
    if (params.kernel !== undefined) this.kernel = params.kernel;
    if (params.gamma !== undefined) this.gamma = params.gamma;
    if (params.degree !== undefined) this.degree = params.degree;
    if (params.coef0 !== undefined) this.coef0 = params.coef0;
    if (params.maxIter !== undefined) this.maxIter = params.maxIter;
    if (params.tolerance !== undefined) this.tolerance = params.tolerance;
    if (params.learningRate !== undefined) this.learningRate = params.learningRate;
    this.validateOptions();
    return this;
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    if (this.classes_.length < 2) {
      throw new Error("SVC requires at least two classes.");
    }

    const kernelConfig = resolveKernelConfig(X, this.getParams());
    this.XTrain = X.map((row) => row.slice());
    this.classModels = new Array<BinaryModel>(this.classes_.length);
    for (let classIndex = 0; classIndex < this.classes_.length; classIndex += 1) {
      const label = this.classes_[classIndex];
      const ySigned = y.map((value) => (value === label ? 1 : -1));
      this.classModels[classIndex] = this.fitBinary(this.XTrain, ySigned, kernelConfig);
    }

    const supportMask = new Uint8Array(X.length);
    for (let c = 0; c < this.classModels.length; c += 1) {
      const alpha = this.classModels[c].alpha;
      for (let i = 0; i < alpha.length; i += 1) {
        if (alpha[i] > 1e-12) {
          supportMask[i] = 1;
        }
      }
    }
    this.support_ = [];
    this.supportVectors_ = [];
    for (let i = 0; i < supportMask.length; i += 1) {
      if (supportMask[i] === 1) {
        this.support_.push(i);
        this.supportVectors_.push(this.XTrain[i]);
      }
    }

    this.isFitted = true;
    return this;
  }

  decisionFunction(X: Matrix): Vector | Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.XTrain![0].length) {
      throw new Error(`Feature size mismatch. Expected ${this.XTrain![0].length}, got ${X[0].length}.`);
    }

    const kernelConfig = resolveKernelConfig(this.XTrain!, this.getParams());
    const out: Matrix = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      const rowScores = new Array<number>(this.classModels.length);
      for (let classIndex = 0; classIndex < this.classModels.length; classIndex += 1) {
        const model = this.classModels[classIndex];
        let score = model.bias;
        for (let sampleIndex = 0; sampleIndex < this.XTrain!.length; sampleIndex += 1) {
          const alpha = model.alpha[sampleIndex];
          if (alpha <= 0) {
            continue;
          }
          score +=
            alpha *
            model.ySigned[sampleIndex] *
            kernelValue(this.XTrain![sampleIndex], X[i], kernelConfig);
        }
        rowScores[classIndex] = score;
      }
      out[i] = rowScores;
    }

    if (this.classes_.length === 2) {
      return out.map((row) => row[1] - row[0]);
    }
    return out;
  }

  predictProba(X: Matrix): Matrix {
    const decision = this.decisionFunction(X);
    if (!Array.isArray(decision[0])) {
      const binary = decision as Vector;
      return binary.map((score) => {
        const p1 = 1 / (1 + Math.exp(-score));
        return [1 - p1, p1];
      });
    }

    return (decision as Matrix).map((row) => {
      const logits = row.map((value) => Math.exp(value));
      normalizeProbabilitiesInPlace(logits);
      return logits;
    });
  }

  predict(X: Matrix): Vector {
    const decision = this.decisionFunction(X);
    if (!Array.isArray(decision[0])) {
      const binary = decision as Vector;
      return binary.map((score) => (score >= 0 ? this.classes_[1] : this.classes_[0]));
    }
    return (decision as Matrix).map((row) => this.classes_[argmax(row)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted || this.XTrain === null || this.classModels.length === 0) {
      throw new Error("SVC has not been fitted.");
    }
  }

  private fitBinary(X: Matrix, ySigned: Vector, kernelConfig: ReturnType<typeof resolveKernelConfig>): BinaryModel {
    const alpha = new Array<number>(X.length).fill(0);
    let bias = 0;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      let maxDelta = 0;
      for (let i = 0; i < X.length; i += 1) {
        let score = bias;
        for (let j = 0; j < X.length; j += 1) {
          const a = alpha[j];
          if (a <= 0) {
            continue;
          }
          score += a * ySigned[j] * kernelValue(X[j], X[i], kernelConfig);
        }

        const margin = ySigned[i] * score;
        if (margin < 1) {
          const oldAlpha = alpha[i];
          alpha[i] = Math.min(this.C, alpha[i] + this.learningRate);
          const delta = Math.abs(alpha[i] - oldAlpha);
          if (delta > maxDelta) {
            maxDelta = delta;
          }
          bias += this.learningRate * ySigned[i];
        }
      }
      if (maxDelta < this.tolerance) {
        break;
      }
    }

    return { alpha, ySigned: ySigned.slice(), bias };
  }

  private validateOptions(): void {
    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new Error(`C must be finite and > 0. Got ${this.C}.`);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new Error(`learningRate must be finite and > 0. Got ${this.learningRate}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance <= 0) {
      throw new Error(`tolerance must be finite and > 0. Got ${this.tolerance}.`);
    }
  }
}

