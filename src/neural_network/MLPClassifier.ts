import type { ClassificationModel, Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { uniqueSortedLabels } from "../utils/classification";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";
import {
  activationDerivativeFromZ,
  applyActivationInPlace,
  argmaxRow,
  gaussianRandom,
  initAdamState,
  initializeNetwork,
  matMulAddBias,
  mulberry32,
  parseHiddenLayerSizes,
  softmaxInPlace,
  type AdamState,
  type MLPActivation,
  type MLPCommonOptions,
  type MLPSolver,
} from "./shared";

export interface MLPClassifierOptions extends MLPCommonOptions {
  beta1?: number;
  beta2?: number;
  epsilon?: number;
}

export class MLPClassifier implements ClassificationModel {
  classes_: Vector = [0, 1];
  coefs_: Matrix[] = [];
  intercepts_: Vector[] = [];
  nIter_ = 0;
  loss_: number | null = null;
  nFeaturesIn_: number | null = null;

  private hiddenLayerSizes: number[];
  private activation: MLPActivation;
  private solver: MLPSolver;
  private alpha: number;
  private batchSize: number;
  private learningRateInit: number;
  private maxIter: number;
  private tolerance: number;
  private randomState?: number;
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private fitted = false;

  constructor(options: MLPClassifierOptions = {}) {
    this.hiddenLayerSizes = parseHiddenLayerSizes(options.hiddenLayerSizes);
    this.activation = options.activation ?? "relu";
    this.solver = options.solver ?? "adam";
    this.alpha = options.alpha ?? 0.0001;
    this.batchSize = options.batchSize ?? 200;
    this.learningRateInit = options.learningRateInit ?? 0.001;
    this.maxIter = options.maxIter ?? 200;
    this.tolerance = options.tolerance ?? 1e-4;
    this.randomState = options.randomState;
    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;
    this.epsilon = options.epsilon ?? 1e-8;
    this.validateOptions();
  }

  fit(X: Matrix, y: Vector, sampleWeight?: Vector): this {
    validateClassificationInputs(X, y);
    this.classes_ = uniqueSortedLabels(y);
    if (this.classes_.length < 2) {
      throw new Error("MLPClassifier requires at least two classes.");
    }

    const classToIndex = new Map<number, number>();
    for (let i = 0; i < this.classes_.length; i += 1) {
      classToIndex.set(this.classes_[i], i);
    }

    const nSamples = X.length;
    const nFeatures = X[0].length;
    const nOutputs = this.classes_.length;
    const yOneHot: Matrix = new Array(nSamples);
    for (let i = 0; i < nSamples; i += 1) {
      const row = new Array<number>(nOutputs).fill(0);
      const classIndex = classToIndex.get(y[i]);
      if (classIndex === undefined) {
        throw new Error(`Unknown class label '${y[i]}' at index ${i}.`);
      }
      row[classIndex] = 1;
      yOneHot[i] = row;
    }

    const random =
      this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    const layerSizes = [nFeatures, ...this.hiddenLayerSizes, nOutputs];
    const params = initializeNetwork(layerSizes, random);
    this.coefs_ = params.coefs;
    this.intercepts_ = params.intercepts;
    const adam = this.solver === "adam" ? initAdamState(this.coefs_, this.intercepts_) : null;

    let prevLoss = Number.POSITIVE_INFINITY;
    let nIter = 0;
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const forward = this.forward(X);
      const probabilities = forward.activations[forward.activations.length - 1];
      const loss = this.classificationLoss(probabilities, yOneHot);
      this.loss_ = loss;
      nIter = iter + 1;

      let delta: Matrix = new Array(nSamples);
      for (let i = 0; i < nSamples; i += 1) {
        const row = new Array<number>(nOutputs);
        for (let j = 0; j < nOutputs; j += 1) {
          row[j] = probabilities[i][j] - yOneHot[i][j];
        }
        delta[i] = row;
      }

      for (let layer = this.coefs_.length - 1; layer >= 0; layer -= 1) {
        const activationsPrev = forward.activations[layer];
        const fanIn = this.coefs_[layer].length;
        const fanOut = this.coefs_[layer][0].length;
        const gradW: Matrix = Array.from({ length: fanIn }, () => new Array<number>(fanOut).fill(0));
        const gradB = new Array<number>(fanOut).fill(0);

        for (let s = 0; s < nSamples; s += 1) {
          for (let j = 0; j < fanOut; j += 1) {
            gradB[j] += delta[s][j];
            for (let i = 0; i < fanIn; i += 1) {
              gradW[i][j] += activationsPrev[s][i] * delta[s][j];
            }
          }
        }

        for (let i = 0; i < fanIn; i += 1) {
          for (let j = 0; j < fanOut; j += 1) {
            gradW[i][j] = gradW[i][j] / nSamples + (this.alpha / nSamples) * this.coefs_[layer][i][j];
          }
        }
        for (let j = 0; j < fanOut; j += 1) {
          gradB[j] /= nSamples;
        }

        let nextDelta: Matrix | null = null;
        if (layer > 0) {
          const prevUnits = this.coefs_[layer].length;
          nextDelta = Array.from({ length: nSamples }, () => new Array<number>(prevUnits).fill(0));
          const zPrev = forward.preActivations[layer - 1];
          for (let s = 0; s < nSamples; s += 1) {
            for (let i = 0; i < prevUnits; i += 1) {
              let sum = 0;
              for (let j = 0; j < fanOut; j += 1) {
                sum += delta[s][j] * this.coefs_[layer][i][j];
              }
              nextDelta[s][i] = sum * activationDerivativeFromZ(zPrev[s][i], this.activation);
            }
          }
        }

        this.applyGradient(layer, gradW, gradB, adam);
        if (nextDelta) {
          delta = nextDelta;
        }
      }

      if (Math.abs(prevLoss - loss) < this.tolerance) {
        break;
      }
      prevLoss = loss;
    }

    this.nIter_ = nIter;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    const forward = this.forward(X);
    return forward.activations[forward.activations.length - 1].map((row) => row.slice());
  }

  predict(X: Matrix): Vector {
    const probabilities = this.predictProba(X);
    return probabilities.map((row) => this.classes_[argmaxRow(row)]);
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private classificationLoss(probabilities: Matrix, yOneHot: Matrix): number {
    const n = probabilities.length;
    let total = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < probabilities[i].length; j += 1) {
        if (yOneHot[i][j] === 1) {
          total += -Math.log(Math.max(1e-12, probabilities[i][j]));
        }
      }
    }
    let l2 = 0;
    for (let layer = 0; layer < this.coefs_.length; layer += 1) {
      for (let i = 0; i < this.coefs_[layer].length; i += 1) {
        for (let j = 0; j < this.coefs_[layer][i].length; j += 1) {
          const w = this.coefs_[layer][i][j];
          l2 += w * w;
        }
      }
    }
    return total / n + (this.alpha * l2) / (2 * n);
  }

  private forward(X: Matrix): { activations: Matrix[]; preActivations: Matrix[] } {
    const activations: Matrix[] = [X.map((row) => row.slice())];
    const preActivations: Matrix[] = [];

    for (let layer = 0; layer < this.coefs_.length; layer += 1) {
      const Z = matMulAddBias(activations[layer], this.coefs_[layer], this.intercepts_[layer]);
      preActivations.push(Z.map((row) => row.slice()));
      const isOutput = layer === this.coefs_.length - 1;
      if (isOutput) {
        softmaxInPlace(Z);
      } else {
        applyActivationInPlace(Z, this.activation);
      }
      activations.push(Z);
    }

    return { activations, preActivations };
  }

  private applyGradient(layer: number, gradW: Matrix, gradB: Vector, adam: AdamState | null): void {
    if (this.solver === "sgd" || adam === null) {
      for (let i = 0; i < this.coefs_[layer].length; i += 1) {
        for (let j = 0; j < this.coefs_[layer][i].length; j += 1) {
          this.coefs_[layer][i][j] -= this.learningRateInit * gradW[i][j];
        }
      }
      for (let j = 0; j < this.intercepts_[layer].length; j += 1) {
        this.intercepts_[layer][j] -= this.learningRateInit * gradB[j];
      }
      return;
    }

    adam.t += 1;
    const t = adam.t;
    for (let i = 0; i < this.coefs_[layer].length; i += 1) {
      for (let j = 0; j < this.coefs_[layer][i].length; j += 1) {
        const g = gradW[i][j];
        adam.mW[layer][i][j] = this.beta1 * adam.mW[layer][i][j] + (1 - this.beta1) * g;
        adam.vW[layer][i][j] = this.beta2 * adam.vW[layer][i][j] + (1 - this.beta2) * g * g;
        const mHat = adam.mW[layer][i][j] / (1 - Math.pow(this.beta1, t));
        const vHat = adam.vW[layer][i][j] / (1 - Math.pow(this.beta2, t));
        this.coefs_[layer][i][j] -= this.learningRateInit * mHat / (Math.sqrt(vHat) + this.epsilon);
      }
    }
    for (let j = 0; j < this.intercepts_[layer].length; j += 1) {
      const g = gradB[j];
      adam.mB[layer][j] = this.beta1 * adam.mB[layer][j] + (1 - this.beta1) * g;
      adam.vB[layer][j] = this.beta2 * adam.vB[layer][j] + (1 - this.beta2) * g * g;
      const mHat = adam.mB[layer][j] / (1 - Math.pow(this.beta1, t));
      const vHat = adam.vB[layer][j] / (1 - Math.pow(this.beta2, t));
      this.intercepts_[layer][j] -= this.learningRateInit * mHat / (Math.sqrt(vHat) + this.epsilon);
    }
  }

  private validateOptions(): void {
    if (!(this.activation === "identity" || this.activation === "logistic" || this.activation === "tanh" || this.activation === "relu")) {
      throw new Error(`activation must be one of identity/logistic/tanh/relu. Got ${this.activation}.`);
    }
    if (!(this.solver === "adam" || this.solver === "sgd")) {
      throw new Error(`solver must be 'adam' or 'sgd'. Got ${this.solver}.`);
    }
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new Error(`alpha must be finite and >= 0. Got ${this.alpha}.`);
    }
    if (!Number.isInteger(this.batchSize) || this.batchSize < 1) {
      throw new Error(`batchSize must be an integer >= 1. Got ${this.batchSize}.`);
    }
    if (!Number.isFinite(this.learningRateInit) || this.learningRateInit <= 0) {
      throw new Error(`learningRateInit must be finite and > 0. Got ${this.learningRateInit}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
    if (!Number.isFinite(this.beta1) || this.beta1 <= 0 || this.beta1 >= 1) {
      throw new Error(`beta1 must be in (0, 1). Got ${this.beta1}.`);
    }
    if (!Number.isFinite(this.beta2) || this.beta2 <= 0 || this.beta2 >= 1) {
      throw new Error(`beta2 must be in (0, 1). Got ${this.beta2}.`);
    }
    if (!Number.isFinite(this.epsilon) || this.epsilon <= 0) {
      throw new Error(`epsilon must be finite and > 0. Got ${this.epsilon}.`);
    }
  }

  private assertFitted(): void {
    if (!this.fitted || this.coefs_.length === 0 || this.intercepts_.length === 0 || this.nFeaturesIn_ === null) {
      throw new Error("MLPClassifier has not been fitted.");
    }
  }
}
