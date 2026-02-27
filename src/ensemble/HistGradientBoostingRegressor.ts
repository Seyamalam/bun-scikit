import type { Matrix, Vector } from "../types";
import { r2Score } from "../metrics/regression";
import { DecisionTreeRegressor } from "../tree/DecisionTreeRegressor";
import { assertFiniteVector, validateRegressionInputs } from "../utils/validation";

export interface HistGradientBoostingRegressorOptions {
  maxIter?: number;
  learningRate?: number;
  maxBins?: number;
  maxDepth?: number;
  maxLeafNodes?: number;
  minSamplesLeaf?: number;
  l2Regularization?: number;
  earlyStopping?: boolean;
  nIterNoChange?: number;
  validationFraction?: number;
  tolerance?: number;
  randomState?: number;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mean(y: Vector): number {
  let total = 0;
  for (let i = 0; i < y.length; i += 1) {
    total += y[i];
  }
  return total / y.length;
}

function meanSquaredError(y: Vector, pred: Vector): number {
  let total = 0;
  for (let i = 0; i < y.length; i += 1) {
    const diff = y[i] - pred[i];
    total += diff * diff;
  }
  return total / Math.max(1, y.length);
}

function subsetMatrix(X: Matrix, indices: number[]): Matrix {
  const out = new Array<Matrix[number]>(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = X[indices[i]];
  }
  return out;
}

function subsetVector(y: Vector, indices: number[]): Vector {
  const out = new Array<number>(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    out[i] = y[indices[i]];
  }
  return out;
}

function resolveMaxDepth(maxDepth: number | undefined, maxLeafNodes: number): number {
  if (maxDepth !== undefined) {
    return maxDepth;
  }
  return Math.max(1, Math.ceil(Math.log2(Math.max(2, maxLeafNodes))));
}

export class HistGradientBoostingRegressor {
  estimators_: DecisionTreeRegressor[] = [];
  baseline_: number | null = null;
  featureImportances_: Vector | null = null;

  private readonly maxIter: number;
  private readonly learningRate: number;
  private readonly maxBins: number;
  private readonly maxDepth?: number;
  private readonly maxLeafNodes: number;
  private readonly minSamplesLeaf: number;
  private readonly l2Regularization: number;
  private readonly earlyStopping: boolean;
  private readonly nIterNoChange: number;
  private readonly validationFraction: number;
  private readonly tolerance: number;
  private readonly randomState?: number;
  private isFitted = false;

  constructor(options: HistGradientBoostingRegressorOptions = {}) {
    this.maxIter = options.maxIter ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxBins = options.maxBins ?? 255;
    this.maxDepth = options.maxDepth;
    this.maxLeafNodes = options.maxLeafNodes ?? 31;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 20;
    this.l2Regularization = options.l2Regularization ?? 0;
    this.earlyStopping = options.earlyStopping ?? true;
    this.nIterNoChange = options.nIterNoChange ?? 10;
    this.validationFraction = options.validationFraction ?? 0.1;
    this.tolerance = options.tolerance ?? 1e-7;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new Error(`maxIter must be an integer >= 1. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new Error(`learningRate must be finite and > 0. Got ${this.learningRate}.`);
    }
    if (!Number.isInteger(this.maxBins) || this.maxBins < 2) {
      throw new Error(`maxBins must be an integer >= 2. Got ${this.maxBins}.`);
    }
    if (this.maxDepth !== undefined && (!Number.isInteger(this.maxDepth) || this.maxDepth < 1)) {
      throw new Error(`maxDepth must be an integer >= 1 when provided. Got ${this.maxDepth}.`);
    }
    if (!Number.isInteger(this.maxLeafNodes) || this.maxLeafNodes < 2) {
      throw new Error(`maxLeafNodes must be an integer >= 2. Got ${this.maxLeafNodes}.`);
    }
    if (!Number.isInteger(this.minSamplesLeaf) || this.minSamplesLeaf < 1) {
      throw new Error(`minSamplesLeaf must be an integer >= 1. Got ${this.minSamplesLeaf}.`);
    }
    if (!Number.isFinite(this.l2Regularization) || this.l2Regularization < 0) {
      throw new Error(
        `l2Regularization must be finite and >= 0. Got ${this.l2Regularization}.`,
      );
    }
    if (!Number.isInteger(this.nIterNoChange) || this.nIterNoChange < 1) {
      throw new Error(`nIterNoChange must be an integer >= 1. Got ${this.nIterNoChange}.`);
    }
    if (
      !Number.isFinite(this.validationFraction) ||
      this.validationFraction <= 0 ||
      this.validationFraction >= 0.5
    ) {
      throw new Error(
        `validationFraction must be finite and in (0, 0.5). Got ${this.validationFraction}.`,
      );
    }
    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`tolerance must be finite and >= 0. Got ${this.tolerance}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateRegressionInputs(X, y);
    const nSamples = X.length;
    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    const resolvedMaxDepth = resolveMaxDepth(this.maxDepth, this.maxLeafNodes);

    let trainX = X;
    let trainY = y;
    let valX: Matrix = [];
    let valY: Vector = [];

    if (this.earlyStopping && nSamples >= 20) {
      const indices = Array.from({ length: nSamples }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i -= 1) {
        const j = Math.floor(random() * (i + 1));
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }
      const valCount = Math.max(1, Math.floor(this.validationFraction * nSamples));
      const splitIndex = nSamples - valCount;
      const trainIndices = indices.slice(0, splitIndex);
      const valIndices = indices.slice(splitIndex);
      trainX = subsetMatrix(X, trainIndices);
      trainY = subsetVector(y, trainIndices);
      valX = subsetMatrix(X, valIndices);
      valY = subsetVector(y, valIndices);
    }

    this.estimators_ = [];
    this.baseline_ = mean(trainY);
    const trainPred = new Array<number>(trainY.length).fill(this.baseline_);
    const valPred = valY.length > 0 ? new Array<number>(valY.length).fill(this.baseline_) : [];

    let bestLoss = Number.POSITIVE_INFINITY;
    let roundsWithoutImprovement = 0;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const residuals = new Array<number>(trainY.length);
      for (let i = 0; i < trainY.length; i += 1) {
        residuals[i] = trainY[i] - trainPred[i];
      }

      const tree = new DecisionTreeRegressor({
        maxDepth: resolvedMaxDepth,
        minSamplesSplit: 2,
        minSamplesLeaf: this.minSamplesLeaf,
        randomState: this.randomState === undefined ? undefined : this.randomState + iter + 1,
      }).fit(trainX, residuals);
      this.estimators_.push(tree);

      const trainUpdate = tree.predict(trainX);
      for (let i = 0; i < trainPred.length; i += 1) {
        trainPred[i] += this.learningRate * trainUpdate[i];
      }

      if (valY.length > 0) {
        const valUpdate = tree.predict(valX);
        for (let i = 0; i < valPred.length; i += 1) {
          valPred[i] += this.learningRate * valUpdate[i];
        }
      }

      if (this.earlyStopping) {
        const currentLoss =
          valY.length > 0 ? meanSquaredError(valY, valPred) : meanSquaredError(trainY, trainPred);
        if (bestLoss - currentLoss > this.tolerance) {
          bestLoss = currentLoss;
          roundsWithoutImprovement = 0;
        } else {
          roundsWithoutImprovement += 1;
          if (roundsWithoutImprovement >= this.nIterNoChange) {
            break;
          }
        }
      }
    }

    this.computeFeatureImportances(X[0].length);
    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();
    const out = new Array<number>(X.length).fill(this.baseline_!);
    for (let t = 0; t < this.estimators_.length; t += 1) {
      const update = this.estimators_[t].predict(X);
      for (let row = 0; row < X.length; row += 1) {
        out[row] += this.learningRate * update[row];
      }
    }
    return out;
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return r2Score(y, this.predict(X));
  }

  private computeFeatureImportances(featureCount: number): void {
    const raw = new Array<number>(featureCount).fill(0);
    for (let i = 0; i < this.estimators_.length; i += 1) {
      const importances = this.estimators_[i].featureImportances_;
      if (!importances) {
        continue;
      }
      for (let j = 0; j < featureCount; j += 1) {
        raw[j] += importances[j];
      }
    }
    let sum = 0;
    for (let i = 0; i < raw.length; i += 1) {
      sum += raw[i];
    }
    this.featureImportances_ =
      sum > 0 ? raw.map((value) => value / sum) : new Array<number>(featureCount).fill(0);
  }

  private assertFitted(): void {
    if (!this.isFitted || this.baseline_ === null) {
      throw new Error("HistGradientBoostingRegressor has not been fitted.");
    }
  }
}
