import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { assertFiniteVector, validateBinaryClassificationInputs } from "../utils/validation";
import { HistGradientBoostingRegressor } from "./HistGradientBoostingRegressor";

export interface HistGradientBoostingClassifierOptions {
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

function sigmoid(z: number): number {
  if (z >= 0) {
    const expNeg = Math.exp(-z);
    return 1 / (1 + expNeg);
  }
  const expPos = Math.exp(z);
  return expPos / (1 + expPos);
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

function binaryLogLoss(y: Vector, logits: Vector): number {
  let total = 0;
  for (let i = 0; i < y.length; i += 1) {
    const p = Math.min(1 - 1e-12, Math.max(1e-12, sigmoid(logits[i])));
    total += y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
  }
  return -total / Math.max(1, y.length);
}

export class HistGradientBoostingClassifier {
  classes_: Vector = [0, 1];
  estimators_: HistGradientBoostingRegressor[] = [];
  baselineLogit_: number | null = null;
  featureImportances_: Vector | null = null;

  private readonly maxIter: number;
  private readonly learningRate: number;
  private readonly maxBins: number;
  private readonly maxDepth?: number;
  private readonly maxLeafNodes?: number;
  private readonly minSamplesLeaf: number;
  private readonly l2Regularization: number;
  private readonly earlyStopping: boolean;
  private readonly nIterNoChange: number;
  private readonly validationFraction: number;
  private readonly tolerance: number;
  private readonly randomState?: number;
  private isFitted = false;

  constructor(options: HistGradientBoostingClassifierOptions = {}) {
    this.maxIter = options.maxIter ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxBins = options.maxBins ?? 255;
    this.maxDepth = options.maxDepth;
    this.maxLeafNodes = options.maxLeafNodes;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 20;
    this.l2Regularization = options.l2Regularization ?? 0;
    this.earlyStopping = options.earlyStopping ?? true;
    this.nIterNoChange = options.nIterNoChange ?? 10;
    this.validationFraction = options.validationFraction ?? 0.1;
    this.tolerance = options.tolerance ?? 1e-7;
    this.randomState = options.randomState;
  }

  fit(X: Matrix, y: Vector): this {
    validateBinaryClassificationInputs(X, y);
    this.estimators_ = [];

    const random = this.randomState === undefined ? Math.random : mulberry32(this.randomState);
    let trainX = X;
    let trainY = y;
    let valX: Matrix = [];
    let valY: Vector = [];

    if (this.earlyStopping && X.length >= 20) {
      const indices = Array.from({ length: X.length }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i -= 1) {
        const j = Math.floor(random() * (i + 1));
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }
      const valCount = Math.max(1, Math.floor(this.validationFraction * X.length));
      const splitIndex = X.length - valCount;
      const trainIndices = indices.slice(0, splitIndex);
      const valIndices = indices.slice(splitIndex);
      trainX = subsetMatrix(X, trainIndices);
      trainY = subsetVector(y, trainIndices);
      valX = subsetMatrix(X, valIndices);
      valY = subsetVector(y, valIndices);
    }

    let positive = 0;
    for (let i = 0; i < trainY.length; i += 1) {
      positive += trainY[i];
    }
    const p = Math.min(1 - 1e-12, Math.max(1e-12, positive / trainY.length));
    this.baselineLogit_ = Math.log(p / (1 - p));

    const logits = new Array<number>(trainY.length).fill(this.baselineLogit_);
    const valLogits = valY.length > 0 ? new Array<number>(valY.length).fill(this.baselineLogit_) : [];

    let bestLoss = Number.POSITIVE_INFINITY;
    let roundsWithoutImprovement = 0;

    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const pseudoResiduals = new Array<number>(trainY.length);
      for (let i = 0; i < trainY.length; i += 1) {
        pseudoResiduals[i] = trainY[i] - sigmoid(logits[i]);
      }

      const regressor = new HistGradientBoostingRegressor({
        maxIter: 1,
        learningRate: 1,
        maxBins: this.maxBins,
        maxDepth: this.maxDepth,
        maxLeafNodes: this.maxLeafNodes,
        minSamplesLeaf: this.minSamplesLeaf,
        l2Regularization: this.l2Regularization,
        earlyStopping: false,
        randomState: this.randomState === undefined ? undefined : this.randomState + iter + 1,
      }).fit(trainX, pseudoResiduals);

      const trainUpdate = regressor.predict(trainX);
      for (let i = 0; i < logits.length; i += 1) {
        logits[i] += this.learningRate * trainUpdate[i];
      }
      if (valY.length > 0) {
        const valUpdate = regressor.predict(valX);
        for (let i = 0; i < valLogits.length; i += 1) {
          valLogits[i] += this.learningRate * valUpdate[i];
        }
      }

      this.estimators_.push(regressor);
      if (this.earlyStopping) {
        const currentLoss =
          valY.length > 0 ? binaryLogLoss(valY, valLogits) : binaryLogLoss(trainY, logits);
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

  decisionFunction(X: Matrix): Vector {
    this.assertFitted();
    const out = new Array<number>(X.length).fill(this.baselineLogit_!);
    for (let i = 0; i < this.estimators_.length; i += 1) {
      const update = this.estimators_[i].predict(X);
      for (let row = 0; row < out.length; row += 1) {
        out[row] += this.learningRate * update[row];
      }
    }
    return out;
  }

  predictProba(X: Matrix): Matrix {
    return this.decisionFunction(X).map((value) => {
      const p1 = sigmoid(value);
      return [1 - p1, p1];
    });
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((row) => (row[1] >= 0.5 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
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
    if (!this.isFitted || this.baselineLogit_ === null) {
      throw new Error("HistGradientBoostingClassifier has not been fitted.");
    }
  }
}
