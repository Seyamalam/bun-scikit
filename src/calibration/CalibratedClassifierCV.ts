import type { Matrix, Vector } from "../types";
import { accuracyScore } from "../metrics/classification";
import { StratifiedKFold, type StratifiedKFoldOptions } from "../model_selection/StratifiedKFold";
import {
  assertFiniteVector,
  validateClassificationInputs,
} from "../utils/validation";

export type CalibrationMethod = "sigmoid" | "isotonic";

export interface CalibratedClassifierCVOptions {
  cv?: number;
  method?: CalibrationMethod;
  ensemble?: boolean;
  randomState?: number;
}

type ProbaLikeEstimator = {
  fit(X: Matrix, y: Vector): unknown;
  predict(X: Matrix): Vector;
  predictProba?: (X: Matrix) => Matrix;
  decisionFunction?: (X: Matrix) => Vector;
};

interface BinaryCalibrator {
  fit(scores: Vector, y: Vector): this;
  predict(scores: Vector): Vector;
}

class SigmoidCalibrator implements BinaryCalibrator {
  private mean = 0;
  private scale = 1;
  private a = 1;
  private b = 0;

  fit(rawScores: Vector, y: Vector): this {
    if (rawScores.length !== y.length || rawScores.length === 0) {
      throw new Error("Sigmoid calibrator requires non-empty equal-length score/target arrays.");
    }

    let mean = 0;
    for (let i = 0; i < rawScores.length; i += 1) {
      mean += rawScores[i];
    }
    mean /= rawScores.length;

    let variance = 0;
    for (let i = 0; i < rawScores.length; i += 1) {
      const centered = rawScores[i] - mean;
      variance += centered * centered;
    }
    variance /= rawScores.length;

    this.mean = mean;
    this.scale = Math.sqrt(variance) || 1;
    this.a = 1;
    this.b = 0;

    const learningRate = 0.1;
    const regularization = 1e-6;
    const maxIter = 3000;
    const tolerance = 1e-8;
    const n = rawScores.length;

    for (let iter = 0; iter < maxIter; iter += 1) {
      let gradA = 0;
      let gradB = 0;
      for (let i = 0; i < n; i += 1) {
        const score = (rawScores[i] - this.mean) / this.scale;
        const prob = this.sigmoid(this.a * score + this.b);
        const diff = prob - y[i];
        gradA += diff * score;
        gradB += diff;
      }

      gradA = gradA / n + regularization * this.a;
      gradB /= n;
      const deltaA = learningRate * gradA;
      const deltaB = learningRate * gradB;
      this.a -= deltaA;
      this.b -= deltaB;

      if (Math.max(Math.abs(deltaA), Math.abs(deltaB)) < tolerance) {
        break;
      }
    }

    return this;
  }

  predict(scores: Vector): Vector {
    const out = new Array<number>(scores.length);
    for (let i = 0; i < scores.length; i += 1) {
      const z = (scores[i] - this.mean) / this.scale;
      out[i] = this.sigmoid(this.a * z + this.b);
    }
    return out;
  }

  private sigmoid(value: number): number {
    if (value >= 0) {
      const expNeg = Math.exp(-value);
      return 1 / (1 + expNeg);
    }
    const expPos = Math.exp(value);
    return expPos / (1 + expPos);
  }
}

class IsotonicCalibrator implements BinaryCalibrator {
  private thresholds: number[] = [];
  private values: number[] = [];

  fit(rawScores: Vector, y: Vector): this {
    if (rawScores.length !== y.length || rawScores.length === 0) {
      throw new Error("Isotonic calibrator requires non-empty equal-length score/target arrays.");
    }

    const sorted = rawScores
      .map((score, index) => ({ score, label: y[index] }))
      .sort((a, b) => a.score - b.score);

    type Block = { start: number; end: number; weight: number; value: number };
    const blocks: Block[] = sorted.map((item, index) => ({
      start: index,
      end: index,
      weight: 1,
      value: item.label,
    }));

    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < blocks.length - 1; i += 1) {
        if (blocks[i].value > blocks[i + 1].value) {
          const totalWeight = blocks[i].weight + blocks[i + 1].weight;
          const mergedValue =
            (blocks[i].value * blocks[i].weight + blocks[i + 1].value * blocks[i + 1].weight) /
            totalWeight;
          blocks[i] = {
            start: blocks[i].start,
            end: blocks[i + 1].end,
            weight: totalWeight,
            value: mergedValue,
          };
          blocks.splice(i + 1, 1);
          merged = true;
          break;
        }
      }
    }

    this.thresholds = new Array<number>(blocks.length);
    this.values = new Array<number>(blocks.length);
    for (let i = 0; i < blocks.length; i += 1) {
      this.thresholds[i] = sorted[blocks[i].end].score;
      this.values[i] = blocks[i].value;
    }
    return this;
  }

  predict(scores: Vector): Vector {
    if (this.thresholds.length === 0 || this.values.length === 0) {
      throw new Error("Isotonic calibrator has not been fitted.");
    }

    const out = new Array<number>(scores.length);
    for (let i = 0; i < scores.length; i += 1) {
      out[i] = this.predictOne(scores[i]);
    }
    return out;
  }

  private predictOne(score: number): number {
    if (score <= this.thresholds[0]) {
      return this.values[0];
    }
    if (score >= this.thresholds[this.thresholds.length - 1]) {
      return this.values[this.values.length - 1];
    }

    let left = 0;
    let right = this.thresholds.length - 1;
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (score <= this.thresholds[mid]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    return this.values[left];
  }
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

function resolveStratifiedFolds(
  X: Matrix,
  y: Vector,
  cv: number | undefined,
  randomState: number,
): ReturnType<StratifiedKFold["split"]> {
  const nSplits = cv ?? 5;
  if (!Number.isInteger(nSplits) || nSplits < 2) {
    throw new Error(`cv must be an integer >= 2. Got ${nSplits}.`);
  }
  const splitter = new StratifiedKFold({
    nSplits,
    shuffle: true,
    randomState,
  } satisfies StratifiedKFoldOptions);
  return splitter.split(X, y);
}

function estimatorScores(estimator: ProbaLikeEstimator, X: Matrix): Vector {
  if (typeof estimator.decisionFunction === "function") {
    return estimator.decisionFunction(X);
  }
  if (typeof estimator.predictProba === "function") {
    const probabilities = estimator.predictProba(X);
    return probabilities.map((row) => row[1]);
  }
  return estimator.predict(X);
}

function makeCalibrator(method: CalibrationMethod): BinaryCalibrator {
  if (method === "sigmoid") {
    return new SigmoidCalibrator();
  }
  return new IsotonicCalibrator();
}

export class CalibratedClassifierCV {
  classes_: Vector = [0, 1];

  private readonly estimatorFactory: () => ProbaLikeEstimator;
  private readonly cv?: number;
  private readonly method: CalibrationMethod;
  private readonly ensemble: boolean;
  private readonly randomState: number;
  private ensembleEstimators: Array<{ estimator: ProbaLikeEstimator; calibrator: BinaryCalibrator }> =
    [];
  private finalEstimator: ProbaLikeEstimator | null = null;
  private finalCalibrator: BinaryCalibrator | null = null;
  private isFitted = false;

  constructor(
    estimatorFactory: () => ProbaLikeEstimator,
    options: CalibratedClassifierCVOptions = {},
  ) {
    if (typeof estimatorFactory !== "function") {
      throw new Error("estimatorFactory must be a function.");
    }
    this.estimatorFactory = estimatorFactory;
    this.cv = options.cv;
    this.method = options.method ?? "sigmoid";
    this.ensemble = options.ensemble ?? true;
    this.randomState = options.randomState ?? 42;
  }

  fit(X: Matrix, y: Vector): this {
    validateClassificationInputs(X, y);

    const folds = resolveStratifiedFolds(X, y, this.cv, this.randomState);
    if (folds.length < 2) {
      throw new Error("CalibratedClassifierCV requires at least two folds.");
    }

    this.ensembleEstimators = [];
    this.finalEstimator = null;
    this.finalCalibrator = null;

    if (this.ensemble) {
      for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
        const fold = folds[foldIndex];
        const estimator = this.estimatorFactory();
        estimator.fit(subsetMatrix(X, fold.trainIndices), subsetVector(y, fold.trainIndices));
        const foldScores = estimatorScores(estimator, subsetMatrix(X, fold.testIndices));
        const foldTargets = subsetVector(y, fold.testIndices);
        const calibrator = makeCalibrator(this.method);
        calibrator.fit(foldScores, foldTargets);
        this.ensembleEstimators.push({ estimator, calibrator });
      }
    } else {
      const oofScores = new Array<number>(X.length).fill(0);
      for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
        const fold = folds[foldIndex];
        const estimator = this.estimatorFactory();
        estimator.fit(subsetMatrix(X, fold.trainIndices), subsetVector(y, fold.trainIndices));
        const foldScores = estimatorScores(estimator, subsetMatrix(X, fold.testIndices));
        for (let i = 0; i < fold.testIndices.length; i += 1) {
          oofScores[fold.testIndices[i]] = foldScores[i];
        }
      }

      this.finalEstimator = this.estimatorFactory();
      this.finalEstimator.fit(X, y);
      const calibrator = makeCalibrator(this.method);
      calibrator.fit(oofScores, y);
      this.finalCalibrator = calibrator;
    }

    this.isFitted = true;
    return this;
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();

    if (this.ensemble) {
      const aggregate = new Array<number>(X.length).fill(0);
      for (let i = 0; i < this.ensembleEstimators.length; i += 1) {
        const pair = this.ensembleEstimators[i];
        const probabilities = pair.calibrator.predict(estimatorScores(pair.estimator, X));
        for (let j = 0; j < probabilities.length; j += 1) {
          aggregate[j] += probabilities[j];
        }
      }
      for (let i = 0; i < aggregate.length; i += 1) {
        aggregate[i] /= this.ensembleEstimators.length;
      }
      return aggregate.map((prob) => [1 - prob, prob]);
    }

    const rawScores = estimatorScores(this.finalEstimator!, X);
    const calibrated = this.finalCalibrator!.predict(rawScores);
    return calibrated.map((prob) => [1 - prob, prob]);
  }

  predict(X: Matrix): Vector {
    return this.predictProba(X).map((pair) => (pair[1] >= 0.5 ? 1 : 0));
  }

  score(X: Matrix, y: Vector): number {
    assertFiniteVector(y);
    return accuracyScore(y, this.predict(X));
  }

  private assertFitted(): void {
    if (!this.isFitted) {
      throw new Error("CalibratedClassifierCV has not been fitted.");
    }
    if (this.ensemble && this.ensembleEstimators.length === 0) {
      throw new Error("CalibratedClassifierCV has not been fitted.");
    }
    if (!this.ensemble && (!this.finalEstimator || !this.finalCalibrator)) {
      throw new Error("CalibratedClassifierCV has not been fitted.");
    }
  }
}
