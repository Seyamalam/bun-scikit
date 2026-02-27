import type { Matrix, Vector } from "../types";

function validateInputs(yTrue: number[], yPred: number[]): void {
  if (yTrue.length === 0 || yPred.length === 0) {
    throw new Error("yTrue and yPred must be non-empty.");
  }

  if (yTrue.length !== yPred.length) {
    throw new Error(`Length mismatch: yTrue=${yTrue.length}, yPred=${yPred.length}.`);
  }
}

function validateMultilabelInputs(yTrue: Matrix, yPred: Matrix): void {
  if (yTrue.length === 0 || yPred.length === 0) {
    throw new Error("yTrue and yPred must be non-empty.");
  }
  if (yTrue.length !== yPred.length) {
    throw new Error(`Length mismatch: yTrue=${yTrue.length}, yPred=${yPred.length}.`);
  }
  const width = yTrue[0].length;
  if (width === 0) {
    throw new Error("Multilabel indicator matrices must have at least one column.");
  }
  for (let i = 0; i < yTrue.length; i += 1) {
    if (yTrue[i].length !== width || yPred[i].length !== width) {
      throw new Error("All multilabel rows must share the same width.");
    }
  }
}

function resolveSampleWeight(sampleWeight: Vector | undefined, nSamples: number): Vector {
  if (!sampleWeight) {
    return new Array<number>(nSamples).fill(1);
  }
  if (sampleWeight.length !== nSamples) {
    throw new Error(
      `sampleWeight length must match sample count. Got ${sampleWeight.length} and ${nSamples}.`,
    );
  }
  const out = new Array<number>(sampleWeight.length);
  for (let i = 0; i < sampleWeight.length; i += 1) {
    const weight = sampleWeight[i];
    if (!Number.isFinite(weight) || weight < 0) {
      throw new Error(`sampleWeight must contain finite non-negative values. Got ${weight} at ${i}.`);
    }
    out[i] = weight;
  }
  return out;
}

function validateBinaryTargets(yTrue: number[]): void {
  for (let i = 0; i < yTrue.length; i += 1) {
    const value = yTrue[i];
    if (!(value === 0 || value === 1)) {
      throw new Error(`Binary classification target expected (0/1). Found ${value} at index ${i}.`);
    }
  }
}

function clampProbability(value: number, eps: number): number {
  if (!Number.isFinite(value)) {
    throw new Error(`Probability must be finite. Got ${value}.`);
  }
  if (value < eps) {
    return eps;
  }
  if (value > 1 - eps) {
    return 1 - eps;
  }
  return value;
}

function confusionCounts(yTrue: number[], yPred: number[], positiveLabel: number): {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
} {
  return confusionCountsWithWeights(yTrue, yPred, positiveLabel);
}

function confusionCountsWithWeights(yTrue: number[], yPred: number[], positiveLabel: number, sampleWeight?: Vector): {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
} {
  validateInputs(yTrue, yPred);
  const weights = resolveSampleWeight(sampleWeight, yTrue.length);

  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;

  for (let i = 0; i < yTrue.length; i += 1) {
    const truthPositive = yTrue[i] === positiveLabel;
    const predPositive = yPred[i] === positiveLabel;
    const weight = weights[i];

    if (truthPositive && predPositive) {
      tp += weight;
    } else if (!truthPositive && predPositive) {
      fp += weight;
    } else if (truthPositive && !predPositive) {
      fn += weight;
    } else {
      tn += weight;
    }
  }

  return { tp, fp, fn, tn };
}

export interface ConfusionMatrixResult {
  labels: number[];
  matrix: number[][];
}

export interface ClassificationReportLabelMetrics {
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface ClassificationReportResult {
  labels: number[];
  perLabel: Record<string, ClassificationReportLabelMetrics>;
  accuracy: number;
  macroAvg: ClassificationReportLabelMetrics;
  weightedAvg: ClassificationReportLabelMetrics;
}

export function accuracyScore(yTrue: number[] | Matrix, yPred: number[] | Matrix, sampleWeight?: Vector): number {
  if (Array.isArray(yTrue[0]) || Array.isArray(yPred[0])) {
    const trueMatrix = yTrue as Matrix;
    const predMatrix = yPred as Matrix;
    validateMultilabelInputs(trueMatrix, predMatrix);
    const weights = resolveSampleWeight(sampleWeight, trueMatrix.length);
    let weightSum = 0;
    let correct = 0;
    for (let i = 0; i < trueMatrix.length; i += 1) {
      let rowEqual = true;
      for (let j = 0; j < trueMatrix[i].length; j += 1) {
        if (trueMatrix[i][j] !== predMatrix[i][j]) {
          rowEqual = false;
          break;
        }
      }
      weightSum += weights[i];
      if (rowEqual) {
        correct += weights[i];
      }
    }
    if (weightSum === 0) {
      return 0;
    }
    return correct / weightSum;
  }

  const yTrueVector = yTrue as number[];
  const yPredVector = yPred as number[];
  validateInputs(yTrueVector, yPredVector);
  const weights = resolveSampleWeight(sampleWeight, yTrueVector.length);

  let correct = 0;
  let weightSum = 0;
  for (let i = 0; i < yTrueVector.length; i += 1) {
    weightSum += weights[i];
    if (yTrueVector[i] === yPredVector[i]) {
      correct += weights[i];
    }
  }
  if (weightSum === 0) {
    return 0;
  }
  return correct / weightSum;
}

export function precisionScore(
  yTrue: number[],
  yPred: number[],
  positiveLabel = 1,
  sampleWeight?: Vector,
): number {
  const { tp, fp } = confusionCountsWithWeights(yTrue, yPred, positiveLabel, sampleWeight);
  const denominator = tp + fp;
  if (denominator === 0) {
    return 0;
  }
  return tp / denominator;
}

export function recallScore(yTrue: number[], yPred: number[], positiveLabel = 1, sampleWeight?: Vector): number {
  const { tp, fn } = confusionCountsWithWeights(yTrue, yPred, positiveLabel, sampleWeight);
  const denominator = tp + fn;
  if (denominator === 0) {
    return 0;
  }
  return tp / denominator;
}

export function f1Score(yTrue: number[], yPred: number[], positiveLabel = 1, sampleWeight?: Vector): number {
  const precision = precisionScore(yTrue, yPred, positiveLabel, sampleWeight);
  const recall = recallScore(yTrue, yPred, positiveLabel, sampleWeight);
  const denominator = precision + recall;
  if (denominator === 0) {
    return 0;
  }
  return (2 * precision * recall) / denominator;
}

export function confusionMatrix(
  yTrue: number[],
  yPred: number[],
  labels?: number[],
): ConfusionMatrixResult {
  validateInputs(yTrue, yPred);
  const resolvedLabels =
    labels && labels.length > 0
      ? labels.slice()
      : uniqueSorted([...yTrue, ...yPred]);
  if (resolvedLabels.length === 0) {
    throw new Error("confusionMatrix requires at least one label.");
  }

  const labelToIndex = new Map<number, number>();
  for (let i = 0; i < resolvedLabels.length; i += 1) {
    labelToIndex.set(resolvedLabels[i], i);
  }

  const matrix = Array.from({ length: resolvedLabels.length }, () =>
    new Array<number>(resolvedLabels.length).fill(0),
  );

  for (let i = 0; i < yTrue.length; i += 1) {
    const trueLabel = yTrue[i];
    const predLabel = yPred[i];
    const trueIndex = labelToIndex.get(trueLabel);
    const predIndex = labelToIndex.get(predLabel);
    if (trueIndex === undefined || predIndex === undefined) {
      continue;
    }
    matrix[trueIndex][predIndex] += 1;
  }

  return { labels: resolvedLabels, matrix };
}

export function logLoss(yTrue: number[], yPredProb: number[] | Matrix, eps = 1e-15): number {
  if (!Number.isFinite(eps) || eps <= 0 || eps >= 0.5) {
    throw new Error(`eps must be finite and in (0, 0.5). Got ${eps}.`);
  }

  if (Array.isArray(yPredProb[0])) {
    const probabilities = yPredProb as Matrix;
    if (yTrue.length === 0 || probabilities.length === 0 || yTrue.length !== probabilities.length) {
      throw new Error("yTrue and yPredProb must be non-empty and equal-length.");
    }
    const labels = uniqueSorted(yTrue);
    const classToIndex = labelToIndex(labels);
    if (labels.length < 2) {
      throw new Error("logLoss requires at least two classes.");
    }

    let total = 0;
    for (let i = 0; i < yTrue.length; i += 1) {
      if (probabilities[i].length < labels.length) {
        throw new Error(
          `Each probability row must have at least ${labels.length} classes. Got ${probabilities[i].length}.`,
        );
      }
      const idx = classToIndex.get(yTrue[i]);
      if (idx === undefined) {
        throw new Error(`Unknown label '${yTrue[i]}' in yTrue.`);
      }
      const p = clampProbability(probabilities[i][idx], eps);
      total += -Math.log(p);
    }
    return total / yTrue.length;
  }

  const probabilities = yPredProb as number[];
  validateInputs(yTrue, probabilities);
  validateBinaryTargets(yTrue);

  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    const p1 = clampProbability(probabilities[i], eps);
    const p0 = 1 - p1;
    total += -(yTrue[i] * Math.log(p1) + (1 - yTrue[i]) * Math.log(p0));
  }

  return total / yTrue.length;
}

export function rocAucScore(yTrue: number[], yScore: number[]): number {
  validateInputs(yTrue, yScore);
  validateBinaryTargets(yTrue);

  let positiveCount = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    if (yTrue[i] === 1) {
      positiveCount += 1;
    }
  }
  const negativeCount = yTrue.length - positiveCount;
  if (positiveCount === 0 || negativeCount === 0) {
    throw new Error("rocAucScore requires both positive and negative samples.");
  }

  const pairs = yScore.map((score, idx) => ({ score, label: yTrue[idx] }));
  pairs.sort((a, b) => a.score - b.score);

  // Average ranks for ties.
  const ranks = new Array<number>(pairs.length);
  let cursor = 0;
  while (cursor < pairs.length) {
    let tieEnd = cursor + 1;
    while (tieEnd < pairs.length && pairs[tieEnd].score === pairs[cursor].score) {
      tieEnd += 1;
    }
    const startRank = cursor + 1;
    const endRank = tieEnd;
    const averageRank = 0.5 * (startRank + endRank);
    for (let i = cursor; i < tieEnd; i += 1) {
      ranks[i] = averageRank;
    }
    cursor = tieEnd;
  }

  let rankSumPositives = 0;
  for (let i = 0; i < pairs.length; i += 1) {
    if (pairs[i].label === 1) {
      rankSumPositives += ranks[i];
    }
  }

  const u = rankSumPositives - (positiveCount * (positiveCount + 1)) / 2;
  return u / (positiveCount * negativeCount);
}

export function classificationReport(
  yTrue: number[],
  yPred: number[],
  labels?: number[],
): ClassificationReportResult {
  validateInputs(yTrue, yPred);

  const { labels: resolvedLabels, matrix } = confusionMatrix(yTrue, yPred, labels);
  const perLabel: Record<string, ClassificationReportLabelMetrics> = {};

  let macroPrecision = 0;
  let macroRecall = 0;
  let macroF1 = 0;
  let weightedPrecision = 0;
  let weightedRecall = 0;
  let weightedF1 = 0;

  for (let labelIndex = 0; labelIndex < resolvedLabels.length; labelIndex += 1) {
    const label = resolvedLabels[labelIndex];
    let rowSum = 0;
    let colSum = 0;
    for (let j = 0; j < resolvedLabels.length; j += 1) {
      rowSum += matrix[labelIndex][j];
      colSum += matrix[j][labelIndex];
    }

    const tp = matrix[labelIndex][labelIndex];
    const precision = colSum === 0 ? 0 : tp / colSum;
    const recall = rowSum === 0 ? 0 : tp / rowSum;
    const denom = precision + recall;
    const f1 = denom === 0 ? 0 : (2 * precision * recall) / denom;

    perLabel[String(label)] = {
      precision,
      recall,
      f1Score: f1,
      support: rowSum,
    };

    macroPrecision += precision;
    macroRecall += recall;
    macroF1 += f1;
    weightedPrecision += precision * rowSum;
    weightedRecall += recall * rowSum;
    weightedF1 += f1 * rowSum;
  }

  const nLabels = resolvedLabels.length;
  const totalSupport = yTrue.length;

  return {
    labels: resolvedLabels,
    perLabel,
    accuracy: accuracyScore(yTrue, yPred),
    macroAvg: {
      precision: macroPrecision / nLabels,
      recall: macroRecall / nLabels,
      f1Score: macroF1 / nLabels,
      support: totalSupport,
    },
    weightedAvg: {
      precision: weightedPrecision / totalSupport,
      recall: weightedRecall / totalSupport,
      f1Score: weightedF1 / totalSupport,
      support: totalSupport,
    },
  };
}

function uniqueSorted(values: Vector): Vector {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function labelToIndex(labels: Vector): Map<number, number> {
  const map = new Map<number, number>();
  for (let i = 0; i < labels.length; i += 1) {
    map.set(labels[i], i);
  }
  return map;
}

export function balancedAccuracyScore(yTrue: number[], yPred: number[]): number {
  const report = classificationReport(yTrue, yPred);
  return report.macroAvg.recall;
}

export function matthewsCorrcoef(
  yTrue: number[],
  yPred: number[],
  positiveLabel = 1,
): number {
  const labels = uniqueSorted([...yTrue, ...yPred]);
  if (labels.length > 2) {
    const { matrix } = confusionMatrix(yTrue, yPred, labels);
    const k = labels.length;
    let c = 0;
    let s = 0;
    let sumPkTk = 0;
    let sumPkSq = 0;
    let sumTkSq = 0;

    const rowSums = new Array<number>(k).fill(0);
    const colSums = new Array<number>(k).fill(0);
    for (let i = 0; i < k; i += 1) {
      for (let j = 0; j < k; j += 1) {
        const value = matrix[i][j];
        s += value;
        rowSums[i] += value;
        colSums[j] += value;
        if (i === j) {
          c += value;
        }
      }
    }
    for (let i = 0; i < k; i += 1) {
      sumPkTk += colSums[i] * rowSums[i];
      sumPkSq += colSums[i] * colSums[i];
      sumTkSq += rowSums[i] * rowSums[i];
    }

    const numerator = c * s - sumPkTk;
    const denominator = Math.sqrt((s * s - sumPkSq) * (s * s - sumTkSq));
    if (denominator === 0) {
      return 0;
    }
    return numerator / denominator;
  }

  const { tp, fp, fn, tn } = confusionCounts(yTrue, yPred, positiveLabel);
  const numerator = tp * tn - fp * fn;
  const denominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  if (denominator === 0) {
    return 0;
  }
  return numerator / denominator;
}

export function brierScoreLoss(yTrue: number[], yPredProb: number[] | Matrix): number {
  if (Array.isArray(yPredProb[0])) {
    const probabilities = yPredProb as Matrix;
    if (yTrue.length === 0 || probabilities.length === 0 || yTrue.length !== probabilities.length) {
      throw new Error("yTrue and yPredProb must be non-empty and equal-length.");
    }
    const labels = uniqueSorted(yTrue);
    const classToIndex = labelToIndex(labels);

    let total = 0;
    for (let i = 0; i < yTrue.length; i += 1) {
      if (probabilities[i].length < labels.length) {
        throw new Error(
          `Each probability row must have at least ${labels.length} classes. Got ${probabilities[i].length}.`,
        );
      }
      const yi = classToIndex.get(yTrue[i]);
      if (yi === undefined) {
        throw new Error(`Unknown label '${yTrue[i]}' in yTrue.`);
      }
      for (let classIndex = 0; classIndex < labels.length; classIndex += 1) {
        const target = classIndex === yi ? 1 : 0;
        const diff = probabilities[i][classIndex] - target;
        total += diff * diff;
      }
    }
    return total / yTrue.length;
  }

  const probabilities = yPredProb as number[];
  validateInputs(yTrue, probabilities);
  validateBinaryTargets(yTrue);
  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    const diff = probabilities[i] - yTrue[i];
    total += diff * diff;
  }
  return total / yTrue.length;
}
