function validateInputs(yTrue: number[], yPred: number[]): void {
  if (yTrue.length === 0 || yPred.length === 0) {
    throw new Error("yTrue and yPred must be non-empty.");
  }

  if (yTrue.length !== yPred.length) {
    throw new Error(`Length mismatch: yTrue=${yTrue.length}, yPred=${yPred.length}.`);
  }
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
  validateInputs(yTrue, yPred);

  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;

  for (let i = 0; i < yTrue.length; i += 1) {
    const truthPositive = yTrue[i] === positiveLabel;
    const predPositive = yPred[i] === positiveLabel;

    if (truthPositive && predPositive) {
      tp += 1;
    } else if (!truthPositive && predPositive) {
      fp += 1;
    } else if (truthPositive && !predPositive) {
      fn += 1;
    } else {
      tn += 1;
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

export function accuracyScore(yTrue: number[], yPred: number[]): number {
  validateInputs(yTrue, yPred);
  let correct = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    if (yTrue[i] === yPred[i]) {
      correct += 1;
    }
  }
  return correct / yTrue.length;
}

export function precisionScore(
  yTrue: number[],
  yPred: number[],
  positiveLabel = 1,
): number {
  const { tp, fp } = confusionCounts(yTrue, yPred, positiveLabel);
  const denominator = tp + fp;
  if (denominator === 0) {
    return 0;
  }
  return tp / denominator;
}

export function recallScore(yTrue: number[], yPred: number[], positiveLabel = 1): number {
  const { tp, fn } = confusionCounts(yTrue, yPred, positiveLabel);
  const denominator = tp + fn;
  if (denominator === 0) {
    return 0;
  }
  return tp / denominator;
}

export function f1Score(yTrue: number[], yPred: number[], positiveLabel = 1): number {
  const precision = precisionScore(yTrue, yPred, positiveLabel);
  const recall = recallScore(yTrue, yPred, positiveLabel);
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
      : Array.from(new Set([...yTrue, ...yPred])).sort((a, b) => a - b);
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

export function logLoss(yTrue: number[], yPredProb: number[], eps = 1e-15): number {
  validateInputs(yTrue, yPredProb);
  validateBinaryTargets(yTrue);
  if (!Number.isFinite(eps) || eps <= 0 || eps >= 0.5) {
    throw new Error(`eps must be finite and in (0, 0.5). Got ${eps}.`);
  }

  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    const p1 = clampProbability(yPredProb[i], eps);
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

export function balancedAccuracyScore(yTrue: number[], yPred: number[]): number {
  const report = classificationReport(yTrue, yPred);
  return report.macroAvg.recall;
}

export function matthewsCorrcoef(
  yTrue: number[],
  yPred: number[],
  positiveLabel = 1,
): number {
  const { tp, fp, fn, tn } = confusionCounts(yTrue, yPred, positiveLabel);
  const numerator = tp * tn - fp * fn;
  const denominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  if (denominator === 0) {
    return 0;
  }
  return numerator / denominator;
}

export function brierScoreLoss(yTrue: number[], yPredProb: number[]): number {
  validateInputs(yTrue, yPredProb);
  validateBinaryTargets(yTrue);
  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    const diff = yPredProb[i] - yTrue[i];
    total += diff * diff;
  }
  return total / yTrue.length;
}
