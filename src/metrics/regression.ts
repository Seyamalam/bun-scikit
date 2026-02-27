import type { Matrix, Vector } from "../types";
import { mean } from "../utils/linalg";

export type MultioutputMode = "uniform_average" | "raw_values" | "variance_weighted";

export interface RegressionMetricOptions {
  sampleWeight?: Vector;
  multioutput?: MultioutputMode;
}

type RegressionTargets = Vector | Matrix;
type RegressionMetricResult = number | Vector;

interface NormalizedTargets {
  yTrue: Matrix;
  yPred: Matrix;
}

function isMatrixLike(value: RegressionTargets): value is Matrix {
  return Array.isArray(value[0]);
}

function validateAndNormalize(yTrue: RegressionTargets, yPred: RegressionTargets): NormalizedTargets {
  const yTrueIsMatrix = isMatrixLike(yTrue);
  const yPredIsMatrix = isMatrixLike(yPred);
  if (yTrueIsMatrix !== yPredIsMatrix) {
    throw new Error("yTrue and yPred must both be vectors or both be matrices.");
  }

  if (yTrue.length === 0 || yPred.length === 0) {
    throw new Error("yTrue and yPred must be non-empty.");
  }
  if (yTrue.length !== yPred.length) {
    throw new Error(`Length mismatch: yTrue=${yTrue.length}, yPred=${yPred.length}.`);
  }

  if (!yTrueIsMatrix) {
    const trueVector = yTrue as Vector;
    const predVector = yPred as Vector;
    const trueMatrix = trueVector.map((value) => [value]);
    const predMatrix = predVector.map((value) => [value]);
    return { yTrue: trueMatrix, yPred: predMatrix };
  }

  const yTrueMatrix = yTrue as Matrix;
  const yPredMatrix = yPred as Matrix;
  const featureCount = yTrueMatrix[0].length;
  if (featureCount === 0) {
    throw new Error("yTrue and yPred matrices must include at least one output.");
  }

  for (let i = 0; i < yTrueMatrix.length; i += 1) {
    if (yTrueMatrix[i].length !== featureCount || yPredMatrix[i].length !== featureCount) {
      throw new Error("All rows in yTrue and yPred must have consistent output dimensionality.");
    }
  }
  return { yTrue: yTrueMatrix, yPred: yPredMatrix };
}

function resolveWeights(sampleWeight: Vector | undefined, nSamples: number): Vector {
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

function weightedMean(values: Vector, weights: Vector): number {
  let total = 0;
  let weightSum = 0;
  for (let i = 0; i < values.length; i += 1) {
    total += values[i] * weights[i];
    weightSum += weights[i];
  }
  if (weightSum === 0) {
    return 0;
  }
  return total / weightSum;
}

function combineMultioutput(
  perOutput: Vector,
  yTrue: Matrix,
  weights: Vector,
  mode: MultioutputMode,
): RegressionMetricResult {
  if (mode === "raw_values") {
    return perOutput;
  }
  if (mode === "uniform_average") {
    return mean(perOutput);
  }

  const variances = new Array<number>(perOutput.length).fill(0);
  for (let outputIndex = 0; outputIndex < perOutput.length; outputIndex += 1) {
    const values = yTrue.map((row) => row[outputIndex]);
    const m = weightedMean(values, weights);
    let varianceTotal = 0;
    let weightSum = 0;
    for (let i = 0; i < values.length; i += 1) {
      const centered = values[i] - m;
      varianceTotal += weights[i] * centered * centered;
      weightSum += weights[i];
    }
    variances[outputIndex] = weightSum === 0 ? 0 : varianceTotal / weightSum;
  }

  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < perOutput.length; i += 1) {
    numerator += perOutput[i] * variances[i];
    denominator += variances[i];
  }
  if (denominator === 0) {
    return mean(perOutput);
  }
  return numerator / denominator;
}

export function meanSquaredError(
  yTrue: Vector,
  yPred: Vector,
  options?: RegressionMetricOptions,
): number;
export function meanSquaredError(
  yTrue: Matrix,
  yPred: Matrix,
  options?: RegressionMetricOptions,
): RegressionMetricResult;
export function meanSquaredError(
  yTrue: RegressionTargets,
  yPred: RegressionTargets,
  options: RegressionMetricOptions = {},
): RegressionMetricResult {
  const normalized = validateAndNormalize(yTrue, yPred);
  const weights = resolveWeights(options.sampleWeight, normalized.yTrue.length);
  const outputCount = normalized.yTrue[0].length;
  const perOutput = new Array<number>(outputCount).fill(0);
  let weightSum = 0;
  for (let i = 0; i < weights.length; i += 1) {
    weightSum += weights[i];
  }
  if (weightSum === 0) {
    return options.multioutput === "raw_values" ? perOutput : 0;
  }

  for (let i = 0; i < normalized.yTrue.length; i += 1) {
    for (let j = 0; j < outputCount; j += 1) {
      const diff = normalized.yTrue[i][j] - normalized.yPred[i][j];
      perOutput[j] += weights[i] * diff * diff;
    }
  }
  for (let j = 0; j < outputCount; j += 1) {
    perOutput[j] /= weightSum;
  }

  return combineMultioutput(
    perOutput,
    normalized.yTrue,
    weights,
    options.multioutput ?? "uniform_average",
  );
}

export function meanAbsoluteError(
  yTrue: Vector,
  yPred: Vector,
  options?: RegressionMetricOptions,
): number;
export function meanAbsoluteError(
  yTrue: Matrix,
  yPred: Matrix,
  options?: RegressionMetricOptions,
): RegressionMetricResult;
export function meanAbsoluteError(
  yTrue: RegressionTargets,
  yPred: RegressionTargets,
  options: RegressionMetricOptions = {},
): RegressionMetricResult {
  const normalized = validateAndNormalize(yTrue, yPred);
  const weights = resolveWeights(options.sampleWeight, normalized.yTrue.length);
  const outputCount = normalized.yTrue[0].length;
  const perOutput = new Array<number>(outputCount).fill(0);
  let weightSum = 0;
  for (let i = 0; i < weights.length; i += 1) {
    weightSum += weights[i];
  }
  if (weightSum === 0) {
    return options.multioutput === "raw_values" ? perOutput : 0;
  }

  for (let i = 0; i < normalized.yTrue.length; i += 1) {
    for (let j = 0; j < outputCount; j += 1) {
      perOutput[j] += weights[i] * Math.abs(normalized.yTrue[i][j] - normalized.yPred[i][j]);
    }
  }
  for (let j = 0; j < outputCount; j += 1) {
    perOutput[j] /= weightSum;
  }

  return combineMultioutput(
    perOutput,
    normalized.yTrue,
    weights,
    options.multioutput ?? "uniform_average",
  );
}

export function r2Score(yTrue: Vector, yPred: Vector, options?: RegressionMetricOptions): number;
export function r2Score(
  yTrue: Matrix,
  yPred: Matrix,
  options?: RegressionMetricOptions,
): RegressionMetricResult;
export function r2Score(
  yTrue: RegressionTargets,
  yPred: RegressionTargets,
  options: RegressionMetricOptions = {},
): RegressionMetricResult {
  const normalized = validateAndNormalize(yTrue, yPred);
  const weights = resolveWeights(options.sampleWeight, normalized.yTrue.length);
  const outputCount = normalized.yTrue[0].length;
  const perOutput = new Array<number>(outputCount).fill(0);

  for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
    const trueColumn = normalized.yTrue.map((row) => row[outputIndex]);
    const predColumn = normalized.yPred.map((row) => row[outputIndex]);
    const yMean = weightedMean(trueColumn, weights);

    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < trueColumn.length; i += 1) {
      const residual = trueColumn[i] - predColumn[i];
      const centered = trueColumn[i] - yMean;
      ssRes += weights[i] * residual * residual;
      ssTot += weights[i] * centered * centered;
    }
    if (ssTot === 0) {
      perOutput[outputIndex] = ssRes === 0 ? 1 : 0;
    } else {
      perOutput[outputIndex] = 1 - ssRes / ssTot;
    }
  }

  return combineMultioutput(
    perOutput,
    normalized.yTrue,
    weights,
    options.multioutput ?? "uniform_average",
  );
}

export function meanAbsolutePercentageError(
  yTrue: Vector,
  yPred: Vector,
  options?: RegressionMetricOptions,
): number;
export function meanAbsolutePercentageError(
  yTrue: Matrix,
  yPred: Matrix,
  options?: RegressionMetricOptions,
): RegressionMetricResult;
export function meanAbsolutePercentageError(
  yTrue: RegressionTargets,
  yPred: RegressionTargets,
  options: RegressionMetricOptions = {},
): RegressionMetricResult {
  const normalized = validateAndNormalize(yTrue, yPred);
  const weights = resolveWeights(options.sampleWeight, normalized.yTrue.length);
  const outputCount = normalized.yTrue[0].length;
  const perOutput = new Array<number>(outputCount).fill(0);
  let weightSum = 0;
  for (let i = 0; i < weights.length; i += 1) {
    weightSum += weights[i];
  }
  if (weightSum === 0) {
    return options.multioutput === "raw_values" ? perOutput : 0;
  }

  for (let i = 0; i < normalized.yTrue.length; i += 1) {
    for (let j = 0; j < outputCount; j += 1) {
      const denom = Math.max(Math.abs(normalized.yTrue[i][j]), 1e-12);
      perOutput[j] += (weights[i] * Math.abs(normalized.yTrue[i][j] - normalized.yPred[i][j])) / denom;
    }
  }
  for (let j = 0; j < outputCount; j += 1) {
    perOutput[j] /= weightSum;
  }

  return combineMultioutput(
    perOutput,
    normalized.yTrue,
    weights,
    options.multioutput ?? "uniform_average",
  );
}

export function explainedVarianceScore(
  yTrue: Vector,
  yPred: Vector,
  options?: RegressionMetricOptions,
): number;
export function explainedVarianceScore(
  yTrue: Matrix,
  yPred: Matrix,
  options?: RegressionMetricOptions,
): RegressionMetricResult;
export function explainedVarianceScore(
  yTrue: RegressionTargets,
  yPred: RegressionTargets,
  options: RegressionMetricOptions = {},
): RegressionMetricResult {
  const normalized = validateAndNormalize(yTrue, yPred);
  const weights = resolveWeights(options.sampleWeight, normalized.yTrue.length);
  const outputCount = normalized.yTrue[0].length;
  const perOutput = new Array<number>(outputCount).fill(0);

  for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
    const trueColumn = normalized.yTrue.map((row) => row[outputIndex]);
    const residuals = trueColumn.map((truth, index) => truth - normalized.yPred[index][outputIndex]);
    const trueMean = weightedMean(trueColumn, weights);
    const residualMean = weightedMean(residuals, weights);

    let varTrue = 0;
    let varResidual = 0;
    let weightSum = 0;
    for (let i = 0; i < trueColumn.length; i += 1) {
      const centeredY = trueColumn[i] - trueMean;
      const centeredR = residuals[i] - residualMean;
      varTrue += weights[i] * centeredY * centeredY;
      varResidual += weights[i] * centeredR * centeredR;
      weightSum += weights[i];
    }
    if (weightSum === 0) {
      perOutput[outputIndex] = 0;
      continue;
    }
    varTrue /= weightSum;
    varResidual /= weightSum;
    if (varTrue === 0) {
      perOutput[outputIndex] = varResidual === 0 ? 1 : 0;
    } else {
      perOutput[outputIndex] = 1 - varResidual / varTrue;
    }
  }

  return combineMultioutput(
    perOutput,
    normalized.yTrue,
    weights,
    options.multioutput ?? "uniform_average",
  );
}
