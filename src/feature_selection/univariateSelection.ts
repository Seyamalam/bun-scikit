import type { Matrix, Vector } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertFiniteVector,
  assertNonEmptyMatrix,
  assertVectorLength,
} from "../utils/validation";

export type UnivariateScoreResult = [scores: Vector, pValues: Vector];
export type UnivariateScoreFunc = (X: Matrix, y: Vector) => UnivariateScoreResult;

const LOG_SQRT_2PI = 0.9189385332046727;
const LANCZOS_COEFFICIENTS = [
  676.5203681218851,
  -1259.1392167224028,
  771.3234287776531,
  -176.6150291621406,
  12.507343278686905,
  -0.13857109526572012,
  9.984369578019572e-6,
  1.5056327351493116e-7,
];

function logGamma(z: number): number {
  if (z < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  const shifted = z - 1;
  let x = 0.9999999999998099;
  for (let i = 0; i < LANCZOS_COEFFICIENTS.length; i += 1) {
    x += LANCZOS_COEFFICIENTS[i] / (shifted + i + 1);
  }
  const t = shifted + LANCZOS_COEFFICIENTS.length - 0.5;
  return LOG_SQRT_2PI + (shifted + 0.5) * Math.log(t) - t + Math.log(x);
}

function regularizedGammaPSeries(a: number, x: number): number {
  let term = 1 / a;
  let sum = term;
  for (let n = 1; n < 5000; n += 1) {
    term *= x / (a + n);
    sum += term;
    if (Math.abs(term) < Math.abs(sum) * 1e-14) {
      break;
    }
  }
  return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
}

function regularizedGammaQCF(a: number, x: number): number {
  let b = x + 1 - a;
  let c = 1 / 1e-30;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 5000; i += 1) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < 1e-30) {
      d = 1e-30;
    }
    c = b + an / c;
    if (Math.abs(c) < 1e-30) {
      c = 1e-30;
    }
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-14) {
      break;
    }
  }
  return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
}

function regularizedGammaQ(a: number, x: number): number {
  if (x <= 0) {
    return 1;
  }
  if (x < a + 1) {
    return 1 - regularizedGammaPSeries(a, x);
  }
  return regularizedGammaQCF(a, x);
}

function regularizedBeta(x: number, a: number, b: number): number {
  if (x <= 0) {
    return 0;
  }
  if (x >= 1) {
    return 1;
  }

  function betaContinuedFraction(value: number, aa: number, bb: number): number {
    let qab = aa + bb;
    let qap = aa + 1;
    let qam = aa - 1;
    let c = 1;
    let d = 1 - (qab * value) / qap;
    if (Math.abs(d) < 1e-30) {
      d = 1e-30;
    }
    d = 1 / d;
    let h = d;
    for (let m = 1; m < 5000; m += 1) {
      const m2 = 2 * m;
      let numerator = (m * (bb - m) * value) / ((qam + m2) * (aa + m2));
      d = 1 + numerator * d;
      if (Math.abs(d) < 1e-30) {
        d = 1e-30;
      }
      c = 1 + numerator / c;
      if (Math.abs(c) < 1e-30) {
        c = 1e-30;
      }
      d = 1 / d;
      h *= d * c;

      numerator = (-(aa + m) * (qab + m) * value) / ((aa + m2) * (qap + m2));
      d = 1 + numerator * d;
      if (Math.abs(d) < 1e-30) {
        d = 1e-30;
      }
      c = 1 + numerator / c;
      if (Math.abs(c) < 1e-30) {
        c = 1e-30;
      }
      d = 1 / d;
      const delta = d * c;
      h *= delta;
      if (Math.abs(delta - 1) < 1e-14) {
        break;
      }
    }
    return h;
  }

  const logBeta =
    logGamma(a + b) -
    logGamma(a) -
    logGamma(b) +
    a * Math.log(x) +
    b * Math.log(1 - x);
  const factor = Math.exp(logBeta);

  if (x < (a + 1) / (a + b + 2)) {
    return (factor * betaContinuedFraction(x, a, b)) / a;
  }
  return 1 - (factor * betaContinuedFraction(1 - x, b, a)) / b;
}

function chiSquareSurvival(value: number, dof: number): number {
  if (!Number.isFinite(value) || value < 0) {
    return 1;
  }
  if (dof <= 0) {
    return 1;
  }
  return regularizedGammaQ(dof / 2, value / 2);
}

function fDistributionSurvival(value: number, dfn: number, dfd: number): number {
  if (!Number.isFinite(value) || value < 0 || dfn <= 0 || dfd <= 0) {
    return 1;
  }
  const x = dfd / (dfd + dfn * value);
  return regularizedBeta(x, dfd / 2, dfn / 2);
}

function validateXy(X: Matrix, y: Vector): void {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  assertVectorLength(y, X.length);
  assertFiniteVector(y);
}

function uniqueClasses(y: Vector): number[] {
  return Array.from(new Set(y)).sort((a, b) => a - b);
}

function mean(values: Vector): number {
  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    total += values[i];
  }
  return total / values.length;
}

function rankFeatureIndices(scores: Vector, take: number): number[] {
  const order = Array.from({ length: scores.length }, (_, index) => index);
  order.sort((a, b) => {
    const scoreA = Number.isFinite(scores[a]) ? scores[a] : Number.NEGATIVE_INFINITY;
    const scoreB = Number.isFinite(scores[b]) ? scores[b] : Number.NEGATIVE_INFINITY;
    if (scoreB !== scoreA) {
      return scoreB - scoreA;
    }
    return a - b;
  });
  const k = Math.max(0, Math.min(take, scores.length));
  return order.slice(0, k).sort((a, b) => a - b);
}

export function chi2(X: Matrix, y: Vector): UnivariateScoreResult {
  validateXy(X, y);
  const classes = uniqueClasses(y);
  if (classes.length < 2) {
    throw new Error("chi2 requires at least two classes.");
  }

  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      if (X[i][j] < 0) {
        throw new Error("chi2 requires non-negative feature values.");
      }
    }
  }

  const classToIndex = new Map<number, number>();
  for (let i = 0; i < classes.length; i += 1) {
    classToIndex.set(classes[i], i);
  }

  const nSamples = X.length;
  const nFeatures = X[0].length;
  const classCounts = new Array<number>(classes.length).fill(0);
  const classFeatureSums: number[][] = Array.from({ length: classes.length }, () =>
    new Array<number>(nFeatures).fill(0),
  );
  const featureTotals = new Array<number>(nFeatures).fill(0);

  for (let i = 0; i < nSamples; i += 1) {
    const classIndex = classToIndex.get(y[i]);
    if (classIndex === undefined) {
      throw new Error(`Internal chi2 class mapping error for label ${y[i]}.`);
    }
    classCounts[classIndex] += 1;
    for (let j = 0; j < nFeatures; j += 1) {
      classFeatureSums[classIndex][j] += X[i][j];
      featureTotals[j] += X[i][j];
    }
  }

  const classProbabilities = classCounts.map((count) => count / nSamples);
  const scores = new Array<number>(nFeatures).fill(0);
  const pValues = new Array<number>(nFeatures).fill(1);
  const dof = classes.length - 1;

  for (let feature = 0; feature < nFeatures; feature += 1) {
    const total = featureTotals[feature];
    if (total <= 0) {
      scores[feature] = 0;
      pValues[feature] = 1;
      continue;
    }

    let stat = 0;
    for (let classIndex = 0; classIndex < classes.length; classIndex += 1) {
      const observed = classFeatureSums[classIndex][feature];
      const expected = classProbabilities[classIndex] * total;
      if (expected > 0) {
        const diff = observed - expected;
        stat += (diff * diff) / expected;
      }
    }
    scores[feature] = stat;
    pValues[feature] = chiSquareSurvival(stat, dof);
  }

  return [scores, pValues];
}

export function f_classif(X: Matrix, y: Vector): UnivariateScoreResult {
  validateXy(X, y);
  const classes = uniqueClasses(y);
  if (classes.length < 2) {
    throw new Error("f_classif requires at least two classes.");
  }

  const classToIndex = new Map<number, number>();
  for (let i = 0; i < classes.length; i += 1) {
    classToIndex.set(classes[i], i);
  }

  const nSamples = X.length;
  const nFeatures = X[0].length;
  const nClasses = classes.length;
  const classCounts = new Array<number>(nClasses).fill(0);
  const classMeans: number[][] = Array.from({ length: nClasses }, () =>
    new Array<number>(nFeatures).fill(0),
  );
  const overallMeans = new Array<number>(nFeatures).fill(0);

  for (let i = 0; i < nSamples; i += 1) {
    const classIndex = classToIndex.get(y[i]);
    if (classIndex === undefined) {
      throw new Error(`Internal f_classif class mapping error for label ${y[i]}.`);
    }
    classCounts[classIndex] += 1;
    for (let feature = 0; feature < nFeatures; feature += 1) {
      const value = X[i][feature];
      classMeans[classIndex][feature] += value;
      overallMeans[feature] += value;
    }
  }

  for (let feature = 0; feature < nFeatures; feature += 1) {
    overallMeans[feature] /= nSamples;
  }
  for (let classIndex = 0; classIndex < nClasses; classIndex += 1) {
    if (classCounts[classIndex] === 0) {
      continue;
    }
    for (let feature = 0; feature < nFeatures; feature += 1) {
      classMeans[classIndex][feature] /= classCounts[classIndex];
    }
  }

  const ssWithin = new Array<number>(nFeatures).fill(0);
  const ssBetween = new Array<number>(nFeatures).fill(0);
  for (let classIndex = 0; classIndex < nClasses; classIndex += 1) {
    const count = classCounts[classIndex];
    if (count === 0) {
      continue;
    }
    for (let feature = 0; feature < nFeatures; feature += 1) {
      const betweenDiff = classMeans[classIndex][feature] - overallMeans[feature];
      ssBetween[feature] += count * betweenDiff * betweenDiff;
    }
  }
  for (let i = 0; i < nSamples; i += 1) {
    const classIndex = classToIndex.get(y[i])!;
    for (let feature = 0; feature < nFeatures; feature += 1) {
      const diff = X[i][feature] - classMeans[classIndex][feature];
      ssWithin[feature] += diff * diff;
    }
  }

  const dfn = nClasses - 1;
  const dfd = nSamples - nClasses;
  const scores = new Array<number>(nFeatures).fill(0);
  const pValues = new Array<number>(nFeatures).fill(1);
  for (let feature = 0; feature < nFeatures; feature += 1) {
    const msBetween = ssBetween[feature] / Math.max(1, dfn);
    const msWithin = ssWithin[feature] / Math.max(1, dfd);
    let fValue = 0;
    if (msWithin === 0) {
      fValue = msBetween > 0 ? Number.POSITIVE_INFINITY : 0;
    } else {
      fValue = msBetween / msWithin;
    }
    scores[feature] = fValue;
    pValues[feature] = Number.isFinite(fValue)
      ? fDistributionSurvival(fValue, dfn, dfd)
      : 0;
  }

  return [scores, pValues];
}

export interface FRegressionOptions {
  center?: boolean;
}

export function f_regression(
  X: Matrix,
  y: Vector,
  options: FRegressionOptions = {},
): UnivariateScoreResult {
  validateXy(X, y);
  const center = options.center ?? true;
  const nSamples = X.length;
  const nFeatures = X[0].length;
  const yMean = center ? mean(y) : 0;

  let yVar = 0;
  for (let i = 0; i < nSamples; i += 1) {
    const dy = y[i] - yMean;
    yVar += dy * dy;
  }
  if (yVar === 0) {
    return [new Array<number>(nFeatures).fill(0), new Array<number>(nFeatures).fill(1)];
  }

  const scores = new Array<number>(nFeatures).fill(0);
  const pValues = new Array<number>(nFeatures).fill(1);
  const dfd = nSamples - 2;
  for (let feature = 0; feature < nFeatures; feature += 1) {
    let xMean = 0;
    if (center) {
      for (let i = 0; i < nSamples; i += 1) {
        xMean += X[i][feature];
      }
      xMean /= nSamples;
    }

    let covariance = 0;
    let xVar = 0;
    for (let i = 0; i < nSamples; i += 1) {
      const dx = X[i][feature] - xMean;
      const dy = y[i] - yMean;
      covariance += dx * dy;
      xVar += dx * dx;
    }

    if (xVar === 0) {
      scores[feature] = 0;
      pValues[feature] = 1;
      continue;
    }

    const denominator = Math.sqrt(xVar * yVar);
    const correlation = denominator === 0 ? 0 : Math.max(-1, Math.min(1, covariance / denominator));
    const r2 = correlation * correlation;
    const fValue = r2 >= 1 ? Number.POSITIVE_INFINITY : (r2 / Math.max(1e-16, 1 - r2)) * dfd;
    scores[feature] = fValue;
    pValues[feature] = Number.isFinite(fValue) ? fDistributionSurvival(fValue, 1, dfd) : 0;
  }
  return [scores, pValues];
}

export interface SelectKBestOptions {
  scoreFunc?: UnivariateScoreFunc;
  k?: number | "all";
}

export class SelectKBest {
  scores_: Vector | null = null;
  pvalues_: Vector | null = null;
  nFeaturesIn_: number | null = null;
  selectedFeatureIndices_: number[] | null = null;

  private readonly scoreFunc: UnivariateScoreFunc;
  private k: number | "all";

  constructor(options: SelectKBestOptions = {}) {
    this.scoreFunc = options.scoreFunc ?? f_classif;
    this.k = options.k ?? 10;
  }

  fit(X: Matrix, y: Vector): this {
    validateXy(X, y);
    const nFeatures = X[0].length;

    if (this.k !== "all") {
      if (!Number.isInteger(this.k) || this.k < 0) {
        throw new Error(`k must be a non-negative integer or 'all'. Got ${this.k}.`);
      }
    }

    const [scores, pValues] = this.scoreFunc(X, y);
    if (scores.length !== nFeatures || pValues.length !== nFeatures) {
      throw new Error("scoreFunc must return [scores, pValues] with one value per feature.");
    }

    const selectedCount = this.k === "all" ? nFeatures : Math.min(this.k, nFeatures);
    this.selectedFeatureIndices_ = rankFeatureIndices(scores, selectedCount);
    this.scores_ = scores.slice();
    this.pvalues_ = pValues.slice();
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.selectedFeatureIndices_ || this.nFeaturesIn_ === null) {
      throw new Error("SelectKBest has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return X.map((row) => this.selectedFeatureIndices_!.map((featureIndex) => row[featureIndex]));
  }

  fitTransform(X: Matrix, y: Vector): Matrix {
    return this.fit(X, y).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.selectedFeatureIndices_ || this.nFeaturesIn_ === null) {
      throw new Error("SelectKBest has not been fitted.");
    }
    if (indices) {
      return this.selectedFeatureIndices_.slice();
    }
    const mask = new Array<boolean>(this.nFeaturesIn_).fill(false);
    for (let i = 0; i < this.selectedFeatureIndices_.length; i += 1) {
      mask[this.selectedFeatureIndices_[i]] = true;
    }
    return mask;
  }
}

export interface SelectPercentileOptions {
  scoreFunc?: UnivariateScoreFunc;
  percentile?: number;
}

export class SelectPercentile {
  scores_: Vector | null = null;
  pvalues_: Vector | null = null;
  nFeaturesIn_: number | null = null;
  selectedFeatureIndices_: number[] | null = null;

  private readonly scoreFunc: UnivariateScoreFunc;
  private percentile: number;

  constructor(options: SelectPercentileOptions = {}) {
    this.scoreFunc = options.scoreFunc ?? f_classif;
    this.percentile = options.percentile ?? 10;
    if (!Number.isFinite(this.percentile) || this.percentile < 0 || this.percentile > 100) {
      throw new Error(`percentile must be within [0, 100]. Got ${this.percentile}.`);
    }
  }

  fit(X: Matrix, y: Vector): this {
    validateXy(X, y);
    const nFeatures = X[0].length;
    const [scores, pValues] = this.scoreFunc(X, y);
    if (scores.length !== nFeatures || pValues.length !== nFeatures) {
      throw new Error("scoreFunc must return [scores, pValues] with one value per feature.");
    }

    const selectedCount = this.percentile === 100
      ? nFeatures
      : Math.ceil((this.percentile / 100) * nFeatures);
    this.selectedFeatureIndices_ = rankFeatureIndices(scores, selectedCount);
    this.scores_ = scores.slice();
    this.pvalues_ = pValues.slice();
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  transform(X: Matrix): Matrix {
    if (!this.selectedFeatureIndices_ || this.nFeaturesIn_ === null) {
      throw new Error("SelectPercentile has not been fitted.");
    }
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (X[0].length !== this.nFeaturesIn_) {
      throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
    }
    return X.map((row) => this.selectedFeatureIndices_!.map((featureIndex) => row[featureIndex]));
  }

  fitTransform(X: Matrix, y: Vector): Matrix {
    return this.fit(X, y).transform(X);
  }

  getSupport(indices = false): boolean[] | number[] {
    if (!this.selectedFeatureIndices_ || this.nFeaturesIn_ === null) {
      throw new Error("SelectPercentile has not been fitted.");
    }
    if (indices) {
      return this.selectedFeatureIndices_.slice();
    }
    const mask = new Array<boolean>(this.nFeaturesIn_).fill(false);
    for (let i = 0; i < this.selectedFeatureIndices_.length; i += 1) {
      mask[this.selectedFeatureIndices_[i]] = true;
    }
    return mask;
  }
}
