import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

function euclideanDistance(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function validateLabels(labels: Vector, nSamples: number): void {
  if (!Array.isArray(labels) || labels.length !== nSamples) {
    throw new Error(`labels length must equal ${nSamples}.`);
  }
}

function uniqueLabels(labels: Vector): number[] {
  return Array.from(new Set(labels)).sort((a, b) => a - b);
}

export function silhouetteScore(X: Matrix, labels: Vector): number {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  validateLabels(labels, X.length);

  const unique = uniqueLabels(labels);
  if (unique.length < 2 || unique.length >= X.length) {
    throw new Error("silhouetteScore requires between 2 and nSamples - 1 clusters.");
  }

  const clusterMembers = new Map<number, number[]>();
  for (let i = 0; i < labels.length; i += 1) {
    const label = labels[i];
    if (!clusterMembers.has(label)) {
      clusterMembers.set(label, []);
    }
    clusterMembers.get(label)!.push(i);
  }

  let total = 0;
  for (let i = 0; i < X.length; i += 1) {
    const ownLabel = labels[i];
    const ownMembers = clusterMembers.get(ownLabel)!;
    let a = 0;
    if (ownMembers.length > 1) {
      for (let j = 0; j < ownMembers.length; j += 1) {
        if (ownMembers[j] === i) {
          continue;
        }
        a += euclideanDistance(X[i], X[ownMembers[j]]);
      }
      a /= ownMembers.length - 1;
    }

    let b = Number.POSITIVE_INFINITY;
    for (const [label, members] of clusterMembers.entries()) {
      if (label === ownLabel || members.length === 0) {
        continue;
      }
      let dist = 0;
      for (let j = 0; j < members.length; j += 1) {
        dist += euclideanDistance(X[i], X[members[j]]);
      }
      dist /= members.length;
      if (dist < b) {
        b = dist;
      }
    }
    const denom = Math.max(a, b);
    const s = denom <= 1e-12 ? 0 : (b - a) / denom;
    total += s;
  }
  return total / X.length;
}

export function calinskiHarabaszScore(X: Matrix, labels: Vector): number {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  validateLabels(labels, X.length);
  const unique = uniqueLabels(labels);
  const k = unique.length;
  const nSamples = X.length;
  if (k < 2 || k >= nSamples) {
    throw new Error("calinskiHarabaszScore requires between 2 and nSamples - 1 clusters.");
  }

  const nFeatures = X[0].length;
  const overallMean = new Array<number>(nFeatures).fill(0);
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < nFeatures; j += 1) {
      overallMean[j] += X[i][j];
    }
  }
  for (let j = 0; j < nFeatures; j += 1) {
    overallMean[j] /= nSamples;
  }

  const clusterSums = new Map<number, Vector>();
  const clusterCounts = new Map<number, number>();
  for (let i = 0; i < labels.length; i += 1) {
    const label = labels[i];
    if (!clusterSums.has(label)) {
      clusterSums.set(label, new Array<number>(nFeatures).fill(0));
      clusterCounts.set(label, 0);
    }
    const sum = clusterSums.get(label)!;
    for (let j = 0; j < nFeatures; j += 1) {
      sum[j] += X[i][j];
    }
    clusterCounts.set(label, clusterCounts.get(label)! + 1);
  }

  const clusterMeans = new Map<number, Vector>();
  for (const label of unique) {
    const sum = clusterSums.get(label)!;
    const count = clusterCounts.get(label)!;
    clusterMeans.set(label, sum.map((value) => value / count));
  }

  let between = 0;
  let within = 0;
  for (const label of unique) {
    const mean = clusterMeans.get(label)!;
    const count = clusterCounts.get(label)!;
    let centerDist = 0;
    for (let j = 0; j < nFeatures; j += 1) {
      const d = mean[j] - overallMean[j];
      centerDist += d * d;
    }
    between += count * centerDist;
  }

  for (let i = 0; i < X.length; i += 1) {
    const mean = clusterMeans.get(labels[i])!;
    let dist = 0;
    for (let j = 0; j < nFeatures; j += 1) {
      const d = X[i][j] - mean[j];
      dist += d * d;
    }
    within += dist;
  }
  if (within <= 1e-12) {
    return Number.POSITIVE_INFINITY;
  }
  return (between / (k - 1)) / (within / (nSamples - k));
}

export function daviesBouldinScore(X: Matrix, labels: Vector): number {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);
  validateLabels(labels, X.length);
  const unique = uniqueLabels(labels);
  if (unique.length < 2) {
    throw new Error("daviesBouldinScore requires at least two clusters.");
  }

  const nFeatures = X[0].length;
  const centers = new Map<number, Vector>();
  const counts = new Map<number, number>();
  for (const label of unique) {
    centers.set(label, new Array<number>(nFeatures).fill(0));
    counts.set(label, 0);
  }
  for (let i = 0; i < X.length; i += 1) {
    const center = centers.get(labels[i])!;
    for (let j = 0; j < nFeatures; j += 1) {
      center[j] += X[i][j];
    }
    counts.set(labels[i], counts.get(labels[i])! + 1);
  }
  for (const label of unique) {
    const center = centers.get(label)!;
    const count = counts.get(label)!;
    for (let j = 0; j < nFeatures; j += 1) {
      center[j] /= Math.max(1, count);
    }
  }

  const scatter = new Map<number, number>();
  for (const label of unique) {
    let total = 0;
    let count = 0;
    const center = centers.get(label)!;
    for (let i = 0; i < X.length; i += 1) {
      if (labels[i] !== label) {
        continue;
      }
      total += euclideanDistance(X[i], center);
      count += 1;
    }
    scatter.set(label, count === 0 ? 0 : total / count);
  }

  let db = 0;
  for (const iLabel of unique) {
    let maxRatio = Number.NEGATIVE_INFINITY;
    for (const jLabel of unique) {
      if (iLabel === jLabel) {
        continue;
      }
      const numerator = (scatter.get(iLabel) ?? 0) + (scatter.get(jLabel) ?? 0);
      const denominator = euclideanDistance(centers.get(iLabel)!, centers.get(jLabel)!);
      const ratio = denominator <= 1e-12 ? Number.POSITIVE_INFINITY : numerator / denominator;
      if (ratio > maxRatio) {
        maxRatio = ratio;
      }
    }
    db += maxRatio;
  }
  return db / unique.length;
}

export function adjustedRandScore(labelsTrue: Vector, labelsPred: Vector): number {
  if (labelsTrue.length === 0 || labelsPred.length === 0 || labelsTrue.length !== labelsPred.length) {
    throw new Error("labelsTrue and labelsPred must be non-empty and equal-length.");
  }
  const n = labelsTrue.length;
  const trueLabels = uniqueLabels(labelsTrue);
  const predLabels = uniqueLabels(labelsPred);

  const trueIndex = new Map<number, number>();
  const predIndex = new Map<number, number>();
  for (let i = 0; i < trueLabels.length; i += 1) {
    trueIndex.set(trueLabels[i], i);
  }
  for (let i = 0; i < predLabels.length; i += 1) {
    predIndex.set(predLabels[i], i);
  }

  const contingency: Matrix = Array.from({ length: trueLabels.length }, () =>
    new Array<number>(predLabels.length).fill(0),
  );
  for (let i = 0; i < n; i += 1) {
    contingency[trueIndex.get(labelsTrue[i])!][predIndex.get(labelsPred[i])!] += 1;
  }

  const comb2 = (value: number): number => (value * (value - 1)) / 2;
  let sumComb = 0;
  const trueSums = new Array<number>(trueLabels.length).fill(0);
  const predSums = new Array<number>(predLabels.length).fill(0);
  for (let i = 0; i < contingency.length; i += 1) {
    for (let j = 0; j < contingency[i].length; j += 1) {
      const v = contingency[i][j];
      sumComb += comb2(v);
      trueSums[i] += v;
      predSums[j] += v;
    }
  }
  let sumTrue = 0;
  let sumPred = 0;
  for (let i = 0; i < trueSums.length; i += 1) {
    sumTrue += comb2(trueSums[i]);
  }
  for (let i = 0; i < predSums.length; i += 1) {
    sumPred += comb2(predSums[i]);
  }
  const totalComb = comb2(n);
  if (totalComb === 0) {
    return 0;
  }
  const expected = (sumTrue * sumPred) / totalComb;
  const maxIndex = 0.5 * (sumTrue + sumPred);
  const denominator = maxIndex - expected;
  if (Math.abs(denominator) <= 1e-12) {
    return 0;
  }
  return (sumComb - expected) / denominator;
}
