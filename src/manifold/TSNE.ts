import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { PCA } from "../decomposition/PCA";

export interface TSNEOptions {
  nComponents?: number;
  perplexity?: number;
  learningRate?: number;
  maxIter?: number;
  earlyExaggeration?: number;
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

function sampleStandardNormal(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function squaredEuclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function pairwiseSquaredDistances(X: Matrix): Matrix {
  const n = X.length;
  const out: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const dist = squaredEuclideanDistance(X[i], X[j]);
      out[i][j] = dist;
      out[j][i] = dist;
    }
  }
  return out;
}

function conditionalProbabilities(
  distRow: number[],
  selfIndex: number,
  perplexity: number,
): number[] {
  const targetEntropy = Math.log(perplexity);
  const probs = new Array<number>(distRow.length).fill(0);
  let beta = 1;
  let betaMin = Number.NEGATIVE_INFINITY;
  let betaMax = Number.POSITIVE_INFINITY;

  for (let iter = 0; iter < 64; iter += 1) {
    let sum = 0;
    let weightedDistance = 0;
    for (let j = 0; j < distRow.length; j += 1) {
      if (j === selfIndex) {
        probs[j] = 0;
        continue;
      }
      const value = Math.exp(-distRow[j] * beta);
      probs[j] = value;
      sum += value;
      weightedDistance += distRow[j] * value;
    }

    if (sum <= 1e-20) {
      const uniform = 1 / Math.max(1, distRow.length - 1);
      for (let j = 0; j < distRow.length; j += 1) {
        probs[j] = j === selfIndex ? 0 : uniform;
      }
      return probs;
    }

    const entropy = Math.log(sum) + (beta * weightedDistance) / sum;
    const diff = entropy - targetEntropy;
    if (Math.abs(diff) < 1e-5) {
      break;
    }

    if (diff > 0) {
      betaMin = beta;
      beta = Number.isFinite(betaMax) ? 0.5 * (beta + betaMax) : beta * 2;
    } else {
      betaMax = beta;
      beta = Number.isFinite(betaMin) ? 0.5 * (beta + betaMin) : beta / 2;
    }
  }

  let normalizer = 0;
  for (let j = 0; j < probs.length; j += 1) {
    normalizer += probs[j];
  }
  if (normalizer <= 0) {
    const uniform = 1 / Math.max(1, distRow.length - 1);
    for (let j = 0; j < probs.length; j += 1) {
      probs[j] = j === selfIndex ? 0 : uniform;
    }
    return probs;
  }
  for (let j = 0; j < probs.length; j += 1) {
    probs[j] /= normalizer;
  }
  return probs;
}

function jointProbabilities(distancesSquared: Matrix, perplexity: number): Matrix {
  const n = distancesSquared.length;
  const conditionals: Matrix = new Array(n);
  for (let i = 0; i < n; i += 1) {
    conditionals[i] = conditionalProbabilities(distancesSquared[i], i, perplexity);
  }

  const P: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const value = (conditionals[i][j] + conditionals[j][i]) / (2 * n);
      P[i][j] = value;
      P[j][i] = value;
      total += 2 * value;
    }
  }

  if (total > 0) {
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        P[i][j] /= total;
      }
      P[i][i] = 0;
    }
  }

  return P;
}

function initializeEmbedding(
  X: Matrix,
  nComponents: number,
  randomState: number | undefined,
): Matrix {
  const nSamples = X.length;
  const nFeatures = X[0].length;
  const maxPcaComponents = Math.min(nSamples, nFeatures);

  if (nComponents <= maxPcaComponents) {
    const pca = new PCA({ nComponents }).fit(X);
    const init = pca.transform(X);
    for (let i = 0; i < init.length; i += 1) {
      for (let j = 0; j < init[i].length; j += 1) {
        init[i][j] *= 1e-4;
      }
    }
    return init;
  }

  const rng = randomState === undefined ? Math.random : mulberry32(randomState);
  const out: Matrix = Array.from({ length: nSamples }, () => new Array<number>(nComponents).fill(0));
  for (let i = 0; i < nSamples; i += 1) {
    for (let j = 0; j < nComponents; j += 1) {
      out[i][j] = sampleStandardNormal(rng) * 1e-4;
    }
  }
  return out;
}

function computeGradient(Y: Matrix, P: Matrix): { gradient: Matrix; kl: number } {
  const n = Y.length;
  const d = Y[0].length;
  const num: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const gradient: Matrix = Array.from({ length: n }, () => new Array<number>(d).fill(0));

  let sumNum = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const qNumerator = 1 / (1 + squaredEuclideanDistance(Y[i], Y[j]));
      num[i][j] = qNumerator;
      num[j][i] = qNumerator;
      sumNum += 2 * qNumerator;
    }
  }
  sumNum = Math.max(sumNum, 1e-24);

  let kl = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const p = Math.max(P[i][j], 1e-24);
      const q = Math.max(num[i][j] / sumNum, 1e-24);
      const scale = 4 * (p - q) * num[i][j];
      for (let c = 0; c < d; c += 1) {
        const diff = Y[i][c] - Y[j][c];
        const update = scale * diff;
        gradient[i][c] += update;
        gradient[j][c] -= update;
      }
      kl += 2 * p * Math.log(p / q);
    }
  }

  return { gradient, kl };
}

function centerEmbedding(Y: Matrix): void {
  const n = Y.length;
  const d = Y[0].length;
  const means = new Array<number>(d).fill(0);
  for (let i = 0; i < n; i += 1) {
    for (let c = 0; c < d; c += 1) {
      means[c] += Y[i][c];
    }
  }
  for (let c = 0; c < d; c += 1) {
    means[c] /= n;
  }
  for (let i = 0; i < n; i += 1) {
    for (let c = 0; c < d; c += 1) {
      Y[i][c] -= means[c];
    }
  }
}

export class TSNE {
  embedding_: Matrix | null = null;
  nFeaturesIn_: number | null = null;
  klDivergence_: number | null = null;

  private nComponents: number;
  private perplexity: number;
  private learningRate: number;
  private maxIter: number;
  private earlyExaggeration: number;
  private randomState?: number;

  constructor(options: TSNEOptions = {}) {
    this.nComponents = options.nComponents ?? 2;
    this.perplexity = options.perplexity ?? 30;
    this.learningRate = options.learningRate ?? 200;
    this.maxIter = options.maxIter ?? 1000;
    this.earlyExaggeration = options.earlyExaggeration ?? 12;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new Error(`nComponents must be an integer >= 1. Got ${this.nComponents}.`);
    }
    if (!Number.isFinite(this.perplexity) || this.perplexity <= 0) {
      throw new Error(`perplexity must be finite and > 0. Got ${this.perplexity}.`);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new Error(`learningRate must be finite and > 0. Got ${this.learningRate}.`);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 250) {
      throw new Error(`maxIter must be an integer >= 250. Got ${this.maxIter}.`);
    }
    if (!Number.isFinite(this.earlyExaggeration) || this.earlyExaggeration < 1) {
      throw new Error(
        `earlyExaggeration must be finite and >= 1. Got ${this.earlyExaggeration}.`,
      );
    }
  }

  fit(X: Matrix): this {
    this.embedding_ = this.fitTransform(X);
    return this;
  }

  fitTransform(X: Matrix): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);
    if (this.perplexity >= X.length) {
      throw new Error(
        `perplexity must be smaller than nSamples (${X.length}). Got ${this.perplexity}.`,
      );
    }

    const distancesSquared = pairwiseSquaredDistances(X);
    const P = jointProbabilities(distancesSquared, this.perplexity);
    const exaggeratedP = P.map((row) => row.map((value) => value * this.earlyExaggeration));

    const embedding = initializeEmbedding(X, this.nComponents, this.randomState);
    const velocity: Matrix = Array.from({ length: embedding.length }, () =>
      new Array<number>(this.nComponents).fill(0),
    );

    const exaggerationIters = Math.min(250, this.maxIter);
    const momentumSwitchIter = Math.min(250, this.maxIter);
    for (let iter = 0; iter < this.maxIter; iter += 1) {
      const probabilityMatrix = iter < exaggerationIters ? exaggeratedP : P;
      const { gradient } = computeGradient(embedding, probabilityMatrix);
      const momentum = iter < momentumSwitchIter ? 0.5 : 0.8;

      for (let i = 0; i < embedding.length; i += 1) {
        for (let c = 0; c < this.nComponents; c += 1) {
          velocity[i][c] = momentum * velocity[i][c] - this.learningRate * gradient[i][c];
          embedding[i][c] += velocity[i][c];
        }
      }
      centerEmbedding(embedding);
    }

    const { kl } = computeGradient(embedding, P);
    this.embedding_ = embedding.map((row) => row.slice());
    this.nFeaturesIn_ = X[0].length;
    this.klDivergence_ = kl;
    return embedding;
  }
}
