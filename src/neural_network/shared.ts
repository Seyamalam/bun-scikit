import type { Matrix, Vector } from "../types";

export type MLPActivation = "identity" | "logistic" | "tanh" | "relu";
export type MLPSolver = "adam" | "sgd";

export interface MLPCommonOptions {
  hiddenLayerSizes?: number | number[];
  activation?: MLPActivation;
  solver?: MLPSolver;
  alpha?: number;
  batchSize?: number;
  learningRateInit?: number;
  maxIter?: number;
  tolerance?: number;
  randomState?: number;
}

export interface NetworkParams {
  coefs: Matrix[];
  intercepts: Vector[];
}

export interface AdamState {
  mW: Matrix[];
  vW: Matrix[];
  mB: Vector[];
  vB: Vector[];
  t: number;
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianRandom(random: () => number): number {
  const u1 = Math.max(1e-12, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function parseHiddenLayerSizes(hiddenLayerSizes?: number | number[]): number[] {
  if (hiddenLayerSizes === undefined) {
    return [100];
  }
  const layers = Array.isArray(hiddenLayerSizes)
    ? hiddenLayerSizes.slice()
    : [hiddenLayerSizes];
  if (layers.length === 0) {
    return [];
  }
  for (let i = 0; i < layers.length; i += 1) {
    if (!Number.isInteger(layers[i]) || layers[i] < 1) {
      throw new Error(`hiddenLayerSizes entries must be integers >= 1. Got ${layers[i]}.`);
    }
  }
  return layers;
}

export function activationForward(value: number, activation: MLPActivation): number {
  switch (activation) {
    case "identity":
      return value;
    case "logistic": {
      if (value >= 0) {
        const expNeg = Math.exp(-value);
        return 1 / (1 + expNeg);
      }
      const expPos = Math.exp(value);
      return expPos / (1 + expPos);
    }
    case "tanh":
      return Math.tanh(value);
    case "relu":
      return value > 0 ? value : 0;
    default:
      return value;
  }
}

export function activationDerivativeFromZ(value: number, activation: MLPActivation): number {
  switch (activation) {
    case "identity":
      return 1;
    case "logistic": {
      const s = activationForward(value, "logistic");
      return s * (1 - s);
    }
    case "tanh": {
      const t = Math.tanh(value);
      return 1 - t * t;
    }
    case "relu":
      return value > 0 ? 1 : 0;
    default:
      return 1;
  }
}

export function initializeNetwork(
  layerSizes: number[],
  random: () => number,
): NetworkParams {
  const coefs: Matrix[] = [];
  const intercepts: Vector[] = [];
  for (let layer = 0; layer < layerSizes.length - 1; layer += 1) {
    const fanIn = layerSizes[layer];
    const fanOut = layerSizes[layer + 1];
    const scale = Math.sqrt(2 / Math.max(1, fanIn));
    const weights: Matrix = Array.from({ length: fanIn }, () =>
      new Array<number>(fanOut).fill(0),
    );
    for (let i = 0; i < fanIn; i += 1) {
      for (let j = 0; j < fanOut; j += 1) {
        weights[i][j] = gaussianRandom(random) * scale;
      }
    }
    const bias = new Array<number>(fanOut).fill(0);
    coefs.push(weights);
    intercepts.push(bias);
  }
  return { coefs, intercepts };
}

export function initAdamState(coefs: Matrix[], intercepts: Vector[]): AdamState {
  return {
    mW: coefs.map((layer) => layer.map((row) => row.map(() => 0))),
    vW: coefs.map((layer) => layer.map((row) => row.map(() => 0))),
    mB: intercepts.map((layer) => layer.map(() => 0)),
    vB: intercepts.map((layer) => layer.map(() => 0)),
    t: 0,
  };
}

export function matMulAddBias(A: Matrix, W: Matrix, b: Vector): Matrix {
  const out: Matrix = Array.from({ length: A.length }, () => new Array<number>(b.length).fill(0));
  for (let i = 0; i < A.length; i += 1) {
    for (let j = 0; j < b.length; j += 1) {
      let sum = b[j];
      for (let k = 0; k < W.length; k += 1) {
        sum += A[i][k] * W[k][j];
      }
      out[i][j] = sum;
    }
  }
  return out;
}

export function applyActivationInPlace(X: Matrix, activation: MLPActivation): void {
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      X[i][j] = activationForward(X[i][j], activation);
    }
  }
}

export function softmaxInPlace(X: Matrix): void {
  for (let i = 0; i < X.length; i += 1) {
    let maxValue = X[i][0];
    for (let j = 1; j < X[i].length; j += 1) {
      if (X[i][j] > maxValue) {
        maxValue = X[i][j];
      }
    }
    let sum = 0;
    for (let j = 0; j < X[i].length; j += 1) {
      X[i][j] = Math.exp(X[i][j] - maxValue);
      sum += X[i][j];
    }
    const denom = Math.max(1e-12, sum);
    for (let j = 0; j < X[i].length; j += 1) {
      X[i][j] /= denom;
    }
  }
}

export function argmaxRow(row: Vector): number {
  let bestIndex = 0;
  let bestValue = row[0];
  for (let i = 1; i < row.length; i += 1) {
    if (row[i] > bestValue) {
      bestValue = row[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}
