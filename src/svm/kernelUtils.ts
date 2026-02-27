import type { Matrix, Vector } from "../types";
import { dot, mean } from "../utils/linalg";

export type KernelName = "linear" | "rbf" | "poly" | "sigmoid";
export type GammaOption = "scale" | "auto" | number;

export interface KernelOptions {
  kernel?: KernelName;
  gamma?: GammaOption;
  degree?: number;
  coef0?: number;
}

export interface ResolvedKernelConfig {
  kernel: KernelName;
  gamma: number;
  degree: number;
  coef0: number;
}

function varianceOfMatrix(X: Matrix): number {
  const flat: number[] = [];
  for (let i = 0; i < X.length; i += 1) {
    flat.push(...X[i]);
  }
  const avg = mean(flat);
  let sum = 0;
  for (let i = 0; i < flat.length; i += 1) {
    const d = flat[i] - avg;
    sum += d * d;
  }
  return sum / Math.max(1, flat.length);
}

export function resolveKernelConfig(X: Matrix, options: KernelOptions): ResolvedKernelConfig {
  const kernel = options.kernel ?? "rbf";
  const degree = options.degree ?? 3;
  const coef0 = options.coef0 ?? 0;
  const nFeatures = X[0].length;
  const gammaOpt = options.gamma ?? "scale";
  let gamma: number;
  if (typeof gammaOpt === "number") {
    gamma = gammaOpt;
  } else if (gammaOpt === "auto") {
    gamma = 1 / nFeatures;
  } else {
    const variance = Math.max(varianceOfMatrix(X), 1e-12);
    gamma = 1 / (nFeatures * variance);
  }
  if (!Number.isFinite(gamma) || gamma <= 0) {
    throw new Error(`gamma must resolve to a finite positive number. Got ${gamma}.`);
  }
  if (!Number.isInteger(degree) || degree < 1) {
    throw new Error(`degree must be a positive integer. Got ${degree}.`);
  }
  return { kernel, gamma, degree, coef0 };
}

export function kernelValue(
  a: Vector,
  b: Vector,
  config: ResolvedKernelConfig,
): number {
  switch (config.kernel) {
    case "linear":
      return dot(a, b);
    case "rbf": {
      let sumSquared = 0;
      for (let i = 0; i < a.length; i += 1) {
        const diff = a[i] - b[i];
        sumSquared += diff * diff;
      }
      return Math.exp(-config.gamma * sumSquared);
    }
    case "poly":
      return (config.gamma * dot(a, b) + config.coef0) ** config.degree;
    case "sigmoid":
      return Math.tanh(config.gamma * dot(a, b) + config.coef0);
    default: {
      const exhaustive: never = config.kernel;
      throw new Error(`Unsupported kernel: ${exhaustive}`);
    }
  }
}
