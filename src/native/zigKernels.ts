import { dlopen, FFIType, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

type LogisticTrainEpochFn = (
  x: Float64Array,
  y: Float64Array,
  nSamples: number,
  nFeatures: number,
  weights: Float64Array,
  intercept: Float64Array,
  gradients: Float64Array,
  learningRate: number,
  l2: number,
  fitIntercept: number,
) => number;

type LogisticTrainEpochsFn = (
  x: Float64Array,
  y: Float64Array,
  nSamples: number,
  nFeatures: number,
  weights: Float64Array,
  intercept: Float64Array,
  gradients: Float64Array,
  learningRate: number,
  l2: number,
  fitIntercept: number,
  maxIter: number,
  tolerance: number,
) => bigint;

interface ZigKernelLibrary {
  symbols: {
    logistic_train_epoch: LogisticTrainEpochFn;
    logistic_train_epochs?: LogisticTrainEpochsFn;
  };
}

export interface ZigKernels {
  logisticTrainEpoch: LogisticTrainEpochFn;
  logisticTrainEpochs: LogisticTrainEpochsFn | null;
  libraryPath: string;
}

let cachedKernels: ZigKernels | null | undefined;

function isTruthy(value: string | undefined): boolean {
  if (!value) {
    return false;
  }
  const normalized = value.trim().toLowerCase();
  return !(normalized === "0" || normalized === "false" || normalized === "off");
}

export function isZigBackendEnabled(): boolean {
  const envValue = process.env.BUN_SCIKIT_ENABLE_ZIG;
  if (!envValue) {
    return true;
  }
  return isTruthy(envValue);
}

function candidateLibraryPaths(): string[] {
  const extension = suffix;
  const fileName = `bun_scikit_kernels.${extension}`;
  const explicitPath = process.env.BUN_SCIKIT_ZIG_LIB;

  const candidates = [
    explicitPath,
    resolve(process.cwd(), "dist", "native", fileName),
    resolve(process.cwd(), "native", fileName),
    resolve(import.meta.dir, "../../dist/native", fileName),
    resolve(import.meta.dir, "../../native", fileName),
  ];

  return candidates.filter((entry): entry is string => Boolean(entry));
}

export function getZigKernels(): ZigKernels | null {
  if (!isZigBackendEnabled()) {
    return null;
  }

  if (cachedKernels !== undefined) {
    return cachedKernels;
  }

  for (const libraryPath of candidateLibraryPaths()) {
    if (!existsSync(libraryPath)) {
      continue;
    }

    try {
      try {
        const library = dlopen(libraryPath, {
          logistic_train_epoch: {
            args: [
              FFIType.ptr,
              FFIType.ptr,
              "usize",
              "usize",
              FFIType.ptr,
              FFIType.ptr,
              FFIType.ptr,
              FFIType.f64,
              FFIType.f64,
              FFIType.u8,
            ],
            returns: FFIType.f64,
          },
          logistic_train_epochs: {
            args: [
              FFIType.ptr,
              FFIType.ptr,
              "usize",
              "usize",
              FFIType.ptr,
              FFIType.ptr,
              FFIType.ptr,
              FFIType.f64,
              FFIType.f64,
              FFIType.u8,
              "usize",
              FFIType.f64,
            ],
            returns: "usize",
          },
        }) as ZigKernelLibrary;

        cachedKernels = {
          logisticTrainEpoch: library.symbols.logistic_train_epoch,
          logisticTrainEpochs: library.symbols.logistic_train_epochs ?? null,
          libraryPath,
        };

        return cachedKernels;
      } catch {
        const library = dlopen(libraryPath, {
          logistic_train_epoch: {
            args: [
              FFIType.ptr,
              FFIType.ptr,
              "usize",
              "usize",
              FFIType.ptr,
              FFIType.ptr,
              FFIType.ptr,
              FFIType.f64,
              FFIType.f64,
              FFIType.u8,
            ],
            returns: FFIType.f64,
          },
        }) as ZigKernelLibrary;

        cachedKernels = {
          logisticTrainEpoch: library.symbols.logistic_train_epoch,
          logisticTrainEpochs: null,
          libraryPath,
        };

        return cachedKernels;
      }
    } catch {
      continue;
    }
  }

  cachedKernels = null;
  return cachedKernels;
}
