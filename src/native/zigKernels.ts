import { dlopen, FFIType, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

type NativeHandle = bigint;

type LinearModelCreateFn = (nFeatures: number, fitIntercept: number) => NativeHandle;
type LinearModelDestroyFn = (handle: NativeHandle) => void;
type LinearModelFitFn = (
  handle: NativeHandle,
  x: Float64Array,
  y: Float64Array,
  nSamples: number,
  l2: number,
) => number;
type LinearModelPredictFn = (
  handle: NativeHandle,
  x: Float64Array,
  nSamples: number,
  out: Float64Array,
) => number;
type LinearModelCopyCoefficientsFn = (handle: NativeHandle, out: Float64Array) => number;
type LinearModelGetInterceptFn = (handle: NativeHandle) => number;

type LogisticModelCreateFn = (nFeatures: number, fitIntercept: number) => NativeHandle;
type LogisticModelDestroyFn = (handle: NativeHandle) => void;
type LogisticModelFitFn = (
  handle: NativeHandle,
  x: Float64Array,
  y: Float64Array,
  nSamples: number,
  learningRate: number,
  l2: number,
  maxIter: number,
  tolerance: number,
) => bigint;
type LogisticModelFitLbfgsFn = (
  handle: NativeHandle,
  x: Float64Array,
  y: Float64Array,
  nSamples: number,
  maxIter: number,
  tolerance: number,
  l2: number,
  memory: number,
) => bigint;
type LogisticModelPredictProbaFn = (
  handle: NativeHandle,
  x: Float64Array,
  nSamples: number,
  outPositive: Float64Array,
) => number;
type LogisticModelPredictFn = (
  handle: NativeHandle,
  x: Float64Array,
  nSamples: number,
  outLabels: Uint8Array,
) => number;
type LogisticModelCopyCoefficientsFn = (handle: NativeHandle, out: Float64Array) => number;
type LogisticModelGetInterceptFn = (handle: NativeHandle) => number;

type DecisionTreeModelCreateFn = (
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  maxFeaturesMode: number,
  maxFeaturesValue: number,
  randomState: number,
  useRandomState: number,
  nFeatures: number,
) => NativeHandle;
type DecisionTreeModelDestroyFn = (handle: NativeHandle) => void;
type DecisionTreeModelFitFn = (
  handle: NativeHandle,
  x: Float64Array,
  y: Uint8Array,
  nSamples: number,
  nFeatures: number,
  sampleIndices: Uint32Array,
  sampleCount: number,
) => number;
type DecisionTreeModelPredictFn = (
  handle: NativeHandle,
  x: Float64Array,
  nSamples: number,
  nFeatures: number,
  outLabels: Uint8Array,
) => number;

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
    linear_model_create?: LinearModelCreateFn;
    linear_model_destroy?: LinearModelDestroyFn;
    linear_model_fit?: LinearModelFitFn;
    linear_model_predict?: LinearModelPredictFn;
    linear_model_copy_coefficients?: LinearModelCopyCoefficientsFn;
    linear_model_get_intercept?: LinearModelGetInterceptFn;
    logistic_model_create?: LogisticModelCreateFn;
    logistic_model_destroy?: LogisticModelDestroyFn;
    logistic_model_fit?: LogisticModelFitFn;
    logistic_model_fit_lbfgs?: LogisticModelFitLbfgsFn;
    logistic_model_predict_proba?: LogisticModelPredictProbaFn;
    logistic_model_predict?: LogisticModelPredictFn;
    logistic_model_copy_coefficients?: LogisticModelCopyCoefficientsFn;
    logistic_model_get_intercept?: LogisticModelGetInterceptFn;
    decision_tree_model_create?: DecisionTreeModelCreateFn;
    decision_tree_model_destroy?: DecisionTreeModelDestroyFn;
    decision_tree_model_fit?: DecisionTreeModelFitFn;
    decision_tree_model_predict?: DecisionTreeModelPredictFn;
    logistic_train_epoch?: LogisticTrainEpochFn;
    logistic_train_epochs?: LogisticTrainEpochsFn;
  };
}

export interface ZigKernels {
  linearModelCreate: LinearModelCreateFn | null;
  linearModelDestroy: LinearModelDestroyFn | null;
  linearModelFit: LinearModelFitFn | null;
  linearModelPredict: LinearModelPredictFn | null;
  linearModelCopyCoefficients: LinearModelCopyCoefficientsFn | null;
  linearModelGetIntercept: LinearModelGetInterceptFn | null;
  logisticModelCreate: LogisticModelCreateFn | null;
  logisticModelDestroy: LogisticModelDestroyFn | null;
  logisticModelFit: LogisticModelFitFn | null;
  logisticModelFitLbfgs: LogisticModelFitLbfgsFn | null;
  logisticModelPredictProba: LogisticModelPredictProbaFn | null;
  logisticModelPredict: LogisticModelPredictFn | null;
  logisticModelCopyCoefficients: LogisticModelCopyCoefficientsFn | null;
  logisticModelGetIntercept: LogisticModelGetInterceptFn | null;
  decisionTreeModelCreate: DecisionTreeModelCreateFn | null;
  decisionTreeModelDestroy: DecisionTreeModelDestroyFn | null;
  decisionTreeModelFit: DecisionTreeModelFitFn | null;
  decisionTreeModelPredict: DecisionTreeModelPredictFn | null;
  logisticTrainEpoch: LogisticTrainEpochFn | null;
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
          linear_model_create: {
            args: ["usize", FFIType.u8],
            returns: "usize",
          },
          linear_model_destroy: {
            args: ["usize"],
            returns: FFIType.void,
          },
          linear_model_fit: {
            args: ["usize", FFIType.ptr, FFIType.ptr, "usize", FFIType.f64],
            returns: FFIType.u8,
          },
          linear_model_predict: {
            args: ["usize", FFIType.ptr, "usize", FFIType.ptr],
            returns: FFIType.u8,
          },
          linear_model_copy_coefficients: {
            args: ["usize", FFIType.ptr],
            returns: FFIType.u8,
          },
          linear_model_get_intercept: {
            args: ["usize"],
            returns: FFIType.f64,
          },
          logistic_model_create: {
            args: ["usize", FFIType.u8],
            returns: "usize",
          },
          logistic_model_destroy: {
            args: ["usize"],
            returns: FFIType.void,
          },
          logistic_model_fit: {
            args: [
              "usize",
              FFIType.ptr,
              FFIType.ptr,
              "usize",
              FFIType.f64,
              FFIType.f64,
              "usize",
              FFIType.f64,
            ],
            returns: "usize",
          },
          logistic_model_fit_lbfgs: {
            args: [
              "usize",
              FFIType.ptr,
              FFIType.ptr,
              "usize",
              "usize",
              FFIType.f64,
              FFIType.f64,
              "usize",
            ],
            returns: "usize",
          },
          logistic_model_predict_proba: {
            args: ["usize", FFIType.ptr, "usize", FFIType.ptr],
            returns: FFIType.u8,
          },
          logistic_model_predict: {
            args: ["usize", FFIType.ptr, "usize", FFIType.ptr],
            returns: FFIType.u8,
          },
          logistic_model_copy_coefficients: {
            args: ["usize", FFIType.ptr],
            returns: FFIType.u8,
          },
          logistic_model_get_intercept: {
            args: ["usize"],
            returns: FFIType.f64,
          },
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
          linearModelCreate: library.symbols.linear_model_create ?? null,
          linearModelDestroy: library.symbols.linear_model_destroy ?? null,
          linearModelFit: library.symbols.linear_model_fit ?? null,
          linearModelPredict: library.symbols.linear_model_predict ?? null,
          linearModelCopyCoefficients:
            library.symbols.linear_model_copy_coefficients ?? null,
          linearModelGetIntercept: library.symbols.linear_model_get_intercept ?? null,
          logisticModelCreate: library.symbols.logistic_model_create ?? null,
          logisticModelDestroy: library.symbols.logistic_model_destroy ?? null,
          logisticModelFit: library.symbols.logistic_model_fit ?? null,
          logisticModelFitLbfgs: library.symbols.logistic_model_fit_lbfgs ?? null,
          logisticModelPredictProba: library.symbols.logistic_model_predict_proba ?? null,
          logisticModelPredict: library.symbols.logistic_model_predict ?? null,
          logisticModelCopyCoefficients:
            library.symbols.logistic_model_copy_coefficients ?? null,
          logisticModelGetIntercept: library.symbols.logistic_model_get_intercept ?? null,
          decisionTreeModelCreate: library.symbols.decision_tree_model_create ?? null,
          decisionTreeModelDestroy: library.symbols.decision_tree_model_destroy ?? null,
          decisionTreeModelFit: library.symbols.decision_tree_model_fit ?? null,
          decisionTreeModelPredict: library.symbols.decision_tree_model_predict ?? null,
          logisticTrainEpoch: library.symbols.logistic_train_epoch ?? null,
          logisticTrainEpochs: library.symbols.logistic_train_epochs ?? null,
          libraryPath,
        };

        return cachedKernels;
      } catch {
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
            linearModelCreate: null,
            linearModelDestroy: null,
            linearModelFit: null,
            linearModelPredict: null,
            linearModelCopyCoefficients: null,
            linearModelGetIntercept: null,
            logisticModelCreate: null,
            logisticModelDestroy: null,
            logisticModelFit: null,
            logisticModelFitLbfgs: null,
            logisticModelPredictProba: null,
            logisticModelPredict: null,
            logisticModelCopyCoefficients: null,
            logisticModelGetIntercept: null,
            decisionTreeModelCreate: null,
            decisionTreeModelDestroy: null,
            decisionTreeModelFit: null,
            decisionTreeModelPredict: null,
            logisticTrainEpoch: library.symbols.logistic_train_epoch ?? null,
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
            linearModelCreate: null,
            linearModelDestroy: null,
            linearModelFit: null,
            linearModelPredict: null,
            linearModelCopyCoefficients: null,
            linearModelGetIntercept: null,
            logisticModelCreate: null,
            logisticModelDestroy: null,
            logisticModelFit: null,
            logisticModelFitLbfgs: null,
            logisticModelPredictProba: null,
            logisticModelPredict: null,
            logisticModelCopyCoefficients: null,
            logisticModelGetIntercept: null,
            decisionTreeModelCreate: null,
            decisionTreeModelDestroy: null,
            decisionTreeModelFit: null,
            decisionTreeModelPredict: null,
            logisticTrainEpoch: library.symbols.logistic_train_epoch ?? null,
            logisticTrainEpochs: null,
            libraryPath,
          };

          return cachedKernels;
        }
      }
    } catch {
      continue;
    }
  }

  cachedKernels = null;
  return cachedKernels;
}
