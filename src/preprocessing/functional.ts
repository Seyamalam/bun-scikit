import type { Matrix } from "../types";
import {
  assertConsistentRowSize,
  assertFiniteMatrix,
  assertNonEmptyMatrix,
} from "../utils/validation";
import { Binarizer } from "./Binarizer";
import { MaxAbsScaler } from "./MaxAbsScaler";
import { MinMaxScaler, type MinMaxScalerOptions } from "./MinMaxScaler";
import { RobustScaler, type RobustScalerOptions } from "./RobustScaler";
import { StandardScaler } from "./StandardScaler";

export interface AddDummyFeatureOptions {
  value?: number;
}

export interface BinarizeOptions {
  threshold?: number;
}

export function addDummyFeature(X: Matrix, options: AddDummyFeatureOptions = {}): Matrix {
  assertNonEmptyMatrix(X);
  assertConsistentRowSize(X);
  assertFiniteMatrix(X);

  const value = options.value ?? 1;
  if (!Number.isFinite(value)) {
    throw new Error(`value must be finite. Got ${value}.`);
  }

  return X.map((row) => [value, ...row]);
}

export function binarize(X: Matrix, options: BinarizeOptions = {}): Matrix {
  return new Binarizer({ threshold: options.threshold }).fitTransform(X);
}

export function scale(X: Matrix): Matrix {
  return new StandardScaler().fitTransform(X);
}

export function minmaxScale(X: Matrix, options: MinMaxScalerOptions = {}): Matrix {
  return new MinMaxScaler(options).fitTransform(X);
}

export function maxabsScale(X: Matrix): Matrix {
  return new MaxAbsScaler().fitTransform(X);
}

export function robustScale(X: Matrix, options: RobustScalerOptions = {}): Matrix {
  return new RobustScaler(options).fitTransform(X);
}
