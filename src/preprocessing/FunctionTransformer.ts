import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

export type TransformFunction = (X: Matrix) => Matrix;

export interface FunctionTransformerOptions {
  func?: TransformFunction | null;
  inverseFunc?: TransformFunction | null;
  validate?: boolean;
  acceptSparse?: boolean;
  checkInverse?: boolean;
}

function cloneMatrix(X: Matrix): Matrix {
  return X.map((row) => row.slice());
}

export class FunctionTransformer {
  nFeaturesIn_: number | null = null;

  private func: TransformFunction | null;
  private inverseFunc: TransformFunction | null;
  private validate: boolean;
  private acceptSparse: boolean;
  private checkInverse: boolean;
  private fitted = false;

  constructor(options: FunctionTransformerOptions = {}) {
    this.func = options.func ?? null;
    this.inverseFunc = options.inverseFunc ?? null;
    this.validate = options.validate ?? false;
    this.acceptSparse = options.acceptSparse ?? false;
    this.checkInverse = options.checkInverse ?? true;
  }

  fit(X: Matrix, y?: Vector): this {
    if (this.validate) {
      assertNonEmptyMatrix(X);
      assertConsistentRowSize(X);
      assertFiniteMatrix(X);
    }
    this.nFeaturesIn_ = X.length > 0 ? X[0].length : 0;
    this.fitted = true;

    if (this.checkInverse && this.func && this.inverseFunc && X.length > 0) {
      const transformed = this.func(cloneMatrix(X));
      const restored = this.inverseFunc(cloneMatrix(transformed));
      if (restored.length !== X.length || (restored[0]?.length ?? 0) !== (X[0]?.length ?? 0)) {
        throw new Error("FunctionTransformer inverseFunc appears incompatible with func output shape.");
      }
    }

    return this;
  }

  transform(X: Matrix): Matrix {
    if (this.validate) {
      assertNonEmptyMatrix(X);
      assertConsistentRowSize(X);
      assertFiniteMatrix(X);
      if (this.nFeaturesIn_ !== null && X[0].length !== this.nFeaturesIn_) {
        throw new Error(`Feature size mismatch. Expected ${this.nFeaturesIn_}, got ${X[0].length}.`);
      }
    }
    if (this.func === null) {
      return cloneMatrix(X);
    }
    return this.func(cloneMatrix(X));
  }

  inverseTransform(X: Matrix): Matrix {
    if (!this.fitted) {
      throw new Error("FunctionTransformer has not been fitted.");
    }
    if (this.inverseFunc === null) {
      return cloneMatrix(X);
    }
    return this.inverseFunc(cloneMatrix(X));
  }

  fitTransform(X: Matrix, y?: Vector): Matrix {
    return this.fit(X, y).transform(X);
  }

  get acceptsSparseInput(): boolean {
    return this.acceptSparse;
  }
}

