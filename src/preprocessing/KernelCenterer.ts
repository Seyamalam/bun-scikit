import type { Matrix } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

function columnMeans(X: Matrix): number[] {
  const means = new Array<number>(X[0].length).fill(0);
  for (let i = 0; i < X.length; i += 1) {
    for (let j = 0; j < X[i].length; j += 1) {
      means[j] += X[i][j];
    }
  }
  for (let j = 0; j < means.length; j += 1) {
    means[j] /= X.length;
  }
  return means;
}

export class KernelCenterer {
  KFitRowsMean_: number[] | null = null;
  KAllMean_: number | null = null;

  private fitted = false;

  fit(K: Matrix): this {
    assertNonEmptyMatrix(K, "K");
    assertConsistentRowSize(K, "K");
    assertFiniteMatrix(K, "K");

    this.KFitRowsMean_ = columnMeans(K);
    let allMean = 0;
    for (let i = 0; i < K.length; i += 1) {
      for (let j = 0; j < K[i].length; j += 1) {
        allMean += K[i][j];
      }
    }
    this.KAllMean_ = allMean / (K.length * K[0].length);
    this.fitted = true;
    return this;
  }

  transform(K: Matrix): Matrix {
    this.assertFitted();
    assertNonEmptyMatrix(K, "K");
    assertConsistentRowSize(K, "K");
    assertFiniteMatrix(K, "K");
    if (K[0].length !== this.KFitRowsMean_!.length) {
      throw new Error(`Kernel width mismatch. Expected ${this.KFitRowsMean_!.length}, got ${K[0].length}.`);
    }

    const rowMeans = new Array<number>(K.length).fill(0);
    for (let i = 0; i < K.length; i += 1) {
      let sum = 0;
      for (let j = 0; j < K[i].length; j += 1) {
        sum += K[i][j];
      }
      rowMeans[i] = sum / K[i].length;
    }

    const centered: Matrix = new Array(K.length);
    for (let i = 0; i < K.length; i += 1) {
      const row = new Array<number>(K[i].length);
      for (let j = 0; j < K[i].length; j += 1) {
        row[j] = K[i][j] - this.KFitRowsMean_![j] - rowMeans[i] + this.KAllMean_!;
      }
      centered[i] = row;
    }
    return centered;
  }

  fitTransform(K: Matrix): Matrix {
    return this.fit(K).transform(K);
  }

  private assertFitted(): void {
    if (!this.fitted || this.KFitRowsMean_ === null || this.KAllMean_ === null) {
      throw new Error("KernelCenterer has not been fitted.");
    }
  }
}

