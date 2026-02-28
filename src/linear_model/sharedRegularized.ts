import type { Matrix, Vector } from "../types";
import { dot, solveSymmetricPositiveDefinite, transpose } from "../utils/linalg";

export interface CenteredData {
  XCentered: Matrix;
  yCentered: Vector;
  xOffset: Vector;
  yOffset: number;
}

export function centerData(X: Matrix, y: Vector, fitIntercept: boolean): CenteredData {
  const nSamples = X.length;
  const nFeatures = X[0].length;

  if (!fitIntercept) {
    return {
      XCentered: X.map((row) => row.slice()),
      yCentered: y.slice(),
      xOffset: new Array<number>(nFeatures).fill(0),
      yOffset: 0,
    };
  }

  const xOffset = new Array<number>(nFeatures).fill(0);
  for (let i = 0; i < nSamples; i += 1) {
    for (let j = 0; j < nFeatures; j += 1) {
      xOffset[j] += X[i][j];
    }
  }
  for (let j = 0; j < nFeatures; j += 1) {
    xOffset[j] /= nSamples;
  }

  let yOffset = 0;
  for (let i = 0; i < nSamples; i += 1) {
    yOffset += y[i];
  }
  yOffset /= nSamples;

  const XCentered: Matrix = new Array(nSamples);
  const yCentered = new Array<number>(nSamples);
  for (let i = 0; i < nSamples; i += 1) {
    const row = new Array<number>(nFeatures);
    for (let j = 0; j < nFeatures; j += 1) {
      row[j] = X[i][j] - xOffset[j];
    }
    XCentered[i] = row;
    yCentered[i] = y[i] - yOffset;
  }

  return { XCentered, yCentered, xOffset, yOffset };
}

export function fitRidgeClosedForm(
  X: Matrix,
  y: Vector,
  alpha: number,
  fitIntercept: boolean,
): { coef: Vector; intercept: number } {
  const centered = centerData(X, y, fitIntercept);
  const Xt = transpose(centered.XCentered);
  const nFeatures = Xt.length;

  const gram: Matrix = Array.from({ length: nFeatures }, () =>
    new Array<number>(nFeatures).fill(0),
  );
  for (let i = 0; i < nFeatures; i += 1) {
    for (let j = i; j < nFeatures; j += 1) {
      let sum = 0;
      for (let r = 0; r < centered.XCentered.length; r += 1) {
        sum += Xt[i][r] * centered.XCentered[r][j];
      }
      gram[i][j] = sum;
      gram[j][i] = sum;
    }
  }
  for (let i = 0; i < nFeatures; i += 1) {
    gram[i][i] += alpha;
  }

  const rhs = new Array<number>(nFeatures).fill(0);
  for (let i = 0; i < nFeatures; i += 1) {
    rhs[i] = dot(Xt[i], centered.yCentered);
  }

  const coef = solveSymmetricPositiveDefinite(gram, rhs);
  let intercept = 0;
  if (fitIntercept) {
    intercept = centered.yOffset;
    for (let j = 0; j < coef.length; j += 1) {
      intercept -= coef[j] * centered.xOffset[j];
    }
  }
  return { coef, intercept };
}

function softThreshold(value: number, threshold: number): number {
  if (value > threshold) {
    return value - threshold;
  }
  if (value < -threshold) {
    return value + threshold;
  }
  return 0;
}

export function fitCoordinateDescent(
  X: Matrix,
  y: Vector,
  alpha: number,
  l1Ratio: number,
  fitIntercept: boolean,
  maxIter: number,
  tolerance: number,
): { coef: Vector; intercept: number; nIter: number } {
  const centered = centerData(X, y, fitIntercept);
  const nSamples = centered.XCentered.length;
  const nFeatures = centered.XCentered[0].length;

  const coef = new Array<number>(nFeatures).fill(0);
  const residual = centered.yCentered.slice();
  const columnNorm = new Array<number>(nFeatures).fill(0);

  for (let j = 0; j < nFeatures; j += 1) {
    let sum = 0;
    for (let i = 0; i < nSamples; i += 1) {
      sum += centered.XCentered[i][j] * centered.XCentered[i][j];
    }
    columnNorm[j] = sum / nSamples;
  }

  let nIter = 0;
  for (let iter = 0; iter < maxIter; iter += 1) {
    nIter = iter + 1;
    let maxDelta = 0;

    for (let j = 0; j < nFeatures; j += 1) {
      const old = coef[j];
      if (old !== 0) {
        for (let i = 0; i < nSamples; i += 1) {
          residual[i] += centered.XCentered[i][j] * old;
        }
      }

      let rho = 0;
      for (let i = 0; i < nSamples; i += 1) {
        rho += centered.XCentered[i][j] * residual[i];
      }
      rho /= nSamples;

      const denom = columnNorm[j] + alpha * (1 - l1Ratio);
      const updated =
        denom <= 1e-12 ? 0 : softThreshold(rho, alpha * l1Ratio) / denom;
      coef[j] = updated;

      for (let i = 0; i < nSamples; i += 1) {
        residual[i] -= centered.XCentered[i][j] * updated;
      }

      const delta = Math.abs(updated - old);
      if (delta > maxDelta) {
        maxDelta = delta;
      }
    }

    if (maxDelta < tolerance) {
      break;
    }
  }

  let intercept = 0;
  if (fitIntercept) {
    intercept = centered.yOffset;
    for (let j = 0; j < nFeatures; j += 1) {
      intercept -= coef[j] * centered.xOffset[j];
    }
  }

  return { coef, intercept, nIter };
}

export function meanSquaredError(yTrue: Vector, yPred: Vector): number {
  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    const d = yTrue[i] - yPred[i];
    total += d * d;
  }
  return total / Math.max(1, yTrue.length);
}
