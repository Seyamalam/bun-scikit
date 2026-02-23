import { mean } from "../utils/linalg";

function validateInputs(yTrue: number[], yPred: number[]): void {
  if (yTrue.length === 0 || yPred.length === 0) {
    throw new Error("yTrue and yPred must be non-empty.");
  }

  if (yTrue.length !== yPred.length) {
    throw new Error(`Length mismatch: yTrue=${yTrue.length}, yPred=${yPred.length}.`);
  }
}

export function meanSquaredError(yTrue: number[], yPred: number[]): number {
  validateInputs(yTrue, yPred);
  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    const diff = yTrue[i] - yPred[i];
    total += diff * diff;
  }
  return total / yTrue.length;
}

export function meanAbsoluteError(yTrue: number[], yPred: number[]): number {
  validateInputs(yTrue, yPred);
  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    total += Math.abs(yTrue[i] - yPred[i]);
  }
  return total / yTrue.length;
}

export function r2Score(yTrue: number[], yPred: number[]): number {
  validateInputs(yTrue, yPred);

  const yMean = mean(yTrue);
  let ssRes = 0;
  let ssTot = 0;

  for (let i = 0; i < yTrue.length; i += 1) {
    const residual = yTrue[i] - yPred[i];
    const centered = yTrue[i] - yMean;
    ssRes += residual * residual;
    ssTot += centered * centered;
  }

  if (ssTot === 0) {
    return ssRes === 0 ? 1 : 0;
  }

  return 1 - ssRes / ssTot;
}

export function meanAbsolutePercentageError(yTrue: number[], yPred: number[]): number {
  validateInputs(yTrue, yPred);
  let total = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    const denom = Math.max(Math.abs(yTrue[i]), 1e-12);
    total += Math.abs((yTrue[i] - yPred[i]) / denom);
  }
  return total / yTrue.length;
}

export function explainedVarianceScore(yTrue: number[], yPred: number[]): number {
  validateInputs(yTrue, yPred);
  const n = yTrue.length;
  const yTrueMean = mean(yTrue);
  const residuals = new Array<number>(n);
  let residualMean = 0;
  for (let i = 0; i < n; i += 1) {
    const r = yTrue[i] - yPred[i];
    residuals[i] = r;
    residualMean += r;
  }
  residualMean /= n;

  let varTrue = 0;
  let varResidual = 0;
  for (let i = 0; i < n; i += 1) {
    const centeredY = yTrue[i] - yTrueMean;
    const centeredR = residuals[i] - residualMean;
    varTrue += centeredY * centeredY;
    varResidual += centeredR * centeredR;
  }

  varTrue /= n;
  varResidual /= n;
  if (varTrue === 0) {
    return varResidual === 0 ? 1 : 0;
  }
  return 1 - varResidual / varTrue;
}
