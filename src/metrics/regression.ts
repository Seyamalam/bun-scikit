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
