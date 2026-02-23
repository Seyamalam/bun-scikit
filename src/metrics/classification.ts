function validateInputs(yTrue: number[], yPred: number[]): void {
  if (yTrue.length === 0 || yPred.length === 0) {
    throw new Error("yTrue and yPred must be non-empty.");
  }

  if (yTrue.length !== yPred.length) {
    throw new Error(`Length mismatch: yTrue=${yTrue.length}, yPred=${yPred.length}.`);
  }
}

function confusionCounts(yTrue: number[], yPred: number[], positiveLabel: number): {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
} {
  validateInputs(yTrue, yPred);

  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;

  for (let i = 0; i < yTrue.length; i += 1) {
    const truthPositive = yTrue[i] === positiveLabel;
    const predPositive = yPred[i] === positiveLabel;

    if (truthPositive && predPositive) {
      tp += 1;
    } else if (!truthPositive && predPositive) {
      fp += 1;
    } else if (truthPositive && !predPositive) {
      fn += 1;
    } else {
      tn += 1;
    }
  }

  return { tp, fp, fn, tn };
}

export function accuracyScore(yTrue: number[], yPred: number[]): number {
  validateInputs(yTrue, yPred);
  let correct = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    if (yTrue[i] === yPred[i]) {
      correct += 1;
    }
  }
  return correct / yTrue.length;
}

export function precisionScore(
  yTrue: number[],
  yPred: number[],
  positiveLabel = 1,
): number {
  const { tp, fp } = confusionCounts(yTrue, yPred, positiveLabel);
  const denominator = tp + fp;
  if (denominator === 0) {
    return 0;
  }
  return tp / denominator;
}

export function recallScore(yTrue: number[], yPred: number[], positiveLabel = 1): number {
  const { tp, fn } = confusionCounts(yTrue, yPred, positiveLabel);
  const denominator = tp + fn;
  if (denominator === 0) {
    return 0;
  }
  return tp / denominator;
}

export function f1Score(yTrue: number[], yPred: number[], positiveLabel = 1): number {
  const precision = precisionScore(yTrue, yPred, positiveLabel);
  const recall = recallScore(yTrue, yPred, positiveLabel);
  const denominator = precision + recall;
  if (denominator === 0) {
    return 0;
  }
  return (2 * precision * recall) / denominator;
}

export function confusionMatrix(
  yTrue: number[],
  yPred: number[],
  positiveLabel = 1,
): [[number, number], [number, number]] {
  const { tp, fp, fn, tn } = confusionCounts(yTrue, yPred, positiveLabel);
  return [
    [tn, fp],
    [fn, tp],
  ];
}
