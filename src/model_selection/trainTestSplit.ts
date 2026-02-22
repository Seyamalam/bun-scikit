export interface TrainTestSplitOptions {
  testSize?: number;
  shuffle?: boolean;
  randomState?: number;
}

export interface TrainTestSplitResult<TX, TY> {
  XTrain: TX[];
  XTest: TX[];
  yTrain: TY[];
  yTest: TY[];
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function trainTestSplit<TX, TY>(
  X: TX[],
  y: TY[],
  options: TrainTestSplitOptions = {},
): TrainTestSplitResult<TX, TY> {
  if (X.length !== y.length) {
    throw new Error(`X and y must have the same length. Got ${X.length} and ${y.length}.`);
  }

  if (X.length < 2) {
    throw new Error("At least two samples are required for train/test splitting.");
  }

  const shuffle = options.shuffle ?? true;
  const randomState = options.randomState ?? 42;
  const testSize = options.testSize ?? 0.25;
  const sampleCount = X.length;

  let testCount: number;
  if (testSize > 0 && testSize < 1) {
    testCount = Math.max(1, Math.floor(sampleCount * testSize));
  } else if (Number.isInteger(testSize) && testSize >= 1 && testSize < sampleCount) {
    testCount = testSize;
  } else {
    throw new Error(
      `testSize must be a float in (0, 1) or int in [1, n-1]. Got ${testSize}.`,
    );
  }

  const indices = Array.from({ length: sampleCount }, (_, idx) => idx);

  if (shuffle) {
    const random = mulberry32(randomState);
    for (let i = indices.length - 1; i > 0; i -= 1) {
      const j = Math.floor(random() * (i + 1));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }
  }

  const testIndices = new Set(indices.slice(0, testCount));
  const XTrain: TX[] = [];
  const XTest: TX[] = [];
  const yTrain: TY[] = [];
  const yTest: TY[] = [];

  for (let i = 0; i < sampleCount; i += 1) {
    if (testIndices.has(i)) {
      XTest.push(X[i]);
      yTest.push(y[i]);
    } else {
      XTrain.push(X[i]);
      yTrain.push(y[i]);
    }
  }

  return { XTrain, XTest, yTrain, yTest };
}
