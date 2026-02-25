import {
  DecisionTreeClassifier,
  RandomForestClassifier,
  type ClassificationModel,
  type Matrix,
  type Vector,
} from "../src";

type TreeBackendMode = "js-fast" | "zig-tree";

interface BenchResult {
  label: string;
  backend: TreeBackendMode;
  fitMedianMs: number;
  predictMedianMs: number;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
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

function generateSyntheticDataset(
  samples: number,
  features: number,
  seed: number,
): { XTrain: Matrix; yTrain: Vector; XTest: Matrix; yTest: Vector } {
  const rand = mulberry32(seed);
  const X: Matrix = new Array(samples);
  const y: Vector = new Array(samples);
  const weights = new Array<number>(features);
  for (let j = 0; j < features; j += 1) {
    weights[j] = (rand() - 0.5) * 3;
  }

  for (let i = 0; i < samples; i += 1) {
    const row = new Array<number>(features);
    let score = 0;
    for (let j = 0; j < features; j += 1) {
      const value = rand() * 2 - 1;
      row[j] = value;
      score += value * weights[j];
    }
    score += 0.2 * (row[0] > 0 ? 1 : -1) + 0.1 * (row[1] * row[1] - row[2]);
    X[i] = row;
    y[i] = score > 0 ? 1 : 0;
  }

  const trainSize = Math.floor(samples * 0.8);
  return {
    XTrain: X.slice(0, trainSize),
    yTrain: y.slice(0, trainSize),
    XTest: X.slice(trainSize),
    yTest: y.slice(trainSize),
  };
}

function withTreeBackend<T>(backend: TreeBackendMode, fn: () => T): T {
  const previous = process.env.BUN_SCIKIT_TREE_BACKEND;
  if (backend === "zig-tree") {
    process.env.BUN_SCIKIT_TREE_BACKEND = "zig";
  } else {
    process.env.BUN_SCIKIT_TREE_BACKEND = "js";
  }
  try {
    return fn();
  } finally {
    if (previous === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previous;
    }
  }
}

function runModelBenchmark(
  label: string,
  backend: TreeBackendMode,
  iterations: number,
  warmup: number,
  XTrain: Matrix,
  yTrain: Vector,
  XTest: Matrix,
  factory: () => ClassificationModel & { dispose?: () => void },
): BenchResult {
  return withTreeBackend(backend, () => {
    const fitTimes: number[] = [];
    const predictTimes: number[] = [];
    const loops = warmup + iterations;
    for (let i = 0; i < loops; i += 1) {
      const model = factory();
      try {
        const fitStart = performance.now();
        model.fit(XTrain, yTrain);
        const fitMs = performance.now() - fitStart;

        const predictStart = performance.now();
        model.predict(XTest);
        const predictMs = performance.now() - predictStart;

        if (i >= warmup) {
          fitTimes.push(fitMs);
          predictTimes.push(predictMs);
        }
      } finally {
        model.dispose?.();
      }
    }

    return {
      label,
      backend,
      fitMedianMs: median(fitTimes),
      predictMedianMs: median(predictTimes),
    };
  });
}

function findResult(results: BenchResult[], label: string, backend: TreeBackendMode): BenchResult {
  const match = results.find((entry) => entry.label === label && entry.backend === backend);
  if (!match) {
    throw new Error(`Missing result for ${label} (${backend}).`);
  }
  return match;
}

const iterations = Number(process.env.BENCH_ITERATIONS ?? "20");
const warmup = Number(process.env.BENCH_WARMUP ?? "5");
const samples = Number(process.env.BENCH_TREE_SAMPLES ?? "6000");
const features = Number(process.env.BENCH_TREE_FEATURES ?? "24");
const seed = Number(process.env.BENCH_SEED ?? "42");

const { XTrain, yTrain, XTest } = generateSyntheticDataset(samples, features, seed);
const results: BenchResult[] = [];

results.push(
  runModelBenchmark(
    "decision_tree",
    "js-fast",
    iterations,
    warmup,
    XTrain,
    yTrain,
    XTest,
    () =>
      new DecisionTreeClassifier({
        maxDepth: 10,
        minSamplesLeaf: 2,
        randomState: seed,
      }),
  ),
);
results.push(
  runModelBenchmark(
    "decision_tree",
    "zig-tree",
    iterations,
    warmup,
    XTrain,
    yTrain,
    XTest,
    () =>
      new DecisionTreeClassifier({
        maxDepth: 10,
        minSamplesLeaf: 2,
        randomState: seed,
      }),
  ),
);
results.push(
  runModelBenchmark(
    "random_forest",
    "js-fast",
    iterations,
    warmup,
    XTrain,
    yTrain,
    XTest,
    () =>
      new RandomForestClassifier({
        nEstimators: 120,
        maxDepth: 10,
        minSamplesLeaf: 2,
        randomState: seed,
      }),
  ),
);
results.push(
  runModelBenchmark(
    "random_forest",
    "zig-tree",
    iterations,
    warmup,
    XTrain,
    yTrain,
    XTest,
    () =>
      new RandomForestClassifier({
        nEstimators: 120,
        maxDepth: 10,
        minSamplesLeaf: 2,
        randomState: seed,
      }),
  ),
);

console.log(
  `tree hotpaths | samples=${samples} features=${features} train=${XTrain.length} test=${XTest.length} iterations=${iterations} warmup=${warmup}`,
);

for (const result of results) {
  console.log(
    `${result.label} [${result.backend}] fit=${result.fitMedianMs.toFixed(3)}ms predict=${result.predictMedianMs.toFixed(3)}ms`,
  );
}

const dtJs = findResult(results, "decision_tree", "js-fast");
const dtZig = findResult(results, "decision_tree", "zig-tree");
const rfJs = findResult(results, "random_forest", "js-fast");
const rfZig = findResult(results, "random_forest", "zig-tree");

console.log(
  `speedup decision_tree zig/js fit=${(dtJs.fitMedianMs / dtZig.fitMedianMs).toFixed(3)}x predict=${(dtJs.predictMedianMs / dtZig.predictMedianMs).toFixed(3)}x`,
);
console.log(
  `speedup random_forest zig/js fit=${(rfJs.fitMedianMs / rfZig.fitMedianMs).toFixed(3)}x predict=${(rfJs.predictMedianMs / rfZig.predictMedianMs).toFixed(3)}x`,
);
