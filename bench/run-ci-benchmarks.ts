import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import {
  LinearRegression,
  LogisticRegression,
  StandardScaler,
  accuracyScore,
  f1Score,
  meanSquaredError,
  r2Score,
  trainTestSplit,
} from "../src";
import { loadHeartDataset } from "../test_data/loadHeartDataset";

interface SharedBenchmarkResult {
  implementation: string;
  model: string;
  iterations: number;
  fitMsMedian: number;
  predictMsMedian: number;
  environment: Record<string, string | number>;
}

interface RegressionBenchmarkResult extends SharedBenchmarkResult {
  mse: number;
  r2: number;
}

interface ClassificationBenchmarkResult extends SharedBenchmarkResult {
  accuracy: number;
  f1: number;
}

interface RegressionSuite {
  task: "regression";
  results: [RegressionBenchmarkResult, RegressionBenchmarkResult];
  comparison: {
    fitSpeedupVsSklearn: number;
    predictSpeedupVsSklearn: number;
    mseDeltaVsSklearn: number;
    r2DeltaVsSklearn: number;
  };
}

interface ClassificationSuite {
  task: "classification";
  results: [ClassificationBenchmarkResult, ClassificationBenchmarkResult];
  comparison: {
    fitSpeedupVsSklearn: number;
    predictSpeedupVsSklearn: number;
    accuracyDeltaVsSklearn: number;
    f1DeltaVsSklearn: number;
  };
}

interface BenchmarkSnapshot {
  generatedAt: string;
  benchmarkConfig: {
    iterations: number;
    warmup: number;
  };
  dataset: {
    path: string;
    samples: number;
    features: number;
    trainSize: number;
    testSize: number;
    randomState: number;
    testFraction: number;
  };
  suites: {
    regression: RegressionSuite;
    classification: ClassificationSuite;
  };
}

const DATASET_PATH = "test_data/heart.csv";
const DEFAULT_OUTPUT_PATH = "bench/results/heart-ci-current.json";
const TEST_FRACTION = 0.2;
const RANDOM_STATE = 42;

function parseArgValue(flag: string): string | null {
  const index = Bun.argv.indexOf(flag);
  if (index === -1 || index + 1 >= Bun.argv.length) {
    return null;
  }
  return Bun.argv[index + 1];
}

function median(values: number[]): number {
  if (values.length === 0) {
    throw new Error("Cannot compute median of an empty array.");
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }
  return sorted[middle];
}

function formatMs(value: number): string {
  return `${value.toFixed(4)}ms`;
}

async function runPythonBenchmark<T>(scriptPath: string, args: string[]): Promise<T> {
  const childProcess = Bun.spawn(["python", scriptPath, ...args], {
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...globalThis.process.env,
      PYTHONHASHSEED: "0",
      OMP_NUM_THREADS: "1",
      OPENBLAS_NUM_THREADS: "1",
      MKL_NUM_THREADS: "1",
      NUMEXPR_NUM_THREADS: "1",
    },
  });

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(childProcess.stdout).text(),
    new Response(childProcess.stderr).text(),
    childProcess.exited,
  ]);

  if (exitCode !== 0) {
    throw new Error(
      `Python benchmark (${scriptPath}) failed with exit code ${exitCode}.\n${stderr || "(no stderr)"}`,
    );
  }

  return JSON.parse(stdout) as T;
}

async function prepareHeartSplit() {
  const dataset = await loadHeartDataset();
  const split = trainTestSplit(dataset.X, dataset.y, {
    testSize: TEST_FRACTION,
    randomState: RANDOM_STATE,
    shuffle: true,
  });

  const scaler = new StandardScaler();
  const XTrainScaled = scaler.fitTransform(split.XTrain);
  const XTestScaled = scaler.transform(split.XTest);

  return {
    dataset,
    split,
    XTrainScaled,
    XTestScaled,
  };
}

async function runBunRegressionBenchmark(
  iterations: number,
  warmup: number,
): Promise<RegressionBenchmarkResult> {
  const { split, XTrainScaled, XTestScaled } = await prepareHeartSplit();
  const fitTimes: number[] = [];
  const predictTimes: number[] = [];
  let predictionsForMetrics: number[] | null = null;

  const loops = warmup + iterations;
  for (let i = 0; i < loops; i += 1) {
    const model = new LinearRegression({ solver: "normal" });

    const fitStart = performance.now();
    model.fit(XTrainScaled, split.yTrain);
    const fitMs = performance.now() - fitStart;

    const predictStart = performance.now();
    const predictions = model.predict(XTestScaled);
    const predictMs = performance.now() - predictStart;

    if (i >= warmup) {
      fitTimes.push(fitMs);
      predictTimes.push(predictMs);
      predictionsForMetrics = predictions;
    }
  }

  if (!predictionsForMetrics) {
    throw new Error("No Bun regression benchmark iterations were recorded.");
  }

  return {
    implementation: "bun-scikit",
    model: "StandardScaler + LinearRegression(normal)",
    iterations,
    fitMsMedian: median(fitTimes),
    predictMsMedian: median(predictTimes),
    mse: meanSquaredError(split.yTest, predictionsForMetrics),
    r2: r2Score(split.yTest, predictionsForMetrics),
    environment: {
      bun: Bun.version,
      runtime: "bun",
    },
  };
}

async function runBunClassificationBenchmark(
  iterations: number,
  warmup: number,
): Promise<ClassificationBenchmarkResult> {
  const { split, XTrainScaled, XTestScaled } = await prepareHeartSplit();
  const fitTimes: number[] = [];
  const predictTimes: number[] = [];
  let predictionsForMetrics: number[] | null = null;

  const loops = warmup + iterations;
  for (let i = 0; i < loops; i += 1) {
    const model = new LogisticRegression({
      learningRate: 0.2,
      maxIter: 3_000,
      tolerance: 1e-6,
      l2: 0.01,
    });

    const fitStart = performance.now();
    model.fit(XTrainScaled, split.yTrain);
    const fitMs = performance.now() - fitStart;

    const predictStart = performance.now();
    const predictions = model.predict(XTestScaled);
    const predictMs = performance.now() - predictStart;

    if (i >= warmup) {
      fitTimes.push(fitMs);
      predictTimes.push(predictMs);
      predictionsForMetrics = predictions;
    }
  }

  if (!predictionsForMetrics) {
    throw new Error("No Bun classification benchmark iterations were recorded.");
  }

  return {
    implementation: "bun-scikit",
    model: "StandardScaler + LogisticRegression(gd)",
    iterations,
    fitMsMedian: median(fitTimes),
    predictMsMedian: median(predictTimes),
    accuracy: accuracyScore(split.yTest, predictionsForMetrics),
    f1: f1Score(split.yTest, predictionsForMetrics),
    environment: {
      bun: Bun.version,
      runtime: "bun",
    },
  };
}

function renderMarkdownTable(snapshot: BenchmarkSnapshot): string {
  const regression = snapshot.suites.regression;
  const classification = snapshot.suites.classification;
  const [bunReg, sklearnReg] = regression.results;
  const [bunCls, sklearnCls] = classification.results;

  return [
    "## Regression (Heart Dataset)",
    "",
    "| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |",
    "|---|---|---:|---:|---:|---:|",
    `| ${bunReg.implementation} | ${bunReg.model} | ${bunReg.fitMsMedian.toFixed(4)} | ${bunReg.predictMsMedian.toFixed(4)} | ${bunReg.mse.toFixed(6)} | ${bunReg.r2.toFixed(6)} |`,
    `| ${sklearnReg.implementation} | ${sklearnReg.model} | ${sklearnReg.fitMsMedian.toFixed(4)} | ${sklearnReg.predictMsMedian.toFixed(4)} | ${sklearnReg.mse.toFixed(6)} | ${sklearnReg.r2.toFixed(6)} |`,
    "",
    `Bun fit speedup vs scikit-learn: ${regression.comparison.fitSpeedupVsSklearn.toFixed(3)}x`,
    `Bun predict speedup vs scikit-learn: ${regression.comparison.predictSpeedupVsSklearn.toFixed(3)}x`,
    `MSE delta (bun - sklearn): ${regression.comparison.mseDeltaVsSklearn.toExponential(3)}`,
    `R2 delta (bun - sklearn): ${regression.comparison.r2DeltaVsSklearn.toExponential(3)}`,
    "",
    "## Classification (Heart Dataset)",
    "",
    "| Implementation | Model | Fit median (ms) | Predict median (ms) | Accuracy | F1 |",
    "|---|---|---:|---:|---:|---:|",
    `| ${bunCls.implementation} | ${bunCls.model} | ${bunCls.fitMsMedian.toFixed(4)} | ${bunCls.predictMsMedian.toFixed(4)} | ${bunCls.accuracy.toFixed(6)} | ${bunCls.f1.toFixed(6)} |`,
    `| ${sklearnCls.implementation} | ${sklearnCls.model} | ${sklearnCls.fitMsMedian.toFixed(4)} | ${sklearnCls.predictMsMedian.toFixed(4)} | ${sklearnCls.accuracy.toFixed(6)} | ${sklearnCls.f1.toFixed(6)} |`,
    "",
    `Bun fit speedup vs scikit-learn: ${classification.comparison.fitSpeedupVsSklearn.toFixed(3)}x`,
    `Bun predict speedup vs scikit-learn: ${classification.comparison.predictSpeedupVsSklearn.toFixed(3)}x`,
    `Accuracy delta (bun - sklearn): ${classification.comparison.accuracyDeltaVsSklearn.toExponential(3)}`,
    `F1 delta (bun - sklearn): ${classification.comparison.f1DeltaVsSklearn.toExponential(3)}`,
    "",
    `Snapshot generated at: ${snapshot.generatedAt}`,
  ].join("\n");
}

const outputPath = resolve(parseArgValue("--output") ?? DEFAULT_OUTPUT_PATH);
const outputMarkdownPath = outputPath.replace(/\.json$/i, ".md");
const iterations = Number(process.env.BENCH_ITERATIONS ?? 40);
const warmup = Number(process.env.BENCH_WARMUP ?? 5);

if (!Number.isInteger(iterations) || iterations < 1) {
  throw new Error(`BENCH_ITERATIONS must be a positive integer. Got: ${iterations}`);
}

if (!Number.isInteger(warmup) || warmup < 0) {
  throw new Error(`BENCH_WARMUP must be a non-negative integer. Got: ${warmup}`);
}

const prepared = await prepareHeartSplit();
const testSize = Math.max(1, Math.floor(prepared.dataset.X.length * TEST_FRACTION));
const trainSize = prepared.dataset.X.length - testSize;

const [bunRegression, sklearnRegression, bunClassification, sklearnClassification] =
  await Promise.all([
    runBunRegressionBenchmark(iterations, warmup),
    runPythonBenchmark<RegressionBenchmarkResult>("bench/python/heart_sklearn_bench.py", [
      "--dataset",
      DATASET_PATH,
      "--test-size",
      String(TEST_FRACTION),
      "--random-state",
      String(RANDOM_STATE),
      "--iterations",
      String(iterations),
      "--warmup",
      String(warmup),
    ]),
    runBunClassificationBenchmark(iterations, warmup),
    runPythonBenchmark<ClassificationBenchmarkResult>(
      "bench/python/heart_sklearn_classification_bench.py",
      [
        "--dataset",
        DATASET_PATH,
        "--test-size",
        String(TEST_FRACTION),
        "--random-state",
        String(RANDOM_STATE),
        "--iterations",
        String(iterations),
        "--warmup",
        String(warmup),
      ],
    ),
  ]);

const snapshot: BenchmarkSnapshot = {
  generatedAt: new Date().toISOString(),
  benchmarkConfig: {
    iterations,
    warmup,
  },
  dataset: {
    path: DATASET_PATH,
    samples: prepared.dataset.X.length,
    features: prepared.dataset.featureNames.length,
    trainSize,
    testSize,
    randomState: RANDOM_STATE,
    testFraction: TEST_FRACTION,
  },
  suites: {
    regression: {
      task: "regression",
      results: [bunRegression, sklearnRegression],
      comparison: {
        fitSpeedupVsSklearn: sklearnRegression.fitMsMedian / bunRegression.fitMsMedian,
        predictSpeedupVsSklearn:
          sklearnRegression.predictMsMedian / bunRegression.predictMsMedian,
        mseDeltaVsSklearn: bunRegression.mse - sklearnRegression.mse,
        r2DeltaVsSklearn: bunRegression.r2 - sklearnRegression.r2,
      },
    },
    classification: {
      task: "classification",
      results: [bunClassification, sklearnClassification],
      comparison: {
        fitSpeedupVsSklearn:
          sklearnClassification.fitMsMedian / bunClassification.fitMsMedian,
        predictSpeedupVsSklearn:
          sklearnClassification.predictMsMedian / bunClassification.predictMsMedian,
        accuracyDeltaVsSklearn:
          bunClassification.accuracy - sklearnClassification.accuracy,
        f1DeltaVsSklearn: bunClassification.f1 - sklearnClassification.f1,
      },
    },
  },
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(snapshot, null, 2)}\n`, "utf-8");
await writeFile(outputMarkdownPath, `${renderMarkdownTable(snapshot)}\n`, "utf-8");

console.log(`Wrote benchmark snapshot: ${outputPath}`);
console.log(`Wrote benchmark table: ${outputMarkdownPath}`);
console.log(`Regression Bun fit median: ${formatMs(bunRegression.fitMsMedian)}`);
console.log(`Regression sklearn fit median: ${formatMs(sklearnRegression.fitMsMedian)}`);
console.log(`Classification Bun fit median: ${formatMs(bunClassification.fitMsMedian)}`);
console.log(
  `Classification sklearn fit median: ${formatMs(sklearnClassification.fitMsMedian)}`,
);
