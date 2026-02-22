import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import {
  LinearRegression,
  StandardScaler,
  meanSquaredError,
  r2Score,
  trainTestSplit,
} from "../src";
import { loadHeartDataset } from "../test_data/loadHeartDataset";

interface BenchmarkResult {
  implementation: string;
  model: string;
  iterations: number;
  fitMsMedian: number;
  predictMsMedian: number;
  mse: number;
  r2: number;
  environment: Record<string, string | number>;
}

interface BenchmarkSnapshot {
  generatedAt: string;
  dataset: {
    path: string;
    samples: number;
    features: number;
    trainSize: number;
    testSize: number;
    randomState: number;
    testFraction: number;
  };
  results: [BenchmarkResult, BenchmarkResult];
  comparison: {
    fitSpeedupVsSklearn: number;
    predictSpeedupVsSklearn: number;
    mseDeltaVsSklearn: number;
    r2DeltaVsSklearn: number;
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

async function runBunBenchmark(iterations: number, warmup: number): Promise<BenchmarkResult> {
  const { X, y } = await loadHeartDataset();
  const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, {
    testSize: TEST_FRACTION,
    randomState: RANDOM_STATE,
    shuffle: true,
  });

  const scaler = new StandardScaler();
  const XTrainScaled = scaler.fitTransform(XTrain);
  const XTestScaled = scaler.transform(XTest);

  const fitTimes: number[] = [];
  const predictTimes: number[] = [];
  let predictionsForMetrics: number[] | null = null;

  const loops = warmup + iterations;
  for (let i = 0; i < loops; i += 1) {
    const model = new LinearRegression({ solver: "normal" });

    const fitStart = performance.now();
    model.fit(XTrainScaled, yTrain);
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
    throw new Error("No Bun benchmark iterations were recorded.");
  }

  return {
    implementation: "bun-scikit",
    model: "StandardScaler + LinearRegression(normal)",
    iterations,
    fitMsMedian: median(fitTimes),
    predictMsMedian: median(predictTimes),
    mse: meanSquaredError(yTest, predictionsForMetrics),
    r2: r2Score(yTest, predictionsForMetrics),
    environment: {
      bun: Bun.version,
      runtime: "bun",
    },
  };
}

async function runPythonBenchmark(iterations: number, warmup: number): Promise<BenchmarkResult> {
  const process = Bun.spawn(
    [
      "python",
      "bench/python/heart_sklearn_bench.py",
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
    {
      stdout: "pipe",
      stderr: "pipe",
    },
  );

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
    process.exited,
  ]);

  if (exitCode !== 0) {
    throw new Error(
      `Python benchmark failed with exit code ${exitCode}.\n${stderr || "(no stderr)"}`,
    );
  }

  return JSON.parse(stdout) as BenchmarkResult;
}

function renderMarkdownTable(snapshot: BenchmarkSnapshot): string {
  const [bunResult, sklearnResult] = snapshot.results;
  return [
    "| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |",
    "|---|---|---:|---:|---:|---:|",
    `| ${bunResult.implementation} | ${bunResult.model} | ${bunResult.fitMsMedian.toFixed(4)} | ${bunResult.predictMsMedian.toFixed(4)} | ${bunResult.mse.toFixed(6)} | ${bunResult.r2.toFixed(6)} |`,
    `| ${sklearnResult.implementation} | ${sklearnResult.model} | ${sklearnResult.fitMsMedian.toFixed(4)} | ${sklearnResult.predictMsMedian.toFixed(4)} | ${sklearnResult.mse.toFixed(6)} | ${sklearnResult.r2.toFixed(6)} |`,
    "",
    `Bun fit speedup vs scikit-learn: ${snapshot.comparison.fitSpeedupVsSklearn.toFixed(3)}x`,
    `Bun predict speedup vs scikit-learn: ${snapshot.comparison.predictSpeedupVsSklearn.toFixed(3)}x`,
    `MSE delta (bun - sklearn): ${snapshot.comparison.mseDeltaVsSklearn.toExponential(3)}`,
    `R2 delta (bun - sklearn): ${snapshot.comparison.r2DeltaVsSklearn.toExponential(3)}`,
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

const dataset = await loadHeartDataset();
const testSize = Math.max(1, Math.floor(dataset.X.length * TEST_FRACTION));
const trainSize = dataset.X.length - testSize;

const bunResult = await runBunBenchmark(iterations, warmup);
const sklearnResult = await runPythonBenchmark(iterations, warmup);

const snapshot: BenchmarkSnapshot = {
  generatedAt: new Date().toISOString(),
  dataset: {
    path: DATASET_PATH,
    samples: dataset.X.length,
    features: dataset.featureNames.length,
    trainSize,
    testSize,
    randomState: RANDOM_STATE,
    testFraction: TEST_FRACTION,
  },
  results: [bunResult, sklearnResult],
  comparison: {
    fitSpeedupVsSklearn: sklearnResult.fitMsMedian / bunResult.fitMsMedian,
    predictSpeedupVsSklearn: sklearnResult.predictMsMedian / bunResult.predictMsMedian,
    mseDeltaVsSklearn: bunResult.mse - sklearnResult.mse,
    r2DeltaVsSklearn: bunResult.r2 - sklearnResult.r2,
  },
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(snapshot, null, 2)}\n`, "utf-8");
await writeFile(outputMarkdownPath, `${renderMarkdownTable(snapshot)}\n`, "utf-8");

console.log(`Wrote benchmark snapshot: ${outputPath}`);
console.log(`Wrote benchmark table: ${outputMarkdownPath}`);
console.log(`Bun fit median: ${formatMs(bunResult.fitMsMedian)}`);
console.log(`sklearn fit median: ${formatMs(sklearnResult.fitMsMedian)}`);
