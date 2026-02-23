import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import {
  DecisionTreeClassifier,
  LinearRegression,
  LogisticRegression,
  type Matrix,
  RandomForestClassifier,
  StandardScaler,
  type Vector,
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

type TreeModelKey = "decision_tree" | "random_forest";
type TreeBackendMode = "js-fast" | "zig-tree";

interface TreeClassificationModelResult extends ClassificationBenchmarkResult {
  key: TreeModelKey;
  treeBackendMode: TreeBackendMode;
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

interface TreeModelComparison {
  key: TreeModelKey;
  bun: TreeClassificationModelResult;
  sklearn: TreeClassificationModelResult;
  comparison: {
    fitSpeedupVsSklearn: number;
    predictSpeedupVsSklearn: number;
    accuracyDeltaVsSklearn: number;
    f1DeltaVsSklearn: number;
  };
}

interface TreeClassificationSuite {
  task: "classification-tree";
  models: [TreeModelComparison, TreeModelComparison];
}

interface TreeBackendModeComparison {
  key: TreeModelKey;
  jsFast: TreeClassificationModelResult;
  zigTree: TreeClassificationModelResult;
  sklearn: TreeClassificationModelResult;
  comparison: {
    zigFitSpeedupVsJs: number;
    zigPredictSpeedupVsJs: number;
    jsFitSpeedupVsSklearn: number;
    jsPredictSpeedupVsSklearn: number;
    zigFitSpeedupVsSklearn: number;
    zigPredictSpeedupVsSklearn: number;
  };
}

interface TreeBackendModesSuite {
  task: "classification-tree-backend-modes";
  enabled: boolean;
  models: [TreeBackendModeComparison, TreeBackendModeComparison] | [];
}

interface PythonTreeBenchPayload {
  implementation: string;
  iterations: number;
  environment: Record<string, string | number>;
  models: Array<{
    key: TreeModelKey;
    model: string;
    fitMsMedian: number;
    predictMsMedian: number;
    accuracy: number;
    f1: number;
  }>;
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
    treeClassification: TreeClassificationSuite;
    treeBackendModes: TreeBackendModesSuite;
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

function isTruthy(value: string | undefined): boolean {
  if (!value) {
    return false;
  }
  const normalized = value.trim().toLowerCase();
  return !(normalized === "0" || normalized === "false" || normalized === "off");
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
  let backendLabel: "js" | "zig" = "js";
  let backendLibrary: string | null = null;
  const solverLabel: "gd" | "lbfgs" = "gd";

  const loops = warmup + iterations;
  for (let i = 0; i < loops; i += 1) {
    const model = new LogisticRegression({
      solver: solverLabel,
      learningRate: 0.8,
      maxIter: 20,
      tolerance: 1e-4,
      l2: 0.01,
    });

    const fitStart = performance.now();
    model.fit(XTrainScaled, split.yTrain);
    const fitMs = performance.now() - fitStart;

    const predictStart = performance.now();
    const predictions = model.predict(XTestScaled);
    const predictMs = performance.now() - predictStart;
    backendLabel = model.fitBackend_;
    backendLibrary = model.fitBackendLibrary_;

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
    model: `StandardScaler + LogisticRegression(${solverLabel},${backendLabel})`,
    iterations,
    fitMsMedian: median(fitTimes),
    predictMsMedian: median(predictTimes),
    accuracy: accuracyScore(split.yTest, predictionsForMetrics),
    f1: f1Score(split.yTest, predictionsForMetrics),
    environment: {
      bun: Bun.version,
      runtime: "bun",
      fitBackend: backendLabel,
      fitBackendLibrary: backendLibrary ?? "none",
    },
  };
}

async function runBunTreeModelBenchmark(
  key: TreeModelKey,
  modelLabel: string,
  treeBackendMode: TreeBackendMode,
  iterations: number,
  warmup: number,
  modelFactory: () => {
    fit(X: Matrix, y: Vector): unknown;
    predict(X: Matrix): Vector;
    dispose?: () => void;
  },
): Promise<TreeClassificationModelResult> {
  const previousTreeBackend = process.env.BUN_SCIKIT_TREE_BACKEND;
  if (treeBackendMode === "zig-tree") {
    process.env.BUN_SCIKIT_TREE_BACKEND = "zig";
  } else {
    delete process.env.BUN_SCIKIT_TREE_BACKEND;
  }

  const { split, XTrainScaled, XTestScaled } = await prepareHeartSplit();
  const fitTimes: number[] = [];
  const predictTimes: number[] = [];
  let predictionsForMetrics: number[] | null = null;

  try {
    const loops = warmup + iterations;
    for (let i = 0; i < loops; i += 1) {
      const model = modelFactory();
      let fitMs = 0;
      let predictMs = 0;
      let predictions: Vector = [];
      try {
        const fitStart = performance.now();
        model.fit(XTrainScaled, split.yTrain);
        fitMs = performance.now() - fitStart;

        const predictStart = performance.now();
        predictions = model.predict(XTestScaled);
        predictMs = performance.now() - predictStart;
      } finally {
        model.dispose?.();
      }

      if (i >= warmup) {
        fitTimes.push(fitMs);
        predictTimes.push(predictMs);
        predictionsForMetrics = predictions;
      }
    }
  } finally {
    if (previousTreeBackend === undefined) {
      delete process.env.BUN_SCIKIT_TREE_BACKEND;
    } else {
      process.env.BUN_SCIKIT_TREE_BACKEND = previousTreeBackend;
    }
  }

  if (!predictionsForMetrics) {
    throw new Error(`No Bun tree benchmark iterations were recorded for ${key} (${treeBackendMode}).`);
  }

  return {
    key,
    treeBackendMode,
    implementation: "bun-scikit",
    model: `${modelLabel} [${treeBackendMode}]`,
    iterations,
    fitMsMedian: median(fitTimes),
    predictMsMedian: median(predictTimes),
    accuracy: accuracyScore(split.yTest, predictionsForMetrics),
    f1: f1Score(split.yTest, predictionsForMetrics),
    environment: {
      bun: Bun.version,
      runtime: "bun",
      treeBackendMode,
    },
  };
}

async function runBunTreeBenchmarksForMode(
  treeBackendMode: TreeBackendMode,
  iterations: number,
  warmup: number,
): Promise<[TreeClassificationModelResult, TreeClassificationModelResult]> {
  const decisionTree = await runBunTreeModelBenchmark(
    "decision_tree",
    "DecisionTreeClassifier(maxDepth=8)",
    treeBackendMode,
    iterations,
    warmup,
    () =>
      new DecisionTreeClassifier({
        maxDepth: 8,
        minSamplesLeaf: 3,
        randomState: 42,
      }),
  );
  const randomForest = await runBunTreeModelBenchmark(
    "random_forest",
    "RandomForestClassifier(nEstimators=80,maxDepth=8)",
    treeBackendMode,
    iterations,
    warmup,
    () =>
      new RandomForestClassifier({
        nEstimators: 80,
        maxDepth: 8,
        minSamplesLeaf: 2,
        randomState: 42,
      }),
  );

  return [decisionTree, randomForest];
}

function renderMarkdownTable(snapshot: BenchmarkSnapshot): string {
  const regression = snapshot.suites.regression;
  const classification = snapshot.suites.classification;
  const treeClassification = snapshot.suites.treeClassification;
  const [bunReg, sklearnReg] = regression.results;
  const [bunCls, sklearnCls] = classification.results;
  const [decisionTree, randomForest] = treeClassification.models;
  const treeBackendModes = snapshot.suites.treeBackendModes;
  const hasTreeBackendModes = treeBackendModes.enabled && treeBackendModes.models.length === 2;
  const decisionTreeModes = hasTreeBackendModes ? treeBackendModes.models[0] : null;
  const randomForestModes = hasTreeBackendModes ? treeBackendModes.models[1] : null;

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
    "## Tree Classification (Heart Dataset)",
    "",
    "| Model | Implementation | Fit median (ms) | Predict median (ms) | Accuracy | F1 |",
    "|---|---|---:|---:|---:|---:|",
    `| ${decisionTree.bun.model} | ${decisionTree.bun.implementation} | ${decisionTree.bun.fitMsMedian.toFixed(4)} | ${decisionTree.bun.predictMsMedian.toFixed(4)} | ${decisionTree.bun.accuracy.toFixed(6)} | ${decisionTree.bun.f1.toFixed(6)} |`,
    `| ${decisionTree.sklearn.model} | ${decisionTree.sklearn.implementation} | ${decisionTree.sklearn.fitMsMedian.toFixed(4)} | ${decisionTree.sklearn.predictMsMedian.toFixed(4)} | ${decisionTree.sklearn.accuracy.toFixed(6)} | ${decisionTree.sklearn.f1.toFixed(6)} |`,
    `| ${randomForest.bun.model} | ${randomForest.bun.implementation} | ${randomForest.bun.fitMsMedian.toFixed(4)} | ${randomForest.bun.predictMsMedian.toFixed(4)} | ${randomForest.bun.accuracy.toFixed(6)} | ${randomForest.bun.f1.toFixed(6)} |`,
    `| ${randomForest.sklearn.model} | ${randomForest.sklearn.implementation} | ${randomForest.sklearn.fitMsMedian.toFixed(4)} | ${randomForest.sklearn.predictMsMedian.toFixed(4)} | ${randomForest.sklearn.accuracy.toFixed(6)} | ${randomForest.sklearn.f1.toFixed(6)} |`,
    "",
    `DecisionTree fit speedup vs scikit-learn: ${decisionTree.comparison.fitSpeedupVsSklearn.toFixed(3)}x`,
    `DecisionTree predict speedup vs scikit-learn: ${decisionTree.comparison.predictSpeedupVsSklearn.toFixed(3)}x`,
    `DecisionTree accuracy delta (bun - sklearn): ${decisionTree.comparison.accuracyDeltaVsSklearn.toExponential(3)}`,
    `DecisionTree f1 delta (bun - sklearn): ${decisionTree.comparison.f1DeltaVsSklearn.toExponential(3)}`,
    "",
    `RandomForest fit speedup vs scikit-learn: ${randomForest.comparison.fitSpeedupVsSklearn.toFixed(3)}x`,
    `RandomForest predict speedup vs scikit-learn: ${randomForest.comparison.predictSpeedupVsSklearn.toFixed(3)}x`,
    `RandomForest accuracy delta (bun - sklearn): ${randomForest.comparison.accuracyDeltaVsSklearn.toExponential(3)}`,
    `RandomForest f1 delta (bun - sklearn): ${randomForest.comparison.f1DeltaVsSklearn.toExponential(3)}`,
    "",
    "## Tree Backend Modes (Bun vs Bun vs sklearn)",
    "",
    hasTreeBackendModes
      ? "| Model | Backend | Fit median (ms) | Predict median (ms) | Accuracy | F1 |"
      : "Tree backend mode matrix disabled (`BENCH_TREE_BACKEND_MATRIX=0`).",
    hasTreeBackendModes ? "|---|---|---:|---:|---:|---:|" : "",
    hasTreeBackendModes
      ? `| DecisionTreeClassifier(maxDepth=8) | js-fast | ${decisionTreeModes!.jsFast.fitMsMedian.toFixed(4)} | ${decisionTreeModes!.jsFast.predictMsMedian.toFixed(4)} | ${decisionTreeModes!.jsFast.accuracy.toFixed(6)} | ${decisionTreeModes!.jsFast.f1.toFixed(6)} |`
      : "",
    hasTreeBackendModes
      ? `| DecisionTreeClassifier(maxDepth=8) | zig-tree | ${decisionTreeModes!.zigTree.fitMsMedian.toFixed(4)} | ${decisionTreeModes!.zigTree.predictMsMedian.toFixed(4)} | ${decisionTreeModes!.zigTree.accuracy.toFixed(6)} | ${decisionTreeModes!.zigTree.f1.toFixed(6)} |`
      : "",
    hasTreeBackendModes
      ? `| DecisionTreeClassifier | python-scikit-learn | ${decisionTreeModes!.sklearn.fitMsMedian.toFixed(4)} | ${decisionTreeModes!.sklearn.predictMsMedian.toFixed(4)} | ${decisionTreeModes!.sklearn.accuracy.toFixed(6)} | ${decisionTreeModes!.sklearn.f1.toFixed(6)} |`
      : "",
    hasTreeBackendModes
      ? `| RandomForestClassifier(nEstimators=80,maxDepth=8) | js-fast | ${randomForestModes!.jsFast.fitMsMedian.toFixed(4)} | ${randomForestModes!.jsFast.predictMsMedian.toFixed(4)} | ${randomForestModes!.jsFast.accuracy.toFixed(6)} | ${randomForestModes!.jsFast.f1.toFixed(6)} |`
      : "",
    hasTreeBackendModes
      ? `| RandomForestClassifier(nEstimators=80,maxDepth=8) | zig-tree | ${randomForestModes!.zigTree.fitMsMedian.toFixed(4)} | ${randomForestModes!.zigTree.predictMsMedian.toFixed(4)} | ${randomForestModes!.zigTree.accuracy.toFixed(6)} | ${randomForestModes!.zigTree.f1.toFixed(6)} |`
      : "",
    hasTreeBackendModes
      ? `| RandomForestClassifier | python-scikit-learn | ${randomForestModes!.sklearn.fitMsMedian.toFixed(4)} | ${randomForestModes!.sklearn.predictMsMedian.toFixed(4)} | ${randomForestModes!.sklearn.accuracy.toFixed(6)} | ${randomForestModes!.sklearn.f1.toFixed(6)} |`
      : "",
    "",
    hasTreeBackendModes
      ? `DecisionTree zig/js fit speedup: ${decisionTreeModes!.comparison.zigFitSpeedupVsJs.toFixed(3)}x`
      : "",
    hasTreeBackendModes
      ? `DecisionTree zig/js predict speedup: ${decisionTreeModes!.comparison.zigPredictSpeedupVsJs.toFixed(3)}x`
      : "",
    hasTreeBackendModes
      ? `RandomForest zig/js fit speedup: ${randomForestModes!.comparison.zigFitSpeedupVsJs.toFixed(3)}x`
      : "",
    hasTreeBackendModes
      ? `RandomForest zig/js predict speedup: ${randomForestModes!.comparison.zigPredictSpeedupVsJs.toFixed(3)}x`
      : "",
    "",
    `Snapshot generated at: ${snapshot.generatedAt}`,
  ]
    .filter((line) => line !== "")
    .join("\n");
}

async function main(): Promise<void> {
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

  const bunRegression = await runBunRegressionBenchmark(iterations, warmup);
  const bunClassification = await runBunClassificationBenchmark(iterations, warmup);
  const bunTreeBenchmarksJsFast = await runBunTreeBenchmarksForMode(
    "js-fast",
    iterations,
    warmup,
  );
  const treeBackendMatrixEnabled = isTruthy(process.env.BENCH_TREE_BACKEND_MATRIX ?? "1");
  const bunTreeBenchmarksZig = treeBackendMatrixEnabled
    ? await runBunTreeBenchmarksForMode("zig-tree", iterations, warmup)
    : null;

  const sklearnRegression = await runPythonBenchmark<RegressionBenchmarkResult>(
    "bench/python/heart_sklearn_bench.py",
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
  );
  const sklearnClassification = await runPythonBenchmark<ClassificationBenchmarkResult>(
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
  );
  const sklearnTreeBenchmarks = await runPythonBenchmark<PythonTreeBenchPayload>(
    "bench/python/heart_sklearn_tree_bench.py",
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
  );

  const [bunDecisionTree, bunRandomForest] = bunTreeBenchmarksJsFast;
  const sklearnDecisionTreeRaw = sklearnTreeBenchmarks.models.find(
    (model) => model.key === "decision_tree",
  );
  const sklearnRandomForestRaw = sklearnTreeBenchmarks.models.find(
    (model) => model.key === "random_forest",
  );

  if (!sklearnDecisionTreeRaw || !sklearnRandomForestRaw) {
    throw new Error("Python tree benchmark output is missing expected models.");
  }

  const sklearnDecisionTree: TreeClassificationModelResult = {
    key: "decision_tree",
    treeBackendMode: "js-fast",
    implementation: sklearnTreeBenchmarks.implementation,
    model: sklearnDecisionTreeRaw.model,
    iterations: sklearnTreeBenchmarks.iterations,
    fitMsMedian: sklearnDecisionTreeRaw.fitMsMedian,
    predictMsMedian: sklearnDecisionTreeRaw.predictMsMedian,
    accuracy: sklearnDecisionTreeRaw.accuracy,
    f1: sklearnDecisionTreeRaw.f1,
    environment: sklearnTreeBenchmarks.environment,
  };

  const sklearnRandomForest: TreeClassificationModelResult = {
    key: "random_forest",
    treeBackendMode: "js-fast",
    implementation: sklearnTreeBenchmarks.implementation,
    model: sklearnRandomForestRaw.model,
    iterations: sklearnTreeBenchmarks.iterations,
    fitMsMedian: sklearnRandomForestRaw.fitMsMedian,
    predictMsMedian: sklearnRandomForestRaw.predictMsMedian,
    accuracy: sklearnRandomForestRaw.accuracy,
    f1: sklearnRandomForestRaw.f1,
    environment: sklearnTreeBenchmarks.environment,
  };

  const treeBackendModes: TreeBackendModesSuite = treeBackendMatrixEnabled && bunTreeBenchmarksZig
    ? {
        task: "classification-tree-backend-modes",
        enabled: true,
        models: [
          {
            key: "decision_tree",
            jsFast: bunTreeBenchmarksJsFast[0],
            zigTree: bunTreeBenchmarksZig[0],
            sklearn: sklearnDecisionTree,
            comparison: {
              zigFitSpeedupVsJs:
                bunTreeBenchmarksJsFast[0].fitMsMedian / bunTreeBenchmarksZig[0].fitMsMedian,
              zigPredictSpeedupVsJs:
                bunTreeBenchmarksJsFast[0].predictMsMedian /
                bunTreeBenchmarksZig[0].predictMsMedian,
              jsFitSpeedupVsSklearn:
                sklearnDecisionTree.fitMsMedian / bunTreeBenchmarksJsFast[0].fitMsMedian,
              jsPredictSpeedupVsSklearn:
                sklearnDecisionTree.predictMsMedian / bunTreeBenchmarksJsFast[0].predictMsMedian,
              zigFitSpeedupVsSklearn:
                sklearnDecisionTree.fitMsMedian / bunTreeBenchmarksZig[0].fitMsMedian,
              zigPredictSpeedupVsSklearn:
                sklearnDecisionTree.predictMsMedian / bunTreeBenchmarksZig[0].predictMsMedian,
            },
          },
          {
            key: "random_forest",
            jsFast: bunTreeBenchmarksJsFast[1],
            zigTree: bunTreeBenchmarksZig[1],
            sklearn: sklearnRandomForest,
            comparison: {
              zigFitSpeedupVsJs:
                bunTreeBenchmarksJsFast[1].fitMsMedian / bunTreeBenchmarksZig[1].fitMsMedian,
              zigPredictSpeedupVsJs:
                bunTreeBenchmarksJsFast[1].predictMsMedian /
                bunTreeBenchmarksZig[1].predictMsMedian,
              jsFitSpeedupVsSklearn:
                sklearnRandomForest.fitMsMedian / bunTreeBenchmarksJsFast[1].fitMsMedian,
              jsPredictSpeedupVsSklearn:
                sklearnRandomForest.predictMsMedian /
                bunTreeBenchmarksJsFast[1].predictMsMedian,
              zigFitSpeedupVsSklearn:
                sklearnRandomForest.fitMsMedian / bunTreeBenchmarksZig[1].fitMsMedian,
              zigPredictSpeedupVsSklearn:
                sklearnRandomForest.predictMsMedian / bunTreeBenchmarksZig[1].predictMsMedian,
            },
          },
        ],
      }
    : {
        task: "classification-tree-backend-modes",
        enabled: false,
        models: [],
      };

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
      treeClassification: {
        task: "classification-tree",
        models: [
          {
            key: "decision_tree",
            bun: bunDecisionTree,
            sklearn: sklearnDecisionTree,
            comparison: {
              fitSpeedupVsSklearn:
                sklearnDecisionTree.fitMsMedian / bunDecisionTree.fitMsMedian,
              predictSpeedupVsSklearn:
                sklearnDecisionTree.predictMsMedian / bunDecisionTree.predictMsMedian,
              accuracyDeltaVsSklearn: bunDecisionTree.accuracy - sklearnDecisionTree.accuracy,
              f1DeltaVsSklearn: bunDecisionTree.f1 - sklearnDecisionTree.f1,
            },
          },
          {
            key: "random_forest",
            bun: bunRandomForest,
            sklearn: sklearnRandomForest,
            comparison: {
              fitSpeedupVsSklearn:
                sklearnRandomForest.fitMsMedian / bunRandomForest.fitMsMedian,
              predictSpeedupVsSklearn:
                sklearnRandomForest.predictMsMedian / bunRandomForest.predictMsMedian,
              accuracyDeltaVsSklearn:
                bunRandomForest.accuracy - sklearnRandomForest.accuracy,
              f1DeltaVsSklearn: bunRandomForest.f1 - sklearnRandomForest.f1,
            },
          },
        ],
      },
      treeBackendModes,
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
  console.log(`DecisionTree Bun fit median: ${formatMs(bunDecisionTree.fitMsMedian)}`);
  console.log(`DecisionTree sklearn fit median: ${formatMs(sklearnDecisionTree.fitMsMedian)}`);
  console.log(`RandomForest Bun fit median: ${formatMs(bunRandomForest.fitMsMedian)}`);
  console.log(`RandomForest sklearn fit median: ${formatMs(sklearnRandomForest.fitMsMedian)}`);
  if (treeBackendModes.enabled) {
    const [decisionTreeModeComparison, randomForestModeComparison] = treeBackendModes.models;
    if (!decisionTreeModeComparison || !randomForestModeComparison) {
      throw new Error("treeBackendModes is enabled but missing comparison entries.");
    }
    console.log(
      `DecisionTree zig/js fit speedup: ${decisionTreeModeComparison.comparison.zigFitSpeedupVsJs.toFixed(3)}x`,
    );
    console.log(
      `RandomForest zig/js fit speedup: ${randomForestModeComparison.comparison.zigFitSpeedupVsJs.toFixed(3)}x`,
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
