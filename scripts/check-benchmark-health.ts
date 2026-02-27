import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

interface SharedBenchmarkResult {
  implementation: string;
  fitMsMedian: number;
  predictMsMedian: number;
}

interface RegressionBenchmarkResult extends SharedBenchmarkResult {
  mse: number;
  r2: number;
}

interface ClassificationBenchmarkResult extends SharedBenchmarkResult {
  accuracy: number;
  f1: number;
}

interface TreeModelComparison {
  bun: ClassificationBenchmarkResult;
  sklearn: ClassificationBenchmarkResult;
  comparison: {
    accuracyDeltaVsSklearn: number;
    f1DeltaVsSklearn: number;
  };
}

interface TreeBackendModeComparison {
  jsFast: ClassificationBenchmarkResult;
  zigTree: ClassificationBenchmarkResult;
  comparison: {
    zigFitSpeedupVsJs: number;
    zigPredictSpeedupVsJs: number;
    jsFitSpeedupVsSklearn: number;
    jsPredictSpeedupVsSklearn: number;
    zigFitSpeedupVsSklearn: number;
    zigPredictSpeedupVsSklearn: number;
  };
}

interface BenchmarkSnapshot {
  suites: {
    regression: {
      results: [RegressionBenchmarkResult, RegressionBenchmarkResult];
      comparison: {
        fitSpeedupVsSklearn: number;
        predictSpeedupVsSklearn: number;
        mseDeltaVsSklearn: number;
        r2DeltaVsSklearn: number;
      };
    };
    classification: {
      results: [ClassificationBenchmarkResult, ClassificationBenchmarkResult];
      comparison: {
        fitSpeedupVsSklearn: number;
        predictSpeedupVsSklearn: number;
        accuracyDeltaVsSklearn: number;
        f1DeltaVsSklearn: number;
      };
    };
    treeClassification: {
      models: [
        TreeModelComparison & {
          comparison: TreeModelComparison["comparison"] & {
            fitSpeedupVsSklearn: number;
            predictSpeedupVsSklearn: number;
          };
        },
        TreeModelComparison & {
          comparison: TreeModelComparison["comparison"] & {
            fitSpeedupVsSklearn: number;
            predictSpeedupVsSklearn: number;
          };
        },
      ];
    };
    treeBackendModes: {
      enabled: boolean;
      models: [TreeBackendModeComparison, TreeBackendModeComparison] | [];
    };
  };
}

function parseArgValue(flag: string): string | null {
  const index = Bun.argv.indexOf(flag);
  if (index === -1 || index + 1 >= Bun.argv.length) {
    return null;
  }
  return Bun.argv[index + 1];
}

function speedupThreshold(
  envName: string,
  defaultValue: number,
): number {
  const raw = process.env[envName];
  if (!raw) {
    return defaultValue;
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${envName} must be a positive number when set. Got: ${raw}`);
  }
  return parsed;
}

const inputPath = resolve(parseArgValue("--input") ?? "bench/results/heart-ci-current.json");
const baselinePath = resolve(
  parseArgValue("--baseline") ?? process.env.BENCH_BASELINE_INPUT ?? "bench/results/heart-ci-latest.json",
);
const baselineInputEnabled = inputPath !== baselinePath;

const snapshot = JSON.parse(await readFile(inputPath, "utf-8")) as BenchmarkSnapshot;
const baselineSnapshot = baselineInputEnabled
  ? ((await readFile(baselinePath, "utf-8").then((raw) => JSON.parse(raw) as BenchmarkSnapshot).catch(
      () => null,
    )) as BenchmarkSnapshot | null)
  : null;

const [bunRegression, sklearnRegression] = snapshot.suites.regression.results;
const [bunClassification, sklearnClassification] = snapshot.suites.classification.results;
const [decisionTree, randomForest] = snapshot.suites.treeClassification.models;
const minRegressionFitSpeedup = speedupThreshold("BENCH_MIN_REGRESSION_FIT_SPEEDUP", 1.1);
const minRegressionPredictSpeedup = speedupThreshold("BENCH_MIN_REGRESSION_PREDICT_SPEEDUP", 1.1);
const minClassificationFitSpeedup = speedupThreshold("BENCH_MIN_CLASSIFICATION_FIT_SPEEDUP", 1.2);
const minClassificationPredictSpeedup = speedupThreshold(
  "BENCH_MIN_CLASSIFICATION_PREDICT_SPEEDUP",
  1.2,
);
const minDecisionTreeFitSpeedup = speedupThreshold("BENCH_MIN_DECISION_TREE_FIT_SPEEDUP", 1.2);
const minDecisionTreePredictSpeedup = speedupThreshold(
  "BENCH_MIN_DECISION_TREE_PREDICT_SPEEDUP",
  1.5,
);
const minRandomForestFitSpeedup = speedupThreshold("BENCH_MIN_RANDOM_FOREST_FIT_SPEEDUP", 2.0);
const minRandomForestPredictSpeedup = speedupThreshold(
  "BENCH_MIN_RANDOM_FOREST_PREDICT_SPEEDUP",
  1.2,
);
const maxZigTreeFitSlowdownVsJs = speedupThreshold("BENCH_MAX_ZIG_TREE_FIT_SLOWDOWN_VS_JS", 4);
const maxZigTreePredictSlowdownVsJs = speedupThreshold(
  "BENCH_MAX_ZIG_TREE_PREDICT_SLOWDOWN_VS_JS",
  4,
);
const maxZigForestFitSlowdownVsJs = speedupThreshold(
  "BENCH_MAX_ZIG_FOREST_FIT_SLOWDOWN_VS_JS",
  4,
);
const maxZigForestPredictSlowdownVsJs = speedupThreshold(
  "BENCH_MAX_ZIG_FOREST_PREDICT_SLOWDOWN_VS_JS",
  4,
);
const maxZigTreeAccuracyDropVsJs = speedupThreshold(
  "BENCH_MAX_ZIG_TREE_ACCURACY_DROP_VS_JS",
  0.06,
);
const maxZigTreeF1DropVsJs = speedupThreshold(
  "BENCH_MAX_ZIG_TREE_F1_DROP_VS_JS",
  0.06,
);
const maxZigForestAccuracyDropVsJs = speedupThreshold(
  "BENCH_MAX_ZIG_FOREST_ACCURACY_DROP_VS_JS",
  0.03,
);
const maxZigForestF1DropVsJs = speedupThreshold(
  "BENCH_MAX_ZIG_FOREST_F1_DROP_VS_JS",
  0.03,
);
const minZigTreeFitRetentionVsBaseline = speedupThreshold(
  "BENCH_MIN_ZIG_TREE_FIT_RETENTION_VS_BASELINE",
  0.35,
);
const minZigForestFitRetentionVsBaseline = speedupThreshold(
  "BENCH_MIN_ZIG_FOREST_FIT_RETENTION_VS_BASELINE",
  0.35,
);

for (const result of [
  bunRegression,
  sklearnRegression,
  bunClassification,
  sklearnClassification,
  decisionTree.bun,
  decisionTree.sklearn,
  randomForest.bun,
  randomForest.sklearn,
]) {
  if (!(result.fitMsMedian > 0 && result.predictMsMedian > 0)) {
    throw new Error(`Benchmark timings must be positive for ${result.implementation}.`);
  }
}

if (
  !Number.isFinite(bunRegression.mse) ||
  !Number.isFinite(sklearnRegression.mse) ||
  !Number.isFinite(bunRegression.r2) ||
  !Number.isFinite(sklearnRegression.r2)
) {
  throw new Error("Regression metrics must be finite for both implementations.");
}

if (
  !Number.isFinite(bunClassification.accuracy) ||
  !Number.isFinite(sklearnClassification.accuracy) ||
  !Number.isFinite(bunClassification.f1) ||
  !Number.isFinite(sklearnClassification.f1)
) {
  throw new Error("Classification metrics must be finite for both implementations.");
}

if (Math.abs(snapshot.suites.regression.comparison.mseDeltaVsSklearn) > 0.01) {
  throw new Error(
    `Regression MSE delta too large: ${snapshot.suites.regression.comparison.mseDeltaVsSklearn}.`,
  );
}

if (snapshot.suites.regression.comparison.fitSpeedupVsSklearn < minRegressionFitSpeedup) {
  throw new Error(
    `Regression fit speedup too low: ${snapshot.suites.regression.comparison.fitSpeedupVsSklearn} < ${minRegressionFitSpeedup}.`,
  );
}

if (snapshot.suites.regression.comparison.predictSpeedupVsSklearn < minRegressionPredictSpeedup) {
  throw new Error(
    `Regression predict speedup too low: ${snapshot.suites.regression.comparison.predictSpeedupVsSklearn} < ${minRegressionPredictSpeedup}.`,
  );
}

if (Math.abs(snapshot.suites.regression.comparison.r2DeltaVsSklearn) > 0.01) {
  throw new Error(
    `Regression R2 delta too large: ${snapshot.suites.regression.comparison.r2DeltaVsSklearn}.`,
  );
}

if (Math.abs(snapshot.suites.classification.comparison.accuracyDeltaVsSklearn) > 0.05) {
  throw new Error(
    `Classification accuracy delta too large: ${snapshot.suites.classification.comparison.accuracyDeltaVsSklearn}.`,
  );
}

if (Math.abs(snapshot.suites.classification.comparison.f1DeltaVsSklearn) > 0.05) {
  throw new Error(
    `Classification F1 delta too large: ${snapshot.suites.classification.comparison.f1DeltaVsSklearn}.`,
  );
}

if (snapshot.suites.classification.comparison.fitSpeedupVsSklearn < minClassificationFitSpeedup) {
  throw new Error(
    `Classification fit speedup too low: ${snapshot.suites.classification.comparison.fitSpeedupVsSklearn} < ${minClassificationFitSpeedup}.`,
  );
}

if (
  snapshot.suites.classification.comparison.predictSpeedupVsSklearn <
  minClassificationPredictSpeedup
) {
  throw new Error(
    `Classification predict speedup too low: ${snapshot.suites.classification.comparison.predictSpeedupVsSklearn} < ${minClassificationPredictSpeedup}.`,
  );
}

if (Math.abs(decisionTree.comparison.accuracyDeltaVsSklearn) > 0.08) {
  throw new Error(
    `DecisionTree accuracy delta too large: ${decisionTree.comparison.accuracyDeltaVsSklearn}.`,
  );
}

if (Math.abs(decisionTree.comparison.f1DeltaVsSklearn) > 0.08) {
  throw new Error(`DecisionTree F1 delta too large: ${decisionTree.comparison.f1DeltaVsSklearn}.`);
}

if (decisionTree.comparison.fitSpeedupVsSklearn < minDecisionTreeFitSpeedup) {
  throw new Error(
    `DecisionTree fit speedup too low: ${decisionTree.comparison.fitSpeedupVsSklearn} < ${minDecisionTreeFitSpeedup}.`,
  );
}

if (decisionTree.comparison.predictSpeedupVsSklearn < minDecisionTreePredictSpeedup) {
  throw new Error(
    `DecisionTree predict speedup too low: ${decisionTree.comparison.predictSpeedupVsSklearn} < ${minDecisionTreePredictSpeedup}.`,
  );
}

if (Math.abs(randomForest.comparison.accuracyDeltaVsSklearn) > 0.08) {
  throw new Error(
    `RandomForest accuracy delta too large: ${randomForest.comparison.accuracyDeltaVsSklearn}.`,
  );
}

if (Math.abs(randomForest.comparison.f1DeltaVsSklearn) > 0.08) {
  throw new Error(`RandomForest F1 delta too large: ${randomForest.comparison.f1DeltaVsSklearn}.`);
}

if (randomForest.comparison.fitSpeedupVsSklearn < minRandomForestFitSpeedup) {
  throw new Error(
    `RandomForest fit speedup too low: ${randomForest.comparison.fitSpeedupVsSklearn} < ${minRandomForestFitSpeedup}.`,
  );
}

if (randomForest.comparison.predictSpeedupVsSklearn < minRandomForestPredictSpeedup) {
  throw new Error(
    `RandomForest predict speedup too low: ${randomForest.comparison.predictSpeedupVsSklearn} < ${minRandomForestPredictSpeedup}.`,
  );
}

if (snapshot.suites.treeBackendModes.enabled) {
  const [decisionTreeModes, randomForestModes] = snapshot.suites.treeBackendModes.models;
  if (!decisionTreeModes || !randomForestModes) {
    throw new Error("Tree backend mode suite is enabled but missing model comparisons.");
  }

  const decisionTreeFitSlowdown = 1 / decisionTreeModes.comparison.zigFitSpeedupVsJs;
  const decisionTreePredictSlowdown = 1 / decisionTreeModes.comparison.zigPredictSpeedupVsJs;
  const randomForestFitSlowdown = 1 / randomForestModes.comparison.zigFitSpeedupVsJs;
  const randomForestPredictSlowdown = 1 / randomForestModes.comparison.zigPredictSpeedupVsJs;
  const decisionTreeAccuracyDropVsJs = decisionTreeModes.jsFast.accuracy - decisionTreeModes.zigTree.accuracy;
  const decisionTreeF1DropVsJs = decisionTreeModes.jsFast.f1 - decisionTreeModes.zigTree.f1;
  const randomForestAccuracyDropVsJs = randomForestModes.jsFast.accuracy - randomForestModes.zigTree.accuracy;
  const randomForestF1DropVsJs = randomForestModes.jsFast.f1 - randomForestModes.zigTree.f1;

  if (decisionTreeFitSlowdown > maxZigTreeFitSlowdownVsJs) {
    throw new Error(
      `DecisionTree zig fit slowdown too large vs js-fast: ${decisionTreeFitSlowdown} > ${maxZigTreeFitSlowdownVsJs}.`,
    );
  }
  if (decisionTreePredictSlowdown > maxZigTreePredictSlowdownVsJs) {
    throw new Error(
      `DecisionTree zig predict slowdown too large vs js-fast: ${decisionTreePredictSlowdown} > ${maxZigTreePredictSlowdownVsJs}.`,
    );
  }
  if (randomForestFitSlowdown > maxZigForestFitSlowdownVsJs) {
    throw new Error(
      `RandomForest zig fit slowdown too large vs js-fast: ${randomForestFitSlowdown} > ${maxZigForestFitSlowdownVsJs}.`,
    );
  }
  if (randomForestPredictSlowdown > maxZigForestPredictSlowdownVsJs) {
    throw new Error(
      `RandomForest zig predict slowdown too large vs js-fast: ${randomForestPredictSlowdown} > ${maxZigForestPredictSlowdownVsJs}.`,
    );
  }
  if (decisionTreeAccuracyDropVsJs > maxZigTreeAccuracyDropVsJs) {
    throw new Error(
      `DecisionTree zig accuracy drop too large vs js-fast: ${decisionTreeAccuracyDropVsJs} > ${maxZigTreeAccuracyDropVsJs}.`,
    );
  }
  if (decisionTreeF1DropVsJs > maxZigTreeF1DropVsJs) {
    throw new Error(
      `DecisionTree zig F1 drop too large vs js-fast: ${decisionTreeF1DropVsJs} > ${maxZigTreeF1DropVsJs}.`,
    );
  }
  if (randomForestAccuracyDropVsJs > maxZigForestAccuracyDropVsJs) {
    throw new Error(
      `RandomForest zig accuracy drop too large vs js-fast: ${randomForestAccuracyDropVsJs} > ${maxZigForestAccuracyDropVsJs}.`,
    );
  }
  if (randomForestF1DropVsJs > maxZigForestF1DropVsJs) {
    throw new Error(
      `RandomForest zig F1 drop too large vs js-fast: ${randomForestF1DropVsJs} > ${maxZigForestF1DropVsJs}.`,
    );
  }

  if (baselineSnapshot?.suites?.treeBackendModes?.enabled) {
    const [baselineDecisionTreeModes, baselineRandomForestModes] =
      baselineSnapshot.suites.treeBackendModes.models;
    if (baselineDecisionTreeModes && baselineRandomForestModes) {
      const decisionTreeFitRetention =
        decisionTreeModes.comparison.zigFitSpeedupVsJs /
        baselineDecisionTreeModes.comparison.zigFitSpeedupVsJs;
      const randomForestFitRetention =
        randomForestModes.comparison.zigFitSpeedupVsJs /
        baselineRandomForestModes.comparison.zigFitSpeedupVsJs;

      if (decisionTreeFitRetention < minZigTreeFitRetentionVsBaseline) {
        throw new Error(
          `DecisionTree zig/js fit retention too low vs baseline: ${decisionTreeFitRetention} < ${minZigTreeFitRetentionVsBaseline}.`,
        );
      }
      if (randomForestFitRetention < minZigForestFitRetentionVsBaseline) {
        throw new Error(
          `RandomForest zig/js fit retention too low vs baseline: ${randomForestFitRetention} < ${minZigForestFitRetentionVsBaseline}.`,
        );
      }
    }
  }
}

console.log("Benchmark comparison health checks passed.");
