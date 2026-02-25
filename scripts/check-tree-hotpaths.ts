import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

interface BenchResult {
  label: string;
  backend: "js-fast" | "zig-tree";
  fitMedianMs: number;
  predictMedianMs: number;
}

interface HotpathSnapshot {
  generatedAt: string;
  results: BenchResult[];
  comparison: {
    decisionTree: {
      zigFitSpeedupVsJs: number;
      zigPredictSpeedupVsJs: number;
    };
    randomForest: {
      zigFitSpeedupVsJs: number;
      zigPredictSpeedupVsJs: number;
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

function threshold(envName: string, fallback: number): number {
  const raw = process.env[envName];
  if (!raw) {
    return fallback;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${envName} must be a positive number. Got "${raw}".`);
  }
  return value;
}

const inputPath = resolve(
  parseArgValue("--input") ??
    process.env.BENCH_TREE_HOTPATHS_INPUT ??
    "bench/results/tree-hotpaths-current.json",
);
const baselinePath = resolve(
  parseArgValue("--baseline") ??
    process.env.BENCH_TREE_HOTPATHS_BASELINE ??
    "bench/results/tree-hotpaths-baseline.json",
);

const snapshot = JSON.parse(
  await readFile(inputPath, "utf8"),
) as HotpathSnapshot;
const baseline = JSON.parse(
  await readFile(baselinePath, "utf8"),
) as HotpathSnapshot;

const minDtFitRetention = threshold(
  "BENCH_MIN_DT_HOTPATH_FIT_RETENTION",
  0.6,
);
const minDtPredictRetention = threshold(
  "BENCH_MIN_DT_HOTPATH_PREDICT_RETENTION",
  0.6,
);
const minRfFitRetention = threshold(
  "BENCH_MIN_RF_HOTPATH_FIT_RETENTION",
  0.6,
);
const minRfPredictRetention = threshold(
  "BENCH_MIN_RF_HOTPATH_PREDICT_RETENTION",
  0.6,
);

for (const row of snapshot.results) {
  if (!(row.fitMedianMs > 0) || !(row.predictMedianMs > 0)) {
    throw new Error(
      `Hotpath timings must be positive for ${row.label} (${row.backend}).`,
    );
  }
}

const dtRetentionFit =
  snapshot.comparison.decisionTree.zigFitSpeedupVsJs /
  baseline.comparison.decisionTree.zigFitSpeedupVsJs;
const dtRetentionPredict =
  snapshot.comparison.decisionTree.zigPredictSpeedupVsJs /
  baseline.comparison.decisionTree.zigPredictSpeedupVsJs;
const rfRetentionFit =
  snapshot.comparison.randomForest.zigFitSpeedupVsJs /
  baseline.comparison.randomForest.zigFitSpeedupVsJs;
const rfRetentionPredict =
  snapshot.comparison.randomForest.zigPredictSpeedupVsJs /
  baseline.comparison.randomForest.zigPredictSpeedupVsJs;

if (dtRetentionFit < minDtFitRetention) {
  throw new Error(
    `DecisionTree hotpath fit retention too low: ${dtRetentionFit} < ${minDtFitRetention}.`,
  );
}
if (dtRetentionPredict < minDtPredictRetention) {
  throw new Error(
    `DecisionTree hotpath predict retention too low: ${dtRetentionPredict} < ${minDtPredictRetention}.`,
  );
}
if (rfRetentionFit < minRfFitRetention) {
  throw new Error(
    `RandomForest hotpath fit retention too low: ${rfRetentionFit} < ${minRfFitRetention}.`,
  );
}
if (rfRetentionPredict < minRfPredictRetention) {
  throw new Error(
    `RandomForest hotpath predict retention too low: ${rfRetentionPredict} < ${minRfPredictRetention}.`,
  );
}

console.log("Tree hotpath regression checks passed.");
