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

interface BenchmarkSnapshot {
  suites: {
    regression: {
      results: [RegressionBenchmarkResult, RegressionBenchmarkResult];
      comparison: {
        mseDeltaVsSklearn: number;
        r2DeltaVsSklearn: number;
      };
    };
    classification: {
      results: [ClassificationBenchmarkResult, ClassificationBenchmarkResult];
      comparison: {
        accuracyDeltaVsSklearn: number;
        f1DeltaVsSklearn: number;
      };
    };
  };
}

const pathArgIndex = Bun.argv.indexOf("--input");
const inputPath =
  pathArgIndex !== -1 && pathArgIndex + 1 < Bun.argv.length
    ? resolve(Bun.argv[pathArgIndex + 1])
    : resolve("bench/results/heart-ci-current.json");

const snapshot = JSON.parse(await readFile(inputPath, "utf-8")) as BenchmarkSnapshot;

const [bunRegression, sklearnRegression] = snapshot.suites.regression.results;
const [bunClassification, sklearnClassification] = snapshot.suites.classification.results;

for (const result of [bunRegression, sklearnRegression, bunClassification, sklearnClassification]) {
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

console.log("Benchmark comparison health checks passed.");
