import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

interface SharedBenchmarkResult {
  implementation: string;
  model: string;
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
  generatedAt: string;
  dataset: {
    path: string;
    samples: number;
    features: number;
    testFraction: number;
  };
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
  };
}

const START_MARKER = "<!-- BENCHMARK_TABLE_START -->";
const END_MARKER = "<!-- BENCHMARK_TABLE_END -->";
const README_PATH = resolve("README.md");
const DEFAULT_SNAPSHOT_PATH = resolve("bench/results/heart-ci-latest.json");

function parseArgValue(flag: string): string | null {
  const index = Bun.argv.indexOf(flag);
  if (index === -1 || index + 1 >= Bun.argv.length) {
    return null;
  }
  return Bun.argv[index + 1];
}

function normalizeLineEndings(content: string): string {
  return content.replace(/\r\n/g, "\n");
}

function renderBenchmarkSection(snapshot: BenchmarkSnapshot): string {
  const regression = snapshot.suites.regression;
  const classification = snapshot.suites.classification;
  const [bunReg, sklearnReg] = regression.results;
  const [bunCls, sklearnCls] = classification.results;

  return [
    START_MARKER,
    "Benchmark snapshot source: `bench/results/heart-ci-latest.json` (generated in CI workflow `Benchmark Snapshot`).",
    `Dataset: \`${snapshot.dataset.path}\` (${snapshot.dataset.samples} samples, ${snapshot.dataset.features} features, test fraction ${snapshot.dataset.testFraction}).`,
    "",
    "### Regression",
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
    "### Classification",
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
    END_MARKER,
  ].join("\n");
}

const inputPath = resolve(parseArgValue("--input") ?? DEFAULT_SNAPSHOT_PATH);
const checkMode = Bun.argv.includes("--check");

const [readme, snapshotRaw] = await Promise.all([
  readFile(README_PATH, "utf-8"),
  readFile(inputPath, "utf-8"),
]);
const snapshot = JSON.parse(snapshotRaw) as BenchmarkSnapshot;

const startIndex = readme.indexOf(START_MARKER);
const endIndex = readme.indexOf(END_MARKER);
if (startIndex === -1 || endIndex === -1 || endIndex < startIndex) {
  throw new Error(
    `README markers are missing or invalid. Expected markers: ${START_MARKER} and ${END_MARKER}.`,
  );
}

const existingSectionEnd = endIndex + END_MARKER.length;
const nextReadme =
  readme.slice(0, startIndex) +
  renderBenchmarkSection(snapshot) +
  readme.slice(existingSectionEnd);

if (checkMode) {
  if (normalizeLineEndings(nextReadme) !== normalizeLineEndings(readme)) {
    console.error(
      "README benchmark section is out of date. Run: bun run bench:sync-readme",
    );
    process.exit(1);
  }
  console.log("README benchmark section is up to date.");
  process.exit(0);
}

await writeFile(README_PATH, nextReadme, "utf-8");
console.log(`Updated README benchmark section from ${inputPath}`);
