import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

interface BenchmarkResult {
  implementation: string;
  model: string;
  fitMsMedian: number;
  predictMsMedian: number;
  mse: number;
  r2: number;
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

function renderRows(results: BenchmarkResult[]): string[] {
  return results.map((result) => {
    return `| ${result.implementation} | ${result.model} | ${result.fitMsMedian.toFixed(4)} | ${result.predictMsMedian.toFixed(4)} | ${result.mse.toFixed(6)} | ${result.r2.toFixed(6)} |`;
  });
}

function renderBenchmarkSection(snapshot: BenchmarkSnapshot): string {
  return [
    START_MARKER,
    `Benchmark snapshot source: \`bench/results/heart-ci-latest.json\` (generated in CI workflow \`Benchmark Snapshot\`).`,
    `Dataset: \`${snapshot.dataset.path}\` (${snapshot.dataset.samples} samples, ${snapshot.dataset.features} features, test fraction ${snapshot.dataset.testFraction}).`,
    "",
    "| Implementation | Model | Fit median (ms) | Predict median (ms) | MSE | R2 |",
    "|---|---|---:|---:|---:|---:|",
    ...renderRows(snapshot.results),
    "",
    `Bun fit speedup vs scikit-learn: ${snapshot.comparison.fitSpeedupVsSklearn.toFixed(3)}x`,
    `Bun predict speedup vs scikit-learn: ${snapshot.comparison.predictSpeedupVsSklearn.toFixed(3)}x`,
    `MSE delta (bun - sklearn): ${snapshot.comparison.mseDeltaVsSklearn.toExponential(3)}`,
    `R2 delta (bun - sklearn): ${snapshot.comparison.r2DeltaVsSklearn.toExponential(3)}`,
    "",
    `Snapshot generated at: ${snapshot.generatedAt}`,
    END_MARKER,
  ].join("\n");
}

function normalizeLineEndings(content: string): string {
  return content.replace(/\r\n/g, "\n");
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
      `README benchmark section is out of date. Run: bun run bench:sync-readme`,
    );
    process.exit(1);
  }
  console.log("README benchmark section is up to date.");
  process.exit(0);
}

await writeFile(README_PATH, nextReadme, "utf-8");
console.log(`Updated README benchmark section from ${inputPath}`);
