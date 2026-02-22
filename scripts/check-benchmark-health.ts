import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

interface BenchmarkResult {
  implementation: string;
  fitMsMedian: number;
  predictMsMedian: number;
  mse: number;
  r2: number;
}

interface BenchmarkSnapshot {
  results: BenchmarkResult[];
  comparison: {
    mseDeltaVsSklearn: number;
    r2DeltaVsSklearn: number;
  };
}

const pathArgIndex = Bun.argv.indexOf("--input");
const inputPath =
  pathArgIndex !== -1 && pathArgIndex + 1 < Bun.argv.length
    ? resolve(Bun.argv[pathArgIndex + 1])
    : resolve("bench/results/heart-ci-current.json");

const snapshot = JSON.parse(await readFile(inputPath, "utf-8")) as BenchmarkSnapshot;

if (snapshot.results.length < 2) {
  throw new Error("Expected at least two benchmark result rows.");
}

for (const result of snapshot.results) {
  if (!(result.fitMsMedian > 0 && result.predictMsMedian > 0)) {
    throw new Error(
      `Benchmark timings must be positive for ${result.implementation}.`,
    );
  }

  if (!Number.isFinite(result.mse) || !Number.isFinite(result.r2)) {
    throw new Error(`Benchmark metrics must be finite for ${result.implementation}.`);
  }
}

if (Math.abs(snapshot.comparison.mseDeltaVsSklearn) > 0.01) {
  throw new Error(
    `MSE delta too large: ${snapshot.comparison.mseDeltaVsSklearn}.`,
  );
}

if (Math.abs(snapshot.comparison.r2DeltaVsSklearn) > 0.01) {
  throw new Error(`R2 delta too large: ${snapshot.comparison.r2DeltaVsSklearn}.`);
}

console.log("Benchmark comparison health checks passed.");
