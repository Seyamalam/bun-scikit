import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { resolve } from "node:path";

interface MatrixReport {
  generatedAt: string;
  targetSklearnVersion: string;
  runtimeExportCount: number;
  requiredExportCount: number;
  missingInRuntime: string[];
  missingInMatrix: string[];
  classContractFailures: string[];
  interfaceContractFailures: string[];
  failed: boolean;
}

interface SklearnReport {
  generatedAt: string;
  failed: boolean;
  failures: string[];
  metrics: Record<string, number>;
  limits: Record<string, number>;
}

interface ParityHistoryEntry {
  timestamp: string;
  version: string;
  sha: string;
  matrixFailed: boolean;
  sklearnFailed: boolean;
  metricFailures: string[];
  keyMetrics: Record<string, number>;
}

function loadJson<T>(path: string): T {
  if (!existsSync(path)) {
    throw new Error(`Required report file is missing: ${path}`);
  }
  return JSON.parse(readFileSync(path, "utf-8")) as T;
}

function nowIso(): string {
  return new Date().toISOString();
}

function shortSha(): string {
  const envSha = process.env.GITHUB_SHA;
  if (envSha && envSha.length >= 7) {
    return envSha.slice(0, 7);
  }
  return execSync("git rev-parse --short HEAD", { encoding: "utf-8" }).trim();
}

const outputDir = resolve(process.env.PARITY_REPORT_DIR ?? "bench/results/parity");
const matrixPath = resolve(
  process.env.PARITY_MATRIX_REPORT_PATH ?? `${outputDir}/parity-matrix-report.json`,
);
const sklearnPath = resolve(
  process.env.PARITY_SKLEARN_REPORT_PATH ?? `${outputDir}/parity-sklearn-report.json`,
);
const historyPath = resolve(process.env.PARITY_HISTORY_PATH ?? "bench/results/history/parity-history.jsonl");

mkdirSync(outputDir, { recursive: true });
mkdirSync(resolve("bench/results/history"), { recursive: true });

const matrix = loadJson<MatrixReport>(matrixPath);
const sklearn = loadJson<SklearnReport>(sklearnPath);
const pkg = JSON.parse(readFileSync(resolve("package.json"), "utf-8")) as { version: string };
const version = pkg.version;
const sha = shortSha();
const timestamp = nowIso();

const keyMetrics: Record<string, number> = {};
for (const name of [
  "decision_tree_mismatch",
  "random_forest_mismatch",
  "multi_seed_decision_tree_mismatch_avg",
  "multi_seed_random_forest_mismatch_avg",
  "pipeline_logreg_probe_proba_mad",
  "composition_pipeline_transform_mse",
  "composition_column_transformer_mse",
]) {
  if (typeof sklearn.metrics[name] === "number") {
    keyMetrics[name] = sklearn.metrics[name];
  }
}

const historyEntry: ParityHistoryEntry = {
  timestamp,
  version,
  sha,
  matrixFailed: matrix.failed,
  sklearnFailed: sklearn.failed,
  metricFailures: sklearn.failures.slice(),
  keyMetrics,
};

const previousHistory: ParityHistoryEntry[] = [];
if (existsSync(historyPath)) {
  const lines = readFileSync(historyPath, "utf-8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  for (const line of lines) {
    try {
      previousHistory.push(JSON.parse(line) as ParityHistoryEntry);
    } catch {
      // Ignore malformed historical lines.
    }
  }
}
writeFileSync(historyPath, `${JSON.stringify(historyEntry)}\n`, { flag: "a" });

const driftHistory = previousHistory.slice(-19).concat(historyEntry);
const parityReport = {
  generatedAt: timestamp,
  version,
  sha,
  matrix,
  sklearn,
  driftHistory,
};

const versionedJsonPath = resolve(outputDir, `parity-report-${version}-${sha}.json`);
const latestJsonPath = resolve(outputDir, "parity-report-latest.json");
writeFileSync(versionedJsonPath, JSON.stringify(parityReport, null, 2), "utf-8");
writeFileSync(latestJsonPath, JSON.stringify(parityReport, null, 2), "utf-8");

const metricRows = Object.keys(sklearn.metrics)
  .sort((a, b) => a.localeCompare(b))
  .map((name) => {
    const value = sklearn.metrics[name];
    const limit = sklearn.limits[name];
    const status = value <= limit ? "PASS" : "FAIL";
    return `| ${name} | ${value} | ${limit} | ${status} |`;
  });

const historyRows = driftHistory.map(
  (entry) =>
    `| ${entry.timestamp} | ${entry.version} | ${entry.sha} | ${entry.matrixFailed ? "fail" : "pass"} | ${entry.sklearnFailed ? "fail" : "pass"} | ${entry.metricFailures.length} |`,
);

const markdown = [
  "# Parity Report",
  "",
  `- Generated: ${timestamp}`,
  `- Version: ${version}`,
  `- Commit: ${sha}`,
  `- Matrix gate: ${matrix.failed ? "FAIL" : "PASS"}`,
  `- sklearn gate: ${sklearn.failed ? "FAIL" : "PASS"}`,
  "",
  "## Matrix Coverage",
  "",
  `- Target sklearn: ${matrix.targetSklearnVersion}`,
  `- Runtime exports: ${matrix.runtimeExportCount}`,
  `- Required exports: ${matrix.requiredExportCount}`,
  `- Missing in runtime: ${matrix.missingInRuntime.length}`,
  `- Missing in matrix: ${matrix.missingInMatrix.length}`,
  `- Class contract failures: ${matrix.classContractFailures.length}`,
  `- Interface contract failures: ${matrix.interfaceContractFailures.length}`,
  "",
  "## sklearn Metrics",
  "",
  "| Metric | Value | Limit | Status |",
  "| --- | ---: | ---: | --- |",
  ...metricRows,
  "",
  "## Drift History (latest 20)",
  "",
  "| Timestamp | Version | SHA | Matrix | sklearn | Metric Failures |",
  "| --- | --- | --- | --- | --- | ---: |",
  ...historyRows,
  "",
].join("\n");

const versionedMdPath = resolve(outputDir, `parity-report-${version}-${sha}.md`);
const latestMdPath = resolve(outputDir, "parity-report-latest.md");
writeFileSync(versionedMdPath, markdown, "utf-8");
writeFileSync(latestMdPath, markdown, "utf-8");

console.log(`Wrote parity reports:\n- ${versionedJsonPath}\n- ${versionedMdPath}`);
