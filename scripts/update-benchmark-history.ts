import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

interface BenchmarkSnapshot {
  generatedAt: string;
  benchmarkConfig: {
    iterations: number;
    warmup: number;
  };
  suites: {
    regression: {
      comparison: {
        fitSpeedupVsSklearn: number;
        predictSpeedupVsSklearn: number;
        mseDeltaVsSklearn: number;
        r2DeltaVsSklearn: number;
      };
    };
    classification: {
      comparison: {
        fitSpeedupVsSklearn: number;
        predictSpeedupVsSklearn: number;
        accuracyDeltaVsSklearn: number;
        f1DeltaVsSklearn: number;
      };
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

const inputPath = resolve(
  parseArgValue("--input") ?? "bench/results/heart-ci-latest.json",
);
const outputPath = resolve(
  parseArgValue("--output") ?? "bench/results/history/heart-ci-history.jsonl",
);

const snapshot = JSON.parse(await readFile(inputPath, "utf-8")) as BenchmarkSnapshot;

const historyRecord = {
  generatedAt: snapshot.generatedAt,
  benchmarkConfig: snapshot.benchmarkConfig,
  regression: snapshot.suites.regression.comparison,
  classification: snapshot.suites.classification.comparison,
};

let existing = "";
try {
  existing = await readFile(outputPath, "utf-8");
} catch {
  existing = "";
}

const lines = existing
  .split(/\r?\n/)
  .map((line) => line.trim())
  .filter((line) => line.length > 0);

if (lines.length > 0) {
  const last = JSON.parse(lines[lines.length - 1]) as { generatedAt?: string };
  if (last.generatedAt === historyRecord.generatedAt) {
    console.log(`History already contains snapshot ${historyRecord.generatedAt}.`);
    process.exit(0);
  }
}

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${existing}${JSON.stringify(historyRecord)}\n`, "utf-8");
console.log(`Appended benchmark history: ${outputPath}`);
