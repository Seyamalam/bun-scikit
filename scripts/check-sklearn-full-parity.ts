import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import * as api from "../src";

interface InventorySymbol {
  name: string;
  qualifiedName: string;
  module: string;
  kind: string;
}

interface SklearnPublicApiInventory {
  metadata: {
    sklearnVersion: string;
    symbolCountQualified: number;
    symbolCountUniqueShortName: number;
  };
  uniqueShortNames: string[];
  symbols: InventorySymbol[];
}

interface ModuleCoverage {
  module: string;
  totalUniqueShortNames: number;
  coveredUniqueShortNames: number;
  coveragePercent: number;
}

interface ModuleGatesConfig {
  moduleMinCoveragePercent: Record<string, number>;
}

interface ModuleGateEvaluation {
  module: string;
  minCoveragePercent: number;
  observedCoveragePercent: number;
  passed: boolean;
}

function snakeToCamel(value: string): string {
  if (!value.includes("_")) {
    return value;
  }
  return value.replace(/_([a-zA-Z0-9])/g, (_, group: string) => group.toUpperCase());
}

function candidateRuntimeNames(sklearnSymbolName: string): string[] {
  const candidates = new Set<string>([sklearnSymbolName]);
  if (sklearnSymbolName.includes("_")) {
    candidates.add(snakeToCamel(sklearnSymbolName));
  }
  return Array.from(candidates);
}

function envFlag(name: string, fallback = false): boolean {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const normalized = raw.trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer. Got '${raw}'.`);
  }
  return parsed;
}

function coveragePercent(covered: number, total: number): number {
  if (total <= 0) {
    return 100;
  }
  return (covered / total) * 100;
}

const inventoryPath = resolve(process.env.PARITY_FULL_INVENTORY_PATH ?? "docs/sklearn-public-api.json");
const strict = envFlag("PARITY_FULL_STRICT", false);
const missingPreviewLimit = envInt("PARITY_FULL_MISSING_PREVIEW", 80);
const moduleCoverageTopN = envInt("PARITY_FULL_MODULE_TOP_N", 30);
const moduleGatesPathRaw = process.env.PARITY_FULL_MODULE_GATES_PATH ?? "docs/parity-module-gates.json";
const moduleGatesPath = resolve(moduleGatesPathRaw);

const inventory = JSON.parse(readFileSync(inventoryPath, "utf-8")) as SklearnPublicApiInventory;
const runtimeExports = Object.keys(api).sort();
const runtimeSet = new Set(runtimeExports);

const inventoryUnique = Array.from(new Set(inventory.uniqueShortNames)).sort();
const coveredSymbols = inventoryUnique.filter((name) =>
  candidateRuntimeNames(name).some((candidate) => runtimeSet.has(candidate)),
);
const missingSymbols = inventoryUnique.filter(
  (name) => !candidateRuntimeNames(name).some((candidate) => runtimeSet.has(candidate)),
);
const coveredByExact = inventoryUnique.filter((name) => runtimeSet.has(name));
const coveredByCaseMapping = coveredSymbols.filter((name) => !runtimeSet.has(name));

const moduleSymbolSet = new Map<string, Set<string>>();
for (let i = 0; i < inventory.symbols.length; i += 1) {
  const symbol = inventory.symbols[i];
  if (!moduleSymbolSet.has(symbol.module)) {
    moduleSymbolSet.set(symbol.module, new Set<string>());
  }
  moduleSymbolSet.get(symbol.module)!.add(symbol.name);
}

const moduleCoverage: ModuleCoverage[] = Array.from(moduleSymbolSet.entries())
  .map(([module, names]) => {
    const uniqueNames = Array.from(names);
    const covered = uniqueNames.filter((name) =>
      candidateRuntimeNames(name).some((candidate) => runtimeSet.has(candidate)),
    ).length;
    return {
      module,
      totalUniqueShortNames: uniqueNames.length,
      coveredUniqueShortNames: covered,
      coveragePercent: coveragePercent(covered, uniqueNames.length),
    };
  })
  .sort((a, b) => {
    if (b.totalUniqueShortNames !== a.totalUniqueShortNames) {
      return b.totalUniqueShortNames - a.totalUniqueShortNames;
    }
    return a.module.localeCompare(b.module);
  });

const moduleCoverageByName = new Map(moduleCoverage.map((entry) => [entry.module, entry]));
let moduleGateEvaluations: ModuleGateEvaluation[] = [];
if (existsSync(moduleGatesPath)) {
  const moduleGates = JSON.parse(readFileSync(moduleGatesPath, "utf-8")) as ModuleGatesConfig;
  moduleGateEvaluations = Object.entries(moduleGates.moduleMinCoveragePercent ?? {})
    .map(([module, minCoveragePercent]) => {
      if (!Number.isFinite(minCoveragePercent) || minCoveragePercent < 0 || minCoveragePercent > 100) {
        throw new Error(`Invalid module coverage threshold for ${module}: ${minCoveragePercent}`);
      }
      const observed = moduleCoverageByName.get(module)?.coveragePercent ?? 0;
      return {
        module,
        minCoveragePercent,
        observedCoveragePercent: observed,
        passed: observed + 1e-12 >= minCoveragePercent,
      };
    })
    .sort((a, b) => a.module.localeCompare(b.module));
}

const covered = coveredSymbols.length;
const total = inventoryUnique.length;
const coverage = coveragePercent(covered, total);
const failed = (strict && missingSymbols.length > 0) || moduleGateEvaluations.some((gate) => !gate.passed);

const report = {
  generatedAt: new Date().toISOString(),
  inventoryPath,
  sklearnVersion: inventory.metadata.sklearnVersion,
  totalUniqueSymbols: total,
  runtimeExportCount: runtimeExports.length,
  coveredUniqueSymbols: covered,
  missingUniqueSymbols: missingSymbols.length,
  coveragePercent: Number(coverage.toFixed(3)),
  coveredByExactMatch: coveredByExact.length,
  coveredBySnakeCaseToCamelCase: coveredByCaseMapping.length,
  strictMode: strict,
  failed,
  missingSymbolsPreview: missingSymbols.slice(0, missingPreviewLimit),
  moduleCoverageTop: moduleCoverage.slice(0, moduleCoverageTopN),
  moduleGatesPath: existsSync(moduleGatesPath) ? moduleGatesPath : null,
  moduleGateEvaluations,
};

const reportPath = process.env.PARITY_FULL_REPORT_PATH;
if (reportPath) {
  const absoluteReportPath = resolve(reportPath);
  mkdirSync(dirname(absoluteReportPath), { recursive: true });
  writeFileSync(absoluteReportPath, JSON.stringify(report, null, 2), "utf-8");
}

console.log(
  `Full sklearn parity coverage: ${covered}/${total} (${coverage.toFixed(2)}%) against sklearn ${inventory.metadata.sklearnVersion}.`,
);
if (missingSymbols.length > 0) {
  console.log(`Missing symbols (preview up to ${missingPreviewLimit}):`);
  for (let i = 0; i < Math.min(missingSymbols.length, missingPreviewLimit); i += 1) {
    console.log(`- ${missingSymbols[i]}`);
  }
  if (missingSymbols.length > missingPreviewLimit) {
    console.log(`...and ${missingSymbols.length - missingPreviewLimit} more.`);
  }
}

if (moduleGateEvaluations.length > 0) {
  console.log("Module coverage gates:");
  for (const gate of moduleGateEvaluations) {
    const status = gate.passed ? "PASS" : "FAIL";
    console.log(
      `- ${status} ${gate.module}: ${gate.observedCoveragePercent.toFixed(2)}% (min ${gate.minCoveragePercent.toFixed(2)}%)`,
    );
  }
}

if (failed) {
  console.error("Full sklearn parity check failed (strict missing-symbol gate or module coverage gate).");
  process.exit(1);
}

if (missingSymbols.length > 0) {
  console.log("Full sklearn parity check completed in non-strict mode (missing symbols allowed).");
} else {
  console.log("Full sklearn parity check passed with complete symbol coverage.");
}
