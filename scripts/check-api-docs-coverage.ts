import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const INDEX_PATH = resolve("src/index.ts");
const DOCS_PATH = resolve("docs/api.md");

function normalizeModulePath(pathLiteral: string): string {
  return pathLiteral.replace(/^\.?\//, "");
}

function extractExportedSymbolNames(source: string): string[] {
  const exportMatches = source.matchAll(
    /export\s+(?:class|function|const|type|interface|enum)\s+([A-Za-z0-9_]+)/g,
  );
  return Array.from(exportMatches, (match) => match[1]);
}

const indexSource = await readFile(INDEX_PATH, "utf-8");
const docsSource = await readFile(DOCS_PATH, "utf-8");

const modulePaths = Array.from(indexSource.matchAll(/export\s+\*\s+from\s+"\.\/(.+)";/g)).map(
  (match) => normalizeModulePath(match[1]),
);

const symbolNames = new Set<string>();
for (const modulePath of modulePaths) {
  const moduleSource = await readFile(resolve("src", `${modulePath}.ts`), "utf-8");
  for (const name of extractExportedSymbolNames(moduleSource)) {
    symbolNames.add(name);
  }
}

function escapeRegExp(literal: string): string {
  return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const missing = Array.from(symbolNames)
  .filter((name) => {
    const pattern = new RegExp("`[^`]*\\b" + escapeRegExp(name) + "\\b[^`]*`");
    return !pattern.test(docsSource);
  })
  .sort((a, b) => a.localeCompare(b));

if (missing.length > 0) {
  console.error("docs/api.md is missing exported API symbols:");
  for (const name of missing) {
    console.error(`- ${name}`);
  }
  process.exit(1);
}

console.log(`API docs coverage check passed for ${symbolNames.size} exported symbols.`);
