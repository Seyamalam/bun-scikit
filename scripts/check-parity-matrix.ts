import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as api from "../src";

interface ClassContract {
  name: string;
  requiredMethods: string[];
}

interface InterfaceContract {
  sourceFile: string;
  name: string;
  requiredFields: string[];
}

interface ParityMatrix {
  metadata: {
    targetSklearnVersion: string;
    updatedAt: string;
    policy: string;
  };
  requiredExports: string[];
  contracts: {
    classes: ClassContract[];
    interfaces: InterfaceContract[];
  };
}

function extractInterfaceFields(source: string, interfaceName: string): string[] {
  const pattern = new RegExp(`export\\s+interface\\s+${interfaceName}\\s*\\{([\\s\\S]*?)\\n\\}`, "m");
  const match = pattern.exec(source);
  if (!match) {
    throw new Error(`Interface '${interfaceName}' was not found.`);
  }
  const body = match[1];
  const names: string[] = [];
  const fieldRegex = /^\s*([A-Za-z0-9_]+)\??\s*:/gm;
  let fieldMatch = fieldRegex.exec(body);
  while (fieldMatch) {
    names.push(fieldMatch[1]);
    fieldMatch = fieldRegex.exec(body);
  }
  return names;
}

function assertNoDuplicates(values: string[], label: string): void {
  const seen = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      throw new Error(`Duplicate ${label} entry '${value}'.`);
    }
    seen.add(value);
  }
}

const matrixPath = resolve("docs/parity-matrix.json");
const matrix = JSON.parse(readFileSync(matrixPath, "utf-8")) as ParityMatrix;

assertNoDuplicates(matrix.requiredExports, "required export");

const runtimeExports = Object.keys(api).sort();
const matrixExports = matrix.requiredExports.slice().sort();

const missingInRuntime = matrixExports.filter((name) => !(name in api));
const missingInMatrix = runtimeExports.filter((name) => !matrix.requiredExports.includes(name));

let failed = false;

if (missingInRuntime.length > 0) {
  failed = true;
  console.error("Missing required exports in runtime API:");
  for (const name of missingInRuntime) {
    console.error(`- ${name}`);
  }
}

if (missingInMatrix.length > 0) {
  failed = true;
  console.error("Runtime exports missing from parity matrix:");
  for (const name of missingInMatrix) {
    console.error(`- ${name}`);
  }
}

for (const contract of matrix.contracts.classes) {
  const exported = (api as Record<string, unknown>)[contract.name];
  if (typeof exported !== "function") {
    failed = true;
    console.error(`Class contract '${contract.name}' is not exported as a constructor.`);
    continue;
  }

  const prototype = (exported as { prototype?: Record<string, unknown> }).prototype ?? {};
  for (const method of contract.requiredMethods) {
    if (typeof prototype[method] !== "function") {
      failed = true;
      console.error(`Class '${contract.name}' is missing required method '${method}'.`);
    }
  }
}

for (const contract of matrix.contracts.interfaces) {
  const sourcePath = resolve(contract.sourceFile);
  const source = readFileSync(sourcePath, "utf-8");
  const fields = extractInterfaceFields(source, contract.name);
  for (const field of contract.requiredFields) {
    if (!fields.includes(field)) {
      failed = true;
      console.error(
        `Interface '${contract.name}' (${contract.sourceFile}) is missing required field '${field}'.`,
      );
    }
  }
}

if (failed) {
  console.error("Parity matrix check failed.");
  process.exit(1);
}

console.log(
  `Parity matrix check passed for sklearn ${matrix.metadata.targetSklearnVersion} (${runtimeExports.length} exports).`,
);
