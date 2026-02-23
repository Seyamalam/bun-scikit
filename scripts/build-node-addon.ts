import { cp, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { resolve } from "node:path";

function resolveNodeGypCommand(): string[] {
  const npmNodeGyp = process.env.npm_config_node_gyp?.trim();
  if (npmNodeGyp) {
    return ["node", npmNodeGyp, "rebuild"];
  }

  try {
    const require = createRequire(import.meta.url);
    const nodeGypScript = require.resolve("node-gyp/bin/node-gyp.js");
    return ["node", nodeGypScript, "rebuild"];
  } catch {
    return ["node-gyp", "rebuild"];
  }
}

async function main(): Promise<void> {
  const child = Bun.spawn(resolveNodeGypCommand(), {
    stdout: "inherit",
    stderr: "inherit",
  });

  const exitCode = await child.exited;
  if (exitCode !== 0) {
    throw new Error(`node-gyp rebuild failed with exit code ${exitCode}.`);
  }

  const source = resolve("build", "Release", "bun_scikit_node_addon.node");
  const outputDir = resolve("dist", "native");
  const destination = resolve(outputDir, "bun_scikit_node_addon.node");
  await mkdir(outputDir, { recursive: true });
  await cp(source, destination, { force: true });
  console.log(`Built Node-API addon: ${destination}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
