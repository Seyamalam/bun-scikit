import { cp, mkdir } from "node:fs/promises";
import { resolve } from "node:path";

async function main(): Promise<void> {
  const child = Bun.spawn(["bunx", "node-gyp", "rebuild"], {
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
