import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

function sharedLibraryExtension(): string {
  switch (process.platform) {
    case "win32":
      return "dll";
    case "darwin":
      return "dylib";
    default:
      return "so";
  }
}

async function main(): Promise<void> {
  const extension = sharedLibraryExtension();
  const outputDir = resolve("dist/native");
  const outputFile = resolve(outputDir, `bun_scikit_kernels.${extension}`);

  await mkdir(outputDir, { recursive: true });

  const child = Bun.spawn(
    [
      "zig",
      "build-lib",
      "zig/kernels.zig",
      "-dynamic",
      "-O",
      "ReleaseFast",
      "-fstrip",
      `-femit-bin=${outputFile}`,
    ],
    {
      stdout: "inherit",
      stderr: "inherit",
    },
  );

  const exitCode = await child.exited;
  if (exitCode !== 0) {
    throw new Error(`zig build-lib failed with exit code ${exitCode}.`);
  }

  console.log(`Built Zig kernels: ${outputFile}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
