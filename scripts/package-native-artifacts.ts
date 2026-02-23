import { cp, mkdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";

interface PackageJson {
  version: string;
}

function argValue(flag: string): string | null {
  const index = Bun.argv.indexOf(flag);
  if (index === -1 || index + 1 >= Bun.argv.length) {
    return null;
  }
  return Bun.argv[index + 1];
}

function kernelExtension(osName: string): string {
  switch (osName) {
    case "windows":
      return "dll";
    case "linux":
      return "so";
    default:
      throw new Error(`Unsupported OS for packaging assets: ${osName}`);
  }
}

async function main(): Promise<void> {
  const osName = argValue("--os");
  const arch = argValue("--arch");
  if (!osName || !arch) {
    throw new Error("Usage: bun run scripts/package-native-artifacts.ts --os <linux|windows> --arch <x64|arm64>");
  }

  const packageJsonRaw = await readFile(resolve("package.json"), "utf-8");
  const packageJson = JSON.parse(packageJsonRaw) as PackageJson;
  const version = packageJson.version;
  const extension = kernelExtension(osName);

  const sourceKernel = resolve("dist", "native", `bun_scikit_kernels.${extension}`);
  const sourceAddon = resolve("dist", "native", "bun_scikit_node_addon.node");

  const outDir = resolve("dist", "release-assets");
  await mkdir(outDir, { recursive: true });

  const targetKernel = resolve(
    outDir,
    `bun_scikit_kernels-v${version}-${osName}-${arch}.${extension}`,
  );
  const targetAddon = resolve(
    outDir,
    `bun_scikit_node_addon-v${version}-${osName}-${arch}.node`,
  );

  await cp(sourceKernel, targetKernel, { force: true });
  await cp(sourceAddon, targetAddon, { force: true });
  console.log(`Packaged release assets:\n- ${targetKernel}\n- ${targetAddon}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
