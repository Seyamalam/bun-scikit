import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import { resolve } from "node:path";

interface PackageJson {
  version: string;
  repository?: {
    url?: string;
  };
}

function mapOs(value: NodeJS.Platform): "windows" | "linux" | null {
  if (value === "win32") {
    return "windows";
  }
  if (value === "linux") {
    return "linux";
  }
  return null;
}

function mapArch(value: string): "x64" | "arm64" | null {
  if (value === "x64") {
    return "x64";
  }
  if (value === "arm64") {
    return "arm64";
  }
  return null;
}

function kernelExtension(osName: "windows" | "linux"): string {
  return osName === "windows" ? "dll" : "so";
}

function parseRepositorySlug(url: string | undefined): string | null {
  if (!url) {
    return null;
  }
  const normalized = url.replace("git+", "").replace(/\.git$/, "");
  const match = normalized.match(/github\.com\/([^/]+\/[^/]+)/i);
  return match?.[1] ?? null;
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function downloadToFile(url: string, path: string): Promise<boolean> {
  const response = await fetch(url);
  if (!response.ok) {
    return false;
  }
  const bytes = await response.arrayBuffer();
  await writeFile(path, Buffer.from(bytes));
  return true;
}

async function tryDownloadPrebuilt(): Promise<boolean> {
  const packageJsonRaw = await readFile(resolve("package.json"), "utf-8");
  const packageJson = JSON.parse(packageJsonRaw) as PackageJson;

  const osName = mapOs(process.platform);
  const arch = mapArch(process.arch);
  if (!osName || !arch) {
    console.log(
      `[bun-scikit] prebuilt binaries are unavailable for platform=${process.platform} arch=${process.arch}`,
    );
    return false;
  }

  const repoSlug = parseRepositorySlug(packageJson.repository?.url) ?? "Seyamalam/bun-scikit";
  const version = packageJson.version;
  const ext = kernelExtension(osName);
  const tag = `v${version}`;
  const base =
    `https://github.com/${repoSlug}/releases/download/${tag}`;

  const kernelName = `bun_scikit_kernels-v${version}-${osName}-${arch}.${ext}`;
  const addonName = `bun_scikit_node_addon-v${version}-${osName}-${arch}.node`;
  const kernelUrl = `${base}/${kernelName}`;
  const addonUrl = `${base}/${addonName}`;

  const outDir = resolve("dist", "native");
  await mkdir(outDir, { recursive: true });
  const kernelPath = resolve(outDir, `bun_scikit_kernels.${ext}`);
  const addonPath = resolve(outDir, "bun_scikit_node_addon.node");

  const [kernelOk, addonOk] = await Promise.all([
    downloadToFile(kernelUrl, kernelPath),
    downloadToFile(addonUrl, addonPath),
  ]);

  if (kernelOk && addonOk) {
    console.log(`[bun-scikit] downloaded native prebuilt artifacts for ${osName}-${arch}`);
    return true;
  }

  return false;
}

async function tryLocalBuild(): Promise<void> {
  console.log("[bun-scikit] prebuilt binaries not found; attempting local native build");
  {
    const child = Bun.spawn(["bun", "run", "native:build"], {
      stdout: "inherit",
      stderr: "inherit",
    });
    const code = await child.exited;
    if (code !== 0) {
      throw new Error(`native:build failed with exit code ${code}`);
    }
  }

  {
    const child = Bun.spawn(["bun", "run", "native:build:node-addon"], {
      stdout: "inherit",
      stderr: "inherit",
    });
    const code = await child.exited;
    if (code !== 0) {
      console.warn(
        "[bun-scikit] Node-API addon local build failed; the Zig FFI backend can still be used.",
      );
    }
  }
}

async function main(): Promise<void> {
  if (process.env.BUN_SCIKIT_SKIP_NATIVE_INSTALL === "1" || process.env.CI === "true") {
    console.log("[bun-scikit] skipping native install bootstrap in CI/skip mode");
    return;
  }

  const existingKernelDll = resolve("dist", "native", "bun_scikit_kernels.dll");
  const existingKernelSo = resolve("dist", "native", "bun_scikit_kernels.so");
  if (await fileExists(existingKernelDll) || (await fileExists(existingKernelSo))) {
    return;
  }

  const downloaded = await tryDownloadPrebuilt();
  if (downloaded) {
    return;
  }

  try {
    await tryLocalBuild();
  } catch (error) {
    console.warn("[bun-scikit] native postinstall setup did not complete.", error);
  }
}

main().catch((error) => {
  console.warn("[bun-scikit] native install script error:", error);
});
