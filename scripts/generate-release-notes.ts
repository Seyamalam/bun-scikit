import { execSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

interface ConventionalCommit {
  raw: string;
  type: string;
  scope: string | null;
  description: string;
}

function run(command: string): string {
  return execSync(command, { encoding: "utf-8" }).trim();
}

function parseVersion(version: string): { major: number; minor: number; patch: number } {
  const parts = version.split(".").map((value) => Number(value));
  if (parts.length !== 3 || parts.some((value) => !Number.isInteger(value) || value < 0)) {
    throw new Error(`Invalid semantic version '${version}'.`);
  }
  return { major: parts[0], minor: parts[1], patch: parts[2] };
}

function nextPatchVersion(version: string): string {
  const parsed = parseVersion(version);
  return `${parsed.major}.${parsed.minor}.${parsed.patch + 1}`;
}

function parseCommit(message: string): ConventionalCommit {
  const match = /^([a-z]+)(\(([^)]+)\))?:\s+(.+)$/i.exec(message);
  if (!match) {
    return {
      raw: message,
      type: "other",
      scope: null,
      description: message,
    };
  }
  return {
    raw: message,
    type: match[1].toLowerCase(),
    scope: match[3] ?? null,
    description: match[4],
  };
}

function commitBullets(commits: ConventionalCommit[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (let i = 0; i < commits.length; i += 1) {
    const item = `- ${commits[i].raw}`;
    if (!seen.has(item)) {
      seen.add(item);
      out.push(item);
    }
  }
  return out;
}

function updateChangelog(
  changelogPath: string,
  parityBullets: string[],
  latestTag: string | null,
): boolean {
  const source = readFileSync(changelogPath, "utf-8");
  const unreleasedHeader = "## [Unreleased]";
  const start = source.indexOf(unreleasedHeader);
  if (start === -1) {
    throw new Error("CHANGELOG.md is missing the [Unreleased] section.");
  }
  const afterStart = start + unreleasedHeader.length;
  const nextSectionOffset = source.slice(afterStart).search(/\n## \[/);
  const end = nextSectionOffset === -1 ? source.length : afterStart + nextSectionOffset;
  const unreleasedSection = source.slice(start, end);

  const addedHeader = "\n### Added\n";
  let workingSection = unreleasedSection;
  if (!workingSection.includes(addedHeader)) {
    workingSection += `${addedHeader}`;
  }

  const autoStart = "<!-- PARITY_AUTO_START -->";
  const autoEnd = "<!-- PARITY_AUTO_END -->";
  const generatedBullets = parityBullets.length > 0
    ? parityBullets
    : [`- (auto) No new parity commits detected since ${latestTag ?? "project start"}.`];
  const generatedBlock = `${autoStart}\n${generatedBullets.join("\n")}\n${autoEnd}`;

  const existingStart = workingSection.indexOf(autoStart);
  const existingEnd = workingSection.indexOf(autoEnd);
  if (existingStart !== -1 && existingEnd !== -1 && existingEnd > existingStart) {
    const before = workingSection.slice(0, existingStart);
    const after = workingSection.slice(existingEnd + autoEnd.length);
    workingSection = `${before}${generatedBlock}${after}`;
  } else {
    const addedIndex = workingSection.indexOf(addedHeader);
    const insertionPoint = addedIndex + addedHeader.length;
    workingSection =
      workingSection.slice(0, insertionPoint) +
      `${generatedBlock}\n` +
      workingSection.slice(insertionPoint);
  }

  const updated = source.slice(0, start) + workingSection + source.slice(end);
  if (updated !== source) {
    writeFileSync(changelogPath, updated, "utf-8");
    return true;
  }
  return false;
}

const packageJson = JSON.parse(readFileSync(resolve("package.json"), "utf-8")) as {
  version: string;
};
const forcedVersion = process.env.RELEASE_NOTES_TARGET_VERSION;
const targetVersion = forcedVersion ?? nextPatchVersion(packageJson.version);

const latestTag = run('git tag --list "v*" --sort=-v:refname').split(/\r?\n/).find(Boolean) ?? null;
const range = latestTag ? `${latestTag}..HEAD` : "HEAD";
const commitSubjects = run(`git log --pretty=format:%s ${range}`)
  .split(/\r?\n/)
  .map((value) => value.trim())
  .filter((value) => value.length > 0);

const commits = commitSubjects.map(parseCommit);
const parityCommits = commits.filter(
  (commit) =>
    commit.scope?.toLowerCase().includes("parity") ||
    commit.raw.toLowerCase().includes("parity"),
);
const addedCommits = commits.filter((commit) => commit.type === "feat");
const fixedCommits = commits.filter((commit) => commit.type === "fix");
const changedCommits = commits.filter(
  (commit) =>
    commit.type === "refactor" ||
    commit.type === "perf" ||
    commit.type === "chore" ||
    commit.type === "docs" ||
    commit.type === "test",
);

const releaseNotesDir = resolve("docs/release-notes");
mkdirSync(releaseNotesDir, { recursive: true });
const outputPath = resolve(releaseNotesDir, `v${targetVersion}-draft.md`);

const generatedAt = new Date().toISOString();
const markdown = [
  `# Release Notes Draft v${targetVersion}`,
  "",
  `- Generated: ${generatedAt}`,
  `- Commit range: ${range}`,
  "",
  "## Parity Highlights",
  "",
  ...(parityCommits.length > 0 ? commitBullets(parityCommits) : ["- No new parity commits in this range."]),
  "",
  "## Added",
  "",
  ...(addedCommits.length > 0 ? commitBullets(addedCommits) : ["- None."]),
  "",
  "## Changed",
  "",
  ...(changedCommits.length > 0 ? commitBullets(changedCommits) : ["- None."]),
  "",
  "## Fixed",
  "",
  ...(fixedCommits.length > 0 ? commitBullets(fixedCommits) : ["- None."]),
  "",
].join("\n");

writeFileSync(outputPath, markdown, "utf-8");
const changelogChanged = updateChangelog(
  resolve("CHANGELOG.md"),
  commitBullets(parityCommits),
  latestTag,
);

console.log(`Wrote release notes draft: ${outputPath}`);
if (changelogChanged) {
  console.log("Updated CHANGELOG.md parity automation block.");
} else {
  console.log("CHANGELOG.md parity automation block already up to date.");
}
