# Release Checklist

Use this checklist for each release.

## 1) Pre-release validation

- Ensure `main` CI is green.
- Run `Release Prep` workflow (or rely on release-triggered `release-prep` gate) and confirm it passes:
  - parity matrix coverage (`bun run parity:matrix:check`)
  - tests
  - sklearn snapshot parity checks (`bun run parity:check`, including `test/sklearn-snapshots.test.ts`)
  - typecheck
  - Zig backend guard test
  - tree hot-path microbench + `bench:hotpaths:check`
  - benchmark run + `bench:health`
  - `bench:readme:check`
  - `npm pack` consumer smoke test

## 2) Version bump

- Update `package.json` version.
- Commit the bump.
- Push `main`.

## 3) Create release tag

- Create GitHub release with tag `vX.Y.Z` pointing to `main`.
- This triggers:
  - `Release Prep` (blocking gate)
  - `Release Native Prebuilds` (linux/windows assets)
  - `Publish to npm` (if `NPM_TOKEN` is configured)

## 4) Verify release artifacts

- Confirm release includes:
  - `bun_scikit_kernels-vX.Y.Z-linux-x64.so`
  - `bun_scikit_kernels-vX.Y.Z-windows-x64.dll`
  - `bun_scikit_node_addon-vX.Y.Z-linux-x64.node`
  - `bun_scikit_node_addon-vX.Y.Z-windows-x64.node`

## 5) Verify npm publish

- Confirm workflow `Publish to npm` succeeded.
- Confirm registry version:
  - `npm view bun-scikit version`

## 6) Post-release checks

- Run a clean consumer smoke test:
  - `bun init -y`
  - `bun add bun-scikit`
  - fit/predict with `LinearRegression` and `LogisticRegression`
- Update `CHANGELOG.md` with release notes if needed.
