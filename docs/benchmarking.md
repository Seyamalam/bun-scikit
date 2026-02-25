# Benchmarking

`bun-scikit` benchmarks compare Bun and Python scikit-learn on the same dataset:

- Dataset: `test_data/heart.csv`
- Pipelines:
  - Regression: `StandardScaler + LinearRegression`
  - Classification: `StandardScaler + LogisticRegression`
  - Tree classification:
    - `DecisionTreeClassifier`
    - `RandomForestClassifier`
- Split: deterministic, `randomState=42`, test fraction `0.2`

## Commands

- Run local benchmark summary:
  - `bun run bench`
- Generate current CI-style snapshot:
  - `bun run bench:ci`
- Generate CI-style snapshot with native Zig kernels:
  - `bun run bench:ci:native`
- Generate latest snapshot and sync README benchmark table:
  - `bun run bench:snapshot`
  - This command uses `bench/results/heart-ci-latest.json` as the source of truth and rewrites the README benchmark section.
- Append latest snapshot summary to history:
  - `bun run bench:history:update`
- Run local classification benchmark:
  - `bun run bench:heart:classification`
- Run local tree benchmark:
  - `bun run bench:heart:tree`
- Run synthetic tree/forest hot-path benchmark (`js-fast` vs `zig-tree`):
  - `bun run bench:hotpaths`
  - The benchmark forces `BUN_SCIKIT_TREE_BACKEND=js` for `js-fast` and `BUN_SCIKIT_TREE_BACKEND=zig` for `zig-tree`.
- Run hot-path regression guard against committed baseline:
  - `bun run bench:hotpaths:check`
  - Uses `bench/results/tree-hotpaths-current.json` compared to `bench/results/tree-hotpaths-baseline.json`.
  - CI runs `bench:hotpaths` + `bench:hotpaths:check` as a per-kernel performance gate.
- Verify README benchmark section is synced:
  - `bun run bench:readme:check`
- Build benchmark runner with Bun bytecode + minify (startup optimization):
  - `bun run build:bench:bytecode`
  - `bun run build:bench:compiled`
- Build native Zig kernels used by accelerated fit paths:
  - `bun run native:build`

## Python Dependencies

Install once for comparison benchmarks:

```bash
python -m pip install -r bench/python/requirements.txt
```

## CI Workflows

- `CI` workflow runs tests, typecheck, and benchmark comparison on pushes/PRs.
  Benchmark job builds Zig kernels before benchmark execution.
- `Benchmark Snapshot` workflow (manual/scheduled) regenerates:
  - `bench/results/heart-ci-latest.json`
  - `bench/results/heart-ci-latest.md`
  - `bench/results/history/heart-ci-history.jsonl`
  - README benchmark table section

This keeps the README benchmark table sourced from CI-run benchmark commands.

## Bytecode Note

`--bytecode` and `--minify` primarily reduce startup/parsing overhead. They are
useful for CLI responsiveness, but they do not materially speed up the model
training loops themselves. Fit-time improvements mostly come from algorithm and
data-structure optimizations.
