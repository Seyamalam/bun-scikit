# Benchmarking

`bun-scikit` benchmarks compare Bun and Python scikit-learn on the same dataset:

- Dataset: `test_data/heart.csv`
- Pipeline: `StandardScaler + LinearRegression`
- Split: deterministic, `randomState=42`, test fraction `0.2`

## Commands

- Run local benchmark summary:
  - `bun run bench`
- Generate current CI-style snapshot:
  - `bun run bench:ci`
- Generate latest snapshot and sync README benchmark table:
  - `bun run bench:snapshot`
- Verify README benchmark section is synced:
  - `bun run bench:readme:check`

## Python Dependencies

Install once for comparison benchmarks:

```bash
python -m pip install -r bench/python/requirements.txt
```

## CI Workflows

- `CI` workflow runs tests, typecheck, and benchmark comparison on pushes/PRs.
- `Benchmark Snapshot` workflow (manual/scheduled) regenerates:
  - `bench/results/heart-ci-latest.json`
  - README benchmark table section

This keeps the README benchmark table sourced from CI-run benchmark commands.
