# Contributing

Thanks for contributing to `bun-scikit`.

## Development Setup

1. Install Bun `>=1.3.9`.
2. Install project dependencies:
   - `bun install`
3. (For Python comparison benchmarks) install Python deps:
   - `python -m pip install -r bench/python/requirements.txt`

## Project Commands

- Run tests: `bun run test`
- Typecheck: `bun run typecheck`
- Heart benchmark (human-readable): `bun run bench`
- Generate CI benchmark snapshot locally: `bun run bench:snapshot`

## Pull Request Expectations

1. Keep changes scoped to one objective.
2. Include tests for behavioral changes.
3. Run locally before opening PR:
   - `bun run test`
   - `bun run typecheck`
   - `bun run bench:readme:check` (if benchmark files were changed)
4. Update docs when public APIs or workflows change.
5. Add changelog entries under `## [Unreleased]` in `CHANGELOG.md`.

## Commit Guidelines

- Use clear, imperative commit messages.
- Prefer conventional prefixes (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`).

## Benchmark Policy

- Benchmark table content in `README.md` is generated from `bench/results/heart-ci-latest.json`.
- Do not hand-edit the benchmark section in `README.md`.
- Use `bun run bench:sync-readme` to regenerate from the snapshot file.
