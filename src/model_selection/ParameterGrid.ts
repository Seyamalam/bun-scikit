export type ParamGrid = Record<string, readonly unknown[]>;
export type ParamGridInput = ParamGrid | ParamGrid[];

function assertValidGrid(grid: ParamGrid): void {
  const keys = Object.keys(grid);
  for (let i = 0; i < keys.length; i += 1) {
    const values = grid[keys[i]];
    if (!Array.isArray(values) || values.length === 0) {
      throw new Error(`paramGrid '${keys[i]}' must be a non-empty array.`);
    }
  }
}

function cartesianProduct(grid: ParamGrid): Record<string, unknown>[] {
  const keys = Object.keys(grid);
  if (keys.length === 0) {
    return [{}];
  }

  const out: Record<string, unknown>[] = [];
  const current: Record<string, unknown> = {};
  function recurse(depth: number): void {
    if (depth === keys.length) {
      out.push({ ...current });
      return;
    }
    const key = keys[depth];
    const values = grid[key];
    for (let i = 0; i < values.length; i += 1) {
      current[key] = values[i];
      recurse(depth + 1);
    }
  }

  recurse(0);
  return out;
}

export function expandParamGrid(paramGrid: ParamGridInput): Record<string, unknown>[] {
  const grids = Array.isArray(paramGrid) ? paramGrid : [paramGrid];
  if (grids.length === 0) {
    throw new Error("paramGrid must include at least one grid specification.");
  }

  const out: Record<string, unknown>[] = [];
  for (let i = 0; i < grids.length; i += 1) {
    const grid = grids[i];
    assertValidGrid(grid);
    const entries = cartesianProduct(grid);
    for (let j = 0; j < entries.length; j += 1) {
      out.push(entries[j]);
    }
  }
  return out;
}

export class ParameterGrid implements Iterable<Record<string, unknown>> {
  private readonly combinations: Record<string, unknown>[];

  constructor(paramGrid: ParamGridInput) {
    this.combinations = expandParamGrid(paramGrid);
  }

  get length(): number {
    return this.combinations.length;
  }

  toArray(): Record<string, unknown>[] {
    return this.combinations.map((row) => ({ ...row }));
  }

  [Symbol.iterator](): Iterator<Record<string, unknown>> {
    return this.toArray()[Symbol.iterator]();
  }
}
