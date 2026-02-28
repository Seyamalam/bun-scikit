import { expandParamGrid, type ParamGrid } from "./ParameterGrid";

export type ParamDistributions = ParamGrid;

class Mulberry32 {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state ^ (this.state >>> 15);
    t = Math.imul(t, this.state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  nextInt(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }
}

function assertValidDistributions(distributions: ParamDistributions): void {
  const keys = Object.keys(distributions);
  if (keys.length === 0) {
    throw new Error("paramDistributions must include at least one parameter.");
  }
  for (let i = 0; i < keys.length; i += 1) {
    const values = distributions[keys[i]];
    if (!Array.isArray(values) || values.length === 0) {
      throw new Error(`paramDistributions '${keys[i]}' must be a non-empty array.`);
    }
  }
}

function allDiscreteLists(distributions: ParamDistributions): boolean {
  const keys = Object.keys(distributions);
  for (let i = 0; i < keys.length; i += 1) {
    if (!Array.isArray(distributions[keys[i]])) {
      return false;
    }
  }
  return true;
}

function sampleWithoutReplacement(
  candidates: Record<string, unknown>[],
  nIter: number,
  rng: Mulberry32,
): Record<string, unknown>[] {
  const copy = candidates.map((row) => ({ ...row }));
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = rng.nextInt(i + 1);
    const tmp = copy[i];
    copy[i] = copy[j];
    copy[j] = tmp;
  }
  return copy.slice(0, nIter);
}

export function drawParameterSamples(
  distributions: ParamDistributions,
  nIter: number,
  randomState = 42,
): Record<string, unknown>[] {
  if (!Number.isInteger(nIter) || nIter < 1) {
    throw new Error(`nIter must be an integer >= 1. Got ${nIter}.`);
  }
  assertValidDistributions(distributions);

  const rng = new Mulberry32(randomState);
  const keys = Object.keys(distributions);
  if (allDiscreteLists(distributions)) {
    const candidates = expandParamGrid(distributions);
    if (nIter <= candidates.length) {
      return sampleWithoutReplacement(candidates, nIter, rng);
    }
  }

  const out: Record<string, unknown>[] = [];
  for (let i = 0; i < nIter; i += 1) {
    const params: Record<string, unknown> = {};
    for (let k = 0; k < keys.length; k += 1) {
      const key = keys[k];
      const values = distributions[key];
      params[key] = values[rng.nextInt(values.length)];
    }
    out.push(params);
  }
  return out;
}

export class ParameterSampler implements Iterable<Record<string, unknown>> {
  private readonly samples: Record<string, unknown>[];

  constructor(
    distributions: ParamDistributions,
    nIter: number,
    randomState = 42,
  ) {
    this.samples = drawParameterSamples(distributions, nIter, randomState);
  }

  get length(): number {
    return this.samples.length;
  }

  toArray(): Record<string, unknown>[] {
    return this.samples.map((row) => ({ ...row }));
  }

  [Symbol.iterator](): Iterator<Record<string, unknown>> {
    return this.toArray()[Symbol.iterator]();
  }
}
