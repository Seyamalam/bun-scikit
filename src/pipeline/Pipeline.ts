import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertNonEmptyMatrix } from "../utils/validation";
import { fitWithSampleWeight, type FitSampleWeightRequest } from "../utils/fitWithSampleWeight";

type StepValue = Record<string, unknown>;

interface RuntimeStep {
  name: string;
  value: StepValue;
}

interface Fittable {
  fit(X: Matrix, y?: Vector, sampleWeight?: Vector): unknown;
}

interface TransformLike extends Fittable {
  transform(X: Matrix): Matrix;
  fitTransform?: (X: Matrix, y?: Vector, sampleWeight?: Vector) => Matrix;
}

interface PredictLike extends Fittable {
  predict(X: Matrix): Vector;
}

interface ScoreLike extends Fittable {
  score(X: Matrix, y: Vector): number;
}

interface PredictProbaLike extends Fittable {
  predictProba(X: Matrix): Matrix;
}

interface DecisionFunctionLike extends Fittable {
  decisionFunction(X: Matrix): Vector | Matrix;
}

interface SetParamsLike {
  setParams(params: Record<string, unknown>): unknown;
}

interface GetParamsLike {
  getParams(deep?: boolean): Record<string, unknown>;
}

function isObject(value: unknown): value is StepValue {
  return typeof value === "object" && value !== null;
}

function hasFit(value: StepValue): value is StepValue & Fittable {
  return typeof value.fit === "function";
}

function hasTransform(value: StepValue): value is StepValue & TransformLike {
  return typeof value.transform === "function" && hasFit(value);
}

function hasPredict(value: StepValue): value is StepValue & PredictLike {
  return typeof value.predict === "function" && hasFit(value);
}

function hasScore(value: StepValue): value is StepValue & ScoreLike {
  return typeof value.score === "function" && hasFit(value);
}

function hasPredictProba(value: StepValue): value is StepValue & PredictProbaLike {
  return typeof value.predictProba === "function" && hasFit(value);
}

function hasDecisionFunction(value: StepValue): value is StepValue & DecisionFunctionLike {
  return typeof value.decisionFunction === "function" && hasFit(value);
}

function hasSetParams(value: StepValue): value is StepValue & SetParamsLike {
  return typeof value.setParams === "function";
}

function hasGetParams(value: StepValue): value is StepValue & GetParamsLike {
  return typeof value.getParams === "function";
}

function fallbackGetParams(value: StepValue): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  for (const [key, current] of Object.entries(value)) {
    if (typeof current !== "function" && !key.endsWith("_")) {
      params[key] = current;
    }
  }
  return params;
}

function fitTransformStep(
  step: TransformLike,
  X: Matrix,
  y?: Vector,
  sampleWeight?: Vector,
): Matrix {
  if (typeof step.fitTransform === "function") {
    if (sampleWeight) {
      return step.fitTransform(X, y, sampleWeight);
    }
    return step.fitTransform(X, y);
  }
  fitWithSampleWeight(step, X, y, sampleWeight);
  return step.transform(X);
}

export type PipelineStep = [name: string, step: unknown];

export class Pipeline {
  steps_: ReadonlyArray<readonly [string, unknown]> = [];
  namedSteps_: Record<string, unknown> = {};
  private runtimeSteps: RuntimeStep[];
  private isFitted = false;
  sampleWeightRequest_ = true;

  constructor(steps: PipelineStep[]) {
    if (steps.length === 0) {
      throw new Error("Pipeline requires at least one step.");
    }

    const seen = new Set<string>();
    const runtime: RuntimeStep[] = [];

    for (const [name, step] of steps) {
      if (typeof name !== "string" || name.trim().length === 0) {
        throw new Error("Pipeline step names must be non-empty strings.");
      }
      if (seen.has(name)) {
        throw new Error(`Pipeline step names must be unique. Duplicate step: '${name}'.`);
      }
      if (!isObject(step)) {
        throw new Error(`Pipeline step '${name}' must be an object.`);
      }
      seen.add(name);
      runtime.push({ name, value: step });
    }

    this.runtimeSteps = runtime;
    this.refreshStepViews();
  }

  fit(X: Matrix, y?: Vector, sampleWeight?: Vector): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);

    const routedSampleWeight = this.sampleWeightRequest_ ? sampleWeight : undefined;

    let transformedX = X;
    const lastIndex = this.runtimeSteps.length - 1;

    for (let i = 0; i < lastIndex; i += 1) {
      const current = this.runtimeSteps[i];
      if (!hasTransform(current.value)) {
        throw new Error(
          `Pipeline step '${current.name}' must implement fit() and transform() because it is not the final step.`,
        );
      }
      transformedX = fitTransformStep(current.value, transformedX, y, routedSampleWeight);
    }

    const finalStep = this.runtimeSteps[lastIndex];
    if (!hasFit(finalStep.value)) {
      throw new Error(`Pipeline final step '${finalStep.name}' must implement fit().`);
    }

    if (!hasTransform(finalStep.value) && y === undefined) {
      throw new Error(`Pipeline final step '${finalStep.name}' requires target labels y for fit().`);
    }

    fitWithSampleWeight(finalStep.value, transformedX, y, routedSampleWeight);
    this.isFitted = true;
    return this;
  }

  predict(X: Matrix): Vector {
    this.assertFitted();

    const lastStep = this.runtimeSteps[this.runtimeSteps.length - 1];
    if (!hasPredict(lastStep.value)) {
      throw new Error(`Pipeline final step '${lastStep.name}' does not implement predict().`);
    }

    const transformed = this.transformThroughIntermediates(X);
    return lastStep.value.predict(transformed);
  }

  predictProba(X: Matrix): Matrix {
    this.assertFitted();
    const lastStep = this.runtimeSteps[this.runtimeSteps.length - 1];
    if (!hasPredictProba(lastStep.value)) {
      throw new Error(`Pipeline final step '${lastStep.name}' does not implement predictProba().`);
    }
    const transformed = this.transformThroughIntermediates(X);
    return lastStep.value.predictProba(transformed);
  }

  decisionFunction(X: Matrix): Vector | Matrix {
    this.assertFitted();
    const lastStep = this.runtimeSteps[this.runtimeSteps.length - 1];
    if (!hasDecisionFunction(lastStep.value)) {
      throw new Error(
        `Pipeline final step '${lastStep.name}' does not implement decisionFunction().`,
      );
    }
    const transformed = this.transformThroughIntermediates(X);
    return lastStep.value.decisionFunction(transformed);
  }

  score(X: Matrix, y: Vector): number {
    this.assertFitted();
    const lastStep = this.runtimeSteps[this.runtimeSteps.length - 1];
    if (!hasScore(lastStep.value)) {
      throw new Error(`Pipeline final step '${lastStep.name}' does not implement score().`);
    }
    const transformed = this.transformThroughIntermediates(X);
    return lastStep.value.score(transformed, y);
  }

  transform(X: Matrix): Matrix {
    this.assertFitted();

    const transformed = this.transformThroughIntermediates(X);
    const lastStep = this.runtimeSteps[this.runtimeSteps.length - 1];
    if (!hasTransform(lastStep.value)) {
      throw new Error(`Pipeline final step '${lastStep.name}' does not implement transform().`);
    }

    return lastStep.value.transform(transformed);
  }

  fitTransform(X: Matrix, y?: Vector, sampleWeight?: Vector): Matrix {
    this.fit(X, y, sampleWeight);
    return this.transform(X);
  }

  setFitRequest(request: FitSampleWeightRequest): this {
    if (typeof request.sampleWeight === "boolean") {
      this.sampleWeightRequest_ = request.sampleWeight;
    }
    return this;
  }

  getParams(deep = true): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    for (let i = 0; i < this.runtimeSteps.length; i += 1) {
      const { name, value } = this.runtimeSteps[i];
      params[name] = value;
      if (deep) {
        const nested = hasGetParams(value) ? value.getParams(true) : fallbackGetParams(value);
        for (const [key, nestedValue] of Object.entries(nested)) {
          params[`${name}__${key}`] = nestedValue;
        }
      }
    }
    return params;
  }

  setParams(params: Record<string, unknown>): this {
    const nestedByStep = new Map<string, Record<string, unknown>>();
    for (const [rawKey, rawValue] of Object.entries(params)) {
      if (rawKey.includes("__")) {
        const splitIndex = rawKey.indexOf("__");
        const stepName = rawKey.slice(0, splitIndex);
        const nestedKey = rawKey.slice(splitIndex + 2);
        if (!this.namedSteps_.hasOwnProperty(stepName)) {
          throw new Error(`Unknown pipeline step '${stepName}' in parameter '${rawKey}'.`);
        }
        const bucket = nestedByStep.get(stepName);
        if (bucket) {
          bucket[nestedKey] = rawValue;
        } else {
          nestedByStep.set(stepName, { [nestedKey]: rawValue });
        }
        continue;
      }

      if (!this.namedSteps_.hasOwnProperty(rawKey)) {
        throw new Error(`Unknown pipeline parameter '${rawKey}'.`);
      }
      if (!isObject(rawValue)) {
        throw new Error(`Pipeline step replacement for '${rawKey}' must be an object.`);
      }
      const step = this.runtimeSteps.find((entry) => entry.name === rawKey);
      if (!step) {
        throw new Error(`Unknown pipeline step '${rawKey}'.`);
      }
      step.value = rawValue;
      this.isFitted = false;
    }

    for (const [stepName, nested] of nestedByStep.entries()) {
      const step = this.runtimeSteps.find((entry) => entry.name === stepName);
      if (!step) {
        throw new Error(`Unknown pipeline step '${stepName}'.`);
      }
      if (hasSetParams(step.value)) {
        step.value.setParams(nested);
      } else {
        for (const [key, value] of Object.entries(nested)) {
          (step.value as Record<string, unknown>)[key] = value;
        }
      }
      this.isFitted = false;
    }

    this.refreshStepViews();
    return this;
  }

  private transformThroughIntermediates(X: Matrix): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);

    let transformedX = X;
    const lastIndex = this.runtimeSteps.length - 1;
    for (let i = 0; i < lastIndex; i += 1) {
      const current = this.runtimeSteps[i];
      if (!hasTransform(current.value)) {
        throw new Error(`Pipeline step '${current.name}' must implement transform() for inference.`);
      }
      transformedX = current.value.transform(transformedX);
    }
    return transformedX;
  }

  private refreshStepViews(): void {
    this.steps_ = this.runtimeSteps.map((step) => [step.name, step.value] as const);
    this.namedSteps_ = Object.fromEntries(this.runtimeSteps.map((step) => [step.name, step.value]));
  }

  private assertFitted(): void {
    if (!this.isFitted) {
      throw new Error("Pipeline has not been fitted.");
    }
  }
}
