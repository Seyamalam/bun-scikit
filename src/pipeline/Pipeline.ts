import type { Matrix, Vector } from "../types";
import { assertConsistentRowSize, assertFiniteMatrix, assertNonEmptyMatrix } from "../utils/validation";

type StepValue = Record<string, unknown>;

interface RuntimeStep {
  name: string;
  value: StepValue;
}

interface Fittable {
  fit(X: Matrix, y?: Vector): unknown;
}

interface TransformLike extends Fittable {
  transform(X: Matrix): Matrix;
  fitTransform?: (X: Matrix, y?: Vector) => Matrix;
}

interface PredictLike extends Fittable {
  predict(X: Matrix): Vector;
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

function fitTransformStep(step: TransformLike, X: Matrix, y?: Vector): Matrix {
  if (typeof step.fitTransform === "function") {
    return step.fitTransform(X, y);
  }
  step.fit(X, y);
  return step.transform(X);
}

export type PipelineStep = [name: string, step: unknown];

export class Pipeline {
  readonly steps_: ReadonlyArray<readonly [string, unknown]>;
  readonly namedSteps_: Record<string, unknown>;
  private readonly runtimeSteps: RuntimeStep[];
  private isFitted = false;

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
    this.steps_ = runtime.map((step) => [step.name, step.value] as const);
    this.namedSteps_ = Object.fromEntries(runtime.map((step) => [step.name, step.value]));
  }

  fit(X: Matrix, y?: Vector): this {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    let transformedX = X;
    const lastIndex = this.runtimeSteps.length - 1;

    for (let i = 0; i < lastIndex; i += 1) {
      const current = this.runtimeSteps[i];
      if (!hasTransform(current.value)) {
        throw new Error(
          `Pipeline step '${current.name}' must implement fit() and transform() because it is not the final step.`,
        );
      }
      transformedX = fitTransformStep(current.value, transformedX, y);
    }

    const finalStep = this.runtimeSteps[lastIndex];
    if (!hasFit(finalStep.value)) {
      throw new Error(`Pipeline final step '${finalStep.name}' must implement fit().`);
    }

    if (!hasTransform(finalStep.value) && y === undefined) {
      throw new Error(
        `Pipeline final step '${finalStep.name}' requires target labels y for fit().`,
      );
    }

    finalStep.value.fit(transformedX, y);
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

  transform(X: Matrix): Matrix {
    this.assertFitted();

    const transformed = this.transformThroughIntermediates(X);
    const lastStep = this.runtimeSteps[this.runtimeSteps.length - 1];
    if (!hasTransform(lastStep.value)) {
      throw new Error(`Pipeline final step '${lastStep.name}' does not implement transform().`);
    }

    return lastStep.value.transform(transformed);
  }

  fitTransform(X: Matrix, y?: Vector): Matrix {
    this.fit(X, y);
    return this.transform(X);
  }

  private transformThroughIntermediates(X: Matrix): Matrix {
    assertNonEmptyMatrix(X);
    assertConsistentRowSize(X);
    assertFiniteMatrix(X);

    let transformedX = X;
    const lastIndex = this.runtimeSteps.length - 1;
    for (let i = 0; i < lastIndex; i += 1) {
      const current = this.runtimeSteps[i];
      if (!hasTransform(current.value)) {
        throw new Error(
          `Pipeline step '${current.name}' must implement transform() for inference.`,
        );
      }
      transformedX = current.value.transform(transformedX);
    }
    return transformedX;
  }

  private assertFitted(): void {
    if (!this.isFitted) {
      throw new Error("Pipeline has not been fitted.");
    }
  }
}
