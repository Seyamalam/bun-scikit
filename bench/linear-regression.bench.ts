import { LinearRegression } from "../src/linear_model/LinearRegression";

function makeSyntheticRegressionData(
  samples: number,
  features: number,
): { X: number[][]; y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  const trueWeights = Array.from({ length: features }, (_, i) => i + 1);

  for (let i = 0; i < samples; i += 1) {
    const row = Array.from({ length: features }, (_, j) => ((i + j) % 100) / 10);
    const noise = ((i % 7) - 3) * 0.01;
    const target =
      row.reduce((sum, value, featureIdx) => sum + value * trueWeights[featureIdx], 0) +
      1 +
      noise;
    X.push(row);
    y.push(target);
  }

  return { X, y };
}

const { X, y } = makeSyntheticRegressionData(4_000, 8);
const model = new LinearRegression({ solver: "normal" });

const start = performance.now();
model.fit(X, y);
const elapsed = performance.now() - start;

console.log(`LinearRegression.fit (normal) on 4k x 8: ${elapsed.toFixed(2)}ms`);
console.log(`R2 on training data: ${model.score(X, y).toFixed(6)}`);
