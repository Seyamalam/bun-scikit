import { expect, test } from "bun:test";
import { KernelPCA } from "../src";

test("KernelPCA fitTransform preserves sample count and component width", () => {
  const X = [
    [0, 0],
    [0.1, 0.2],
    [0.2, -0.1],
    [2.0, 2.1],
    [2.2, 1.9],
    [1.8, 2.2],
  ];

  const kpca = new KernelPCA({
    nComponents: 2,
    kernel: "rbf",
    gamma: 0.5,
  });
  const transformed = kpca.fitTransform(X);
  expect(transformed.length).toBe(X.length);
  expect(transformed[0].length).toBe(2);
  expect(kpca.alphas_).not.toBeNull();
  expect(kpca.lambdas_).not.toBeNull();
});

test("KernelPCA transform is deterministic after fit", () => {
  const X = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ];
  const kpca = new KernelPCA({
    nComponents: 2,
    kernel: "poly",
    gamma: 1,
    degree: 2,
    coef0: 1,
  }).fit(X);

  const a = kpca.transform(X);
  const b = kpca.transform(X);
  expect(a).toEqual(b);
});
