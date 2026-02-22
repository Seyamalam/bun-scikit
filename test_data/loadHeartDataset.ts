export interface HeartDataset {
  featureNames: string[];
  X: number[][];
  y: number[];
}

export async function loadHeartDataset(): Promise<HeartDataset> {
  const csvText = await Bun.file(new URL("./heart.csv", import.meta.url)).text();
  const lines = csvText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length < 2) {
    throw new Error("heart.csv must include a header and at least one data row.");
  }

  const headers = lines[0].split(",");
  const targetIndex = headers.indexOf("target");
  if (targetIndex === -1) {
    throw new Error("heart.csv is missing a 'target' column.");
  }

  const featureNames = headers.filter((_, idx) => idx !== targetIndex);
  const X: number[][] = [];
  const y: number[] = [];

  for (let lineNumber = 1; lineNumber < lines.length; lineNumber += 1) {
    const cells = lines[lineNumber].split(",");
    if (cells.length !== headers.length) {
      throw new Error(
        `Row ${lineNumber + 1} has ${cells.length} columns; expected ${headers.length}.`,
      );
    }

    const featureRow: number[] = [];
    let targetValue = 0;

    for (let col = 0; col < cells.length; col += 1) {
      const value = Number.parseFloat(cells[col]);
      if (!Number.isFinite(value)) {
        throw new Error(`Row ${lineNumber + 1}, column '${headers[col]}' is not numeric.`);
      }

      if (col === targetIndex) {
        targetValue = value;
      } else {
        featureRow.push(value);
      }
    }

    X.push(featureRow);
    y.push(targetValue);
  }

  return {
    featureNames,
    X,
    y,
  };
}
