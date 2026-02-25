import { DecisionTreeClassifier, RandomForestClassifier } from "../src";

const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [2, 2],
  [2, 3],
  [3, 2],
];
const y = [0, 0, 0, 1, 1, 1];

const tree = new DecisionTreeClassifier({ maxDepth: 3, randomState: 42 });
tree.fit(X, y);
console.log("DecisionTree fit backend:", tree.fitBackend_, tree.fitBackendLibrary_);

const forest = new RandomForestClassifier({ nEstimators: 25, maxDepth: 4, randomState: 42 });
forest.fit(X, y);
console.log("RandomForest fit backend:", forest.fitBackend_, forest.fitBackendLibrary_);
