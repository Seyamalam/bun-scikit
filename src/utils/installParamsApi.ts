import { CalibratedClassifierCV } from "../calibration/CalibratedClassifierCV";
import { IsolationForest } from "../anomaly/IsolationForest";
import { LocalOutlierFactor } from "../anomaly/LocalOutlierFactor";
import { AgglomerativeClustering } from "../cluster/AgglomerativeClustering";
import { Birch } from "../cluster/Birch";
import { DBSCAN } from "../cluster/DBSCAN";
import { KMeans } from "../cluster/KMeans";
import { OPTICS } from "../cluster/OPTICS";
import { SpectralClustering } from "../cluster/SpectralClustering";
import { FastICA } from "../decomposition/FastICA";
import { KernelPCA } from "../decomposition/KernelPCA";
import { NMF } from "../decomposition/NMF";
import { PCA } from "../decomposition/PCA";
import { TruncatedSVD } from "../decomposition/TruncatedSVD";
import { DummyClassifier } from "../dummy/DummyClassifier";
import { DummyRegressor } from "../dummy/DummyRegressor";
import { AdaBoostClassifier } from "../ensemble/AdaBoostClassifier";
import { BaggingClassifier } from "../ensemble/BaggingClassifier";
import { BaggingRegressor } from "../ensemble/BaggingRegressor";
import { GradientBoostingClassifier } from "../ensemble/GradientBoostingClassifier";
import { GradientBoostingRegressor } from "../ensemble/GradientBoostingRegressor";
import { HistGradientBoostingClassifier } from "../ensemble/HistGradientBoostingClassifier";
import { HistGradientBoostingRegressor } from "../ensemble/HistGradientBoostingRegressor";
import { RandomForestClassifier } from "../ensemble/RandomForestClassifier";
import { RandomForestRegressor } from "../ensemble/RandomForestRegressor";
import { StackingClassifier } from "../ensemble/StackingClassifier";
import { StackingRegressor } from "../ensemble/StackingRegressor";
import { VotingClassifier } from "../ensemble/VotingClassifier";
import { VotingRegressor } from "../ensemble/VotingRegressor";
import { VarianceThreshold } from "../feature_selection/VarianceThreshold";
import {
  SelectKBest,
  SelectPercentile,
} from "../feature_selection/univariateSelection";
import { RFECV, RFE, SelectFromModel } from "../feature_selection/modelBasedSelection";
import {
  GenericUnivariateSelect,
  SelectFdr,
  SelectFpr,
  SelectFwe,
  SequentialFeatureSelector,
} from "../feature_selection/statisticalSelection";
import { LinearRegression } from "../linear_model/LinearRegression";
import { LogisticRegression } from "../linear_model/LogisticRegression";
import { SGDClassifier } from "../linear_model/SGDClassifier";
import { SGDRegressor } from "../linear_model/SGDRegressor";
import { Isomap } from "../manifold/Isomap";
import { LocallyLinearEmbedding } from "../manifold/LocallyLinearEmbedding";
import { MDS } from "../manifold/MDS";
import { TSNE } from "../manifold/TSNE";
import { GridSearchCV } from "../model_selection/GridSearchCV";
import { GroupKFold } from "../model_selection/GroupKFold";
import { GroupShuffleSplit } from "../model_selection/GroupShuffleSplit";
import { KFold } from "../model_selection/KFold";
import { RandomizedSearchCV } from "../model_selection/RandomizedSearchCV";
import { RepeatedKFold } from "../model_selection/RepeatedKFold";
import { RepeatedStratifiedKFold } from "../model_selection/RepeatedStratifiedKFold";
import { StratifiedGroupKFold } from "../model_selection/StratifiedGroupKFold";
import { StratifiedKFold } from "../model_selection/StratifiedKFold";
import { StratifiedShuffleSplit } from "../model_selection/StratifiedShuffleSplit";
import { GaussianNB } from "../naive_bayes/GaussianNB";
import { KNeighborsClassifier } from "../neighbors/KNeighborsClassifier";
import { KNeighborsRegressor } from "../neighbors/KNeighborsRegressor";
import { Binarizer } from "../preprocessing/Binarizer";
import { KNNImputer } from "../preprocessing/KNNImputer";
import { LabelEncoder } from "../preprocessing/LabelEncoder";
import { MaxAbsScaler } from "../preprocessing/MaxAbsScaler";
import { MinMaxScaler } from "../preprocessing/MinMaxScaler";
import { Normalizer } from "../preprocessing/Normalizer";
import { OneHotEncoder } from "../preprocessing/OneHotEncoder";
import { OrdinalEncoder } from "../preprocessing/OrdinalEncoder";
import { PolynomialFeatures } from "../preprocessing/PolynomialFeatures";
import { RobustScaler } from "../preprocessing/RobustScaler";
import { SimpleImputer } from "../preprocessing/SimpleImputer";
import { StandardScaler } from "../preprocessing/StandardScaler";
import { LinearSVC } from "../svm/LinearSVC";
import { OneClassSVM } from "../svm/OneClassSVM";
import { DecisionTreeClassifier } from "../tree/DecisionTreeClassifier";
import { DecisionTreeRegressor } from "../tree/DecisionTreeRegressor";
import { OneVsOneClassifier } from "../multiclass/OneVsOneClassifier";
import { OneVsRestClassifier } from "../multiclass/OneVsRestClassifier";

type Constructor = { prototype: object };

interface ParamApiSpec {
  params: string[];
  aliases?: Record<string, string>;
  validateMethod?: string;
}

function installParamsApi(ctor: Constructor, spec: ParamApiSpec): void {
  const proto = ctor.prototype as Record<string, unknown>;
  if (typeof proto.getParams === "function" && typeof proto.setParams === "function") {
    return;
  }

  const allowed = new Set<string>(spec.params);
  const aliases = spec.aliases ?? {};
  const validateMethod = spec.validateMethod ?? "validateOptions";

  if (typeof proto.getParams !== "function") {
    proto.getParams = function getParams(_deep = true): Record<string, unknown> {
      const out: Record<string, unknown> = {};
      for (let i = 0; i < spec.params.length; i += 1) {
        const key = spec.params[i];
        const field = aliases[key] ?? key;
        out[key] = (this as Record<string, unknown>)[field];
      }
      return out;
    };
  }

  if (typeof proto.setParams !== "function") {
    proto.setParams = function setParams(params: Record<string, unknown>): unknown {
      for (const [key, value] of Object.entries(params)) {
        if (!allowed.has(key)) {
          throw new Error(`Unknown parameter '${key}'.`);
        }
        const field = aliases[key] ?? key;
        (this as Record<string, unknown>)[field] = value;
      }
      const validate = (this as Record<string, unknown>)[validateMethod];
      if (typeof validate === "function") {
        validate.call(this);
      }
      return this;
    };
  }
}

installParamsApi(StandardScaler, { params: [] });
installParamsApi(MinMaxScaler, { params: ["featureRange"] });
installParamsApi(RobustScaler, { params: ["withCentering", "withScaling", "quantileRange"] });
installParamsApi(MaxAbsScaler, { params: [] });
installParamsApi(Normalizer, { params: ["norm"] });
installParamsApi(Binarizer, { params: ["threshold"] });
installParamsApi(LabelEncoder, { params: [] });
installParamsApi(PolynomialFeatures, { params: ["degree", "includeBias", "interactionOnly"] });
installParamsApi(SimpleImputer, { params: ["strategy", "fillValue"] });
installParamsApi(OneHotEncoder, { params: ["handleUnknown"] });
installParamsApi(KNNImputer, { params: ["nNeighbors", "weights"] });
installParamsApi(OrdinalEncoder, { params: ["handleUnknown", "unknownValue"] });

installParamsApi(LinearRegression, { params: ["fitIntercept", "solver"] });
installParamsApi(LogisticRegression, {
  params: ["fitIntercept", "solver", "learningRate", "maxIter", "tolerance", "l2", "lbfgsMemory"],
});
installParamsApi(SGDClassifier, {
  params: ["loss", "fitIntercept", "learningRate", "maxIter", "tolerance", "l2"],
});
installParamsApi(SGDRegressor, {
  params: ["fitIntercept", "learningRate", "maxIter", "tolerance", "l2"],
});
installParamsApi(LinearSVC, {
  params: ["fitIntercept", "C", "learningRate", "maxIter", "tolerance"],
});
installParamsApi(OneClassSVM, { params: ["nu", "kernel", "gamma"] });
installParamsApi(KNeighborsClassifier, { params: ["nNeighbors"] });
installParamsApi(KNeighborsRegressor, { params: ["nNeighbors", "weights"] });
installParamsApi(GaussianNB, { params: ["varSmoothing"] });
installParamsApi(DecisionTreeClassifier, {
  params: ["maxDepth", "minSamplesSplit", "minSamplesLeaf", "maxFeatures", "randomState"],
});
installParamsApi(DecisionTreeRegressor, {
  params: ["maxDepth", "minSamplesSplit", "minSamplesLeaf", "maxFeatures", "randomState"],
});
installParamsApi(RandomForestClassifier, {
  params: [
    "nEstimators",
    "maxDepth",
    "minSamplesSplit",
    "minSamplesLeaf",
    "maxFeatures",
    "bootstrap",
    "randomState",
  ],
});
installParamsApi(RandomForestRegressor, {
  params: [
    "nEstimators",
    "maxDepth",
    "minSamplesSplit",
    "minSamplesLeaf",
    "maxFeatures",
    "bootstrap",
    "randomState",
  ],
});
installParamsApi(AdaBoostClassifier, { params: ["nEstimators", "learningRate", "randomState"] });
installParamsApi(GradientBoostingClassifier, {
  params: [
    "nEstimators",
    "learningRate",
    "maxDepth",
    "minSamplesSplit",
    "minSamplesLeaf",
    "subsample",
    "randomState",
  ],
});
installParamsApi(GradientBoostingRegressor, {
  params: [
    "nEstimators",
    "learningRate",
    "maxDepth",
    "minSamplesSplit",
    "minSamplesLeaf",
    "subsample",
    "randomState",
  ],
});
installParamsApi(HistGradientBoostingClassifier, {
  params: [
    "maxIter",
    "learningRate",
    "maxBins",
    "maxDepth",
    "maxLeafNodes",
    "minSamplesLeaf",
    "l2Regularization",
    "earlyStopping",
    "nIterNoChange",
    "validationFraction",
    "tolerance",
    "randomState",
  ],
});
installParamsApi(HistGradientBoostingRegressor, {
  params: [
    "maxIter",
    "learningRate",
    "maxBins",
    "maxDepth",
    "maxLeafNodes",
    "minSamplesLeaf",
    "l2Regularization",
    "earlyStopping",
    "nIterNoChange",
    "validationFraction",
    "tolerance",
    "randomState",
  ],
});
installParamsApi(BaggingClassifier, {
  params: ["nEstimators", "maxSamples", "maxFeatures", "bootstrap", "bootstrapFeatures", "randomState"],
});
installParamsApi(BaggingRegressor, {
  params: ["nEstimators", "maxSamples", "maxFeatures", "bootstrap", "bootstrapFeatures", "randomState"],
});
installParamsApi(VotingClassifier, { params: ["voting", "weights"] });
installParamsApi(VotingRegressor, { params: ["weights"] });
installParamsApi(StackingClassifier, { params: ["cv", "passthrough", "stackMethod", "randomState"] });
installParamsApi(StackingRegressor, { params: ["cv", "passthrough", "randomState"] });
installParamsApi(OneVsRestClassifier, { params: ["normalizeProba"] });
installParamsApi(OneVsOneClassifier, { params: [] });
installParamsApi(CalibratedClassifierCV, { params: ["cv", "method", "ensemble", "randomState"] });

installParamsApi(KMeans, { params: ["nClusters", "nInit", "maxIter", "tolerance", "randomState"] });
installParamsApi(DBSCAN, { params: ["eps", "minSamples"] });
installParamsApi(AgglomerativeClustering, { params: ["nClusters", "linkage", "metric"] });
installParamsApi(SpectralClustering, {
  params: ["nClusters", "affinity", "gamma", "nNeighbors", "nInit", "maxIter", "randomState"],
});
installParamsApi(Birch, { params: ["threshold", "branchingFactor", "nClusters", "computeLabels"] });
installParamsApi(OPTICS, { params: ["minSamples", "maxEps", "eps", "clusterMethod"] });
installParamsApi(IsolationForest, {
  params: ["nEstimators", "maxSamples", "contamination", "randomState"],
});
installParamsApi(LocalOutlierFactor, { params: ["nNeighbors", "contamination", "novelty"] });

installParamsApi(PCA, { params: ["nComponents", "whiten", "tolerance", "maxIter"] });
installParamsApi(TruncatedSVD, { params: ["nComponents", "nIter", "tolerance", "randomState"] });
installParamsApi(FastICA, { params: ["nComponents", "maxIter", "tolerance", "randomState"] });
installParamsApi(NMF, { params: ["nComponents", "maxIter", "tolerance", "randomState"] });
installParamsApi(KernelPCA, {
  params: ["nComponents", "kernel", "gamma", "degree", "coef0", "tolerance", "maxIter"],
});
installParamsApi(TSNE, {
  params: ["nComponents", "perplexity", "learningRate", "maxIter", "randomState"],
});
installParamsApi(Isomap, { params: ["nNeighbors", "nComponents"] });
installParamsApi(LocallyLinearEmbedding, { params: ["nNeighbors", "nComponents", "reg"] });
installParamsApi(MDS, { params: ["nComponents", "dissimilarity", "randomState", "maxIter"] });

installParamsApi(VarianceThreshold, { params: ["threshold"] });
installParamsApi(SelectKBest, { params: ["scoreFunc", "k"] });
installParamsApi(SelectPercentile, { params: ["scoreFunc", "percentile"] });
installParamsApi(SelectFromModel, {
  params: ["threshold", "maxFeatures", "prefit", "importanceGetter"],
});
installParamsApi(RFE, { params: ["nFeaturesToSelect", "step", "importanceGetter"] });
installParamsApi(RFECV, {
  params: ["cv", "scoring", "minFeaturesToSelect", "step", "importanceGetter"],
});
installParamsApi(SelectFpr, { params: ["scoreFunc", "alpha"] });
installParamsApi(SelectFdr, { params: ["scoreFunc", "alpha"] });
installParamsApi(SelectFwe, { params: ["scoreFunc", "alpha"] });
installParamsApi(GenericUnivariateSelect, { params: ["scoreFunc", "mode", "param"] });
installParamsApi(SequentialFeatureSelector, {
  params: ["nFeaturesToSelect", "direction", "scoring", "cv"],
});

installParamsApi(DummyClassifier, {
  params: ["strategy", "constant", "randomState"],
  aliases: { constant: "configuredConstant" },
});
installParamsApi(DummyRegressor, { params: ["strategy", "constant", "quantile"] });

installParamsApi(KFold, { params: ["nSplits", "shuffle", "randomState"] });
installParamsApi(GroupKFold, { params: ["nSplits"] });
installParamsApi(StratifiedKFold, { params: ["nSplits", "shuffle", "randomState"] });
installParamsApi(StratifiedShuffleSplit, { params: ["nSplits", "testSize", "trainSize", "randomState"] });
installParamsApi(RepeatedKFold, { params: ["nSplits", "nRepeats", "randomState"] });
installParamsApi(RepeatedStratifiedKFold, { params: ["nSplits", "nRepeats", "randomState"] });
installParamsApi(GroupShuffleSplit, { params: ["nSplits", "testSize", "trainSize", "randomState"] });
installParamsApi(StratifiedGroupKFold, { params: ["nSplits", "shuffle", "randomState"] });

installParamsApi(GridSearchCV, { params: ["cv", "scoring", "refit", "errorScore"] });
installParamsApi(RandomizedSearchCV, {
  params: ["cv", "scoring", "refit", "errorScore", "nIter", "randomState"],
});
