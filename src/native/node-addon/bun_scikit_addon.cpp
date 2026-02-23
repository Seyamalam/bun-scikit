#include <napi.h>

#include <cstdint>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using NativeHandle = std::uintptr_t;

using AbiVersionFn = std::uint32_t (*)();
using LinearModelCreateFn = NativeHandle (*)(std::size_t, std::uint8_t);
using LinearModelDestroyFn = void (*)(NativeHandle);
using LinearModelFitFn = std::uint8_t (*)(NativeHandle, const double*, const double*, std::size_t, double);
using LinearModelCopyCoefficientsFn = std::uint8_t (*)(NativeHandle, double*);
using LinearModelGetInterceptFn = double (*)(NativeHandle);

using LogisticModelCreateFn = NativeHandle (*)(std::size_t, std::uint8_t);
using LogisticModelDestroyFn = void (*)(NativeHandle);
using LogisticModelFitFn = std::size_t (*)(NativeHandle, const double*, const double*, std::size_t, double, double, std::size_t, double);
using LogisticModelFitLbfgsFn = std::size_t (*)(NativeHandle, const double*, const double*, std::size_t, std::size_t, double, double, std::size_t);
using LogisticModelCopyCoefficientsFn = std::uint8_t (*)(NativeHandle, double*);
using LogisticModelGetInterceptFn = double (*)(NativeHandle);
using DecisionTreeModelCreateFn = NativeHandle (*)(std::size_t, std::size_t, std::size_t, std::uint8_t, std::size_t, std::uint32_t, std::uint8_t, std::size_t);
using DecisionTreeModelDestroyFn = void (*)(NativeHandle);
using DecisionTreeModelFitFn = std::uint8_t (*)(NativeHandle, const double*, const std::uint8_t*, std::size_t, std::size_t, const std::uint32_t*, std::size_t);
using DecisionTreeModelPredictFn = std::uint8_t (*)(NativeHandle, const double*, std::size_t, std::size_t, std::uint8_t*);
using RandomForestClassifierModelCreateFn = NativeHandle (*)(
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t,
    std::uint8_t,
    std::size_t,
    std::uint8_t,
    std::uint32_t,
    std::uint8_t,
    std::size_t);
using RandomForestClassifierModelDestroyFn = void (*)(NativeHandle);
using RandomForestClassifierModelFitFn = std::uint8_t (*)(NativeHandle, const double*, const std::uint8_t*, std::size_t, std::size_t);
using RandomForestClassifierModelPredictFn = std::uint8_t (*)(NativeHandle, const double*, std::size_t, std::size_t, std::uint8_t*);

struct KernelLibrary {
#if defined(_WIN32)
  HMODULE handle{nullptr};
#else
  void* handle{nullptr};
#endif
  std::string path{};
  AbiVersionFn abi_version{nullptr};
  LinearModelCreateFn linear_model_create{nullptr};
  LinearModelDestroyFn linear_model_destroy{nullptr};
  LinearModelFitFn linear_model_fit{nullptr};
  LinearModelCopyCoefficientsFn linear_model_copy_coefficients{nullptr};
  LinearModelGetInterceptFn linear_model_get_intercept{nullptr};
  LogisticModelCreateFn logistic_model_create{nullptr};
  LogisticModelDestroyFn logistic_model_destroy{nullptr};
  LogisticModelFitFn logistic_model_fit{nullptr};
  LogisticModelFitLbfgsFn logistic_model_fit_lbfgs{nullptr};
  LogisticModelCopyCoefficientsFn logistic_model_copy_coefficients{nullptr};
  LogisticModelGetInterceptFn logistic_model_get_intercept{nullptr};
  DecisionTreeModelCreateFn decision_tree_model_create{nullptr};
  DecisionTreeModelDestroyFn decision_tree_model_destroy{nullptr};
  DecisionTreeModelFitFn decision_tree_model_fit{nullptr};
  DecisionTreeModelPredictFn decision_tree_model_predict{nullptr};
  RandomForestClassifierModelCreateFn random_forest_classifier_model_create{nullptr};
  RandomForestClassifierModelDestroyFn random_forest_classifier_model_destroy{nullptr};
  RandomForestClassifierModelFitFn random_forest_classifier_model_fit{nullptr};
  RandomForestClassifierModelPredictFn random_forest_classifier_model_predict{nullptr};
};

KernelLibrary g_library{};

void unloadLibrary() {
  if (!g_library.handle) {
    return;
  }
#if defined(_WIN32)
  FreeLibrary(g_library.handle);
#else
  dlclose(g_library.handle);
#endif
  g_library = KernelLibrary{};
}

void* lookupSymbol(const char* name) {
  if (!g_library.handle) {
    return nullptr;
  }
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(g_library.handle, name));
#else
  return dlsym(g_library.handle, name);
#endif
}

template <typename T>
T loadSymbol(const char* name) {
  return reinterpret_cast<T>(lookupSymbol(name));
}

void throwTypeError(const Napi::Env& env, const char* message) {
  Napi::TypeError::New(env, message).ThrowAsJavaScriptException();
}

void throwError(const Napi::Env& env, const char* message) {
  Napi::Error::New(env, message).ThrowAsJavaScriptException();
}

NativeHandle handleFromBigInt(const Napi::Value& value, const Napi::Env& env) {
  if (!value.IsBigInt()) {
    throwTypeError(env, "Expected a BigInt handle.");
    return 0;
  }
  bool lossless = false;
  const std::uint64_t raw = value.As<Napi::BigInt>().Uint64Value(&lossless);
  if (!lossless) {
    throwTypeError(env, "BigInt handle is not lossless as uint64.");
    return 0;
  }
  return static_cast<NativeHandle>(raw);
}

Napi::Value LoadNativeLibrary(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsString()) {
    throwTypeError(env, "loadLibrary(path) expects a string path.");
    return env.Null();
  }

  unloadLibrary();
  const std::string path = info[0].As<Napi::String>().Utf8Value();

#if defined(_WIN32)
  g_library.handle = ::LoadLibraryA(path.c_str());
#else
  g_library.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (!g_library.handle) {
    return Napi::Boolean::New(env, false);
  }

  g_library.path = path;
  g_library.abi_version = loadSymbol<AbiVersionFn>("bun_scikit_abi_version");
  g_library.linear_model_create = loadSymbol<LinearModelCreateFn>("linear_model_create");
  g_library.linear_model_destroy = loadSymbol<LinearModelDestroyFn>("linear_model_destroy");
  g_library.linear_model_fit = loadSymbol<LinearModelFitFn>("linear_model_fit");
  g_library.linear_model_copy_coefficients =
      loadSymbol<LinearModelCopyCoefficientsFn>("linear_model_copy_coefficients");
  g_library.linear_model_get_intercept =
      loadSymbol<LinearModelGetInterceptFn>("linear_model_get_intercept");
  g_library.logistic_model_create = loadSymbol<LogisticModelCreateFn>("logistic_model_create");
  g_library.logistic_model_destroy = loadSymbol<LogisticModelDestroyFn>("logistic_model_destroy");
  g_library.logistic_model_fit = loadSymbol<LogisticModelFitFn>("logistic_model_fit");
  g_library.logistic_model_fit_lbfgs =
      loadSymbol<LogisticModelFitLbfgsFn>("logistic_model_fit_lbfgs");
  g_library.logistic_model_copy_coefficients =
      loadSymbol<LogisticModelCopyCoefficientsFn>("logistic_model_copy_coefficients");
  g_library.logistic_model_get_intercept =
      loadSymbol<LogisticModelGetInterceptFn>("logistic_model_get_intercept");
  g_library.decision_tree_model_create =
      loadSymbol<DecisionTreeModelCreateFn>("decision_tree_model_create");
  g_library.decision_tree_model_destroy =
      loadSymbol<DecisionTreeModelDestroyFn>("decision_tree_model_destroy");
  g_library.decision_tree_model_fit =
      loadSymbol<DecisionTreeModelFitFn>("decision_tree_model_fit");
  g_library.decision_tree_model_predict =
      loadSymbol<DecisionTreeModelPredictFn>("decision_tree_model_predict");
  g_library.random_forest_classifier_model_create =
      loadSymbol<RandomForestClassifierModelCreateFn>("random_forest_classifier_model_create");
  g_library.random_forest_classifier_model_destroy =
      loadSymbol<RandomForestClassifierModelDestroyFn>("random_forest_classifier_model_destroy");
  g_library.random_forest_classifier_model_fit =
      loadSymbol<RandomForestClassifierModelFitFn>("random_forest_classifier_model_fit");
  g_library.random_forest_classifier_model_predict =
      loadSymbol<RandomForestClassifierModelPredictFn>("random_forest_classifier_model_predict");

  return Napi::Boolean::New(env, true);
}

Napi::Value UnloadLibrary(const Napi::CallbackInfo& info) {
  unloadLibrary();
  return info.Env().Undefined();
}

bool isLibraryLoaded(const Napi::Env& env) {
  if (!g_library.handle) {
    throwError(env, "Native library is not loaded.");
    return false;
  }
  return true;
}

Napi::Value AbiVersion(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.abi_version) {
    throwError(env, "Symbol bun_scikit_abi_version is unavailable.");
    return env.Null();
  }
  return Napi::Number::New(env, g_library.abi_version());
}

Napi::Value LoadedPath(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!g_library.handle) {
    return env.Null();
  }
  return Napi::String::New(env, g_library.path);
}

Napi::Value LinearModelCreate(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.linear_model_create) {
    throwError(env, "Symbol linear_model_create is unavailable.");
    return env.Null();
  }
  if (info.Length() != 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
    throwTypeError(env, "linearModelCreate(nFeatures, fitIntercept) expects two numbers.");
    return env.Null();
  }
  const std::size_t n_features = static_cast<std::size_t>(info[0].As<Napi::Number>().Uint32Value());
  const std::uint8_t fit_intercept = static_cast<std::uint8_t>(info[1].As<Napi::Number>().Uint32Value());
  const NativeHandle handle = g_library.linear_model_create(n_features, fit_intercept);
  return Napi::BigInt::New(env, static_cast<std::uint64_t>(handle));
}

Napi::Value LinearModelDestroy(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.linear_model_destroy) {
    throwError(env, "Symbol linear_model_destroy is unavailable.");
    return env.Null();
  }
  if (info.Length() != 1) {
    throwTypeError(env, "linearModelDestroy(handle) expects one BigInt.");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  g_library.linear_model_destroy(handle);
  return env.Undefined();
}

Napi::Value LinearModelFit(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.linear_model_fit) {
    throwError(env, "Symbol linear_model_fit is unavailable.");
    return env.Null();
  }
  if (info.Length() != 5 || !info[1].IsTypedArray() || !info[2].IsTypedArray() ||
      !info[3].IsNumber() || !info[4].IsNumber()) {
    throwTypeError(env, "linearModelFit(handle, x, y, nSamples, l2) expects (BigInt, Float64Array, Float64Array, number, number).");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto x = info[1].As<Napi::Float64Array>();
  auto y = info[2].As<Napi::Float64Array>();
  const std::size_t n_samples = static_cast<std::size_t>(info[3].As<Napi::Number>().Uint32Value());
  const double l2 = info[4].As<Napi::Number>().DoubleValue();
  const std::uint8_t status = g_library.linear_model_fit(handle, x.Data(), y.Data(), n_samples, l2);
  return Napi::Number::New(env, status);
}

Napi::Value LinearModelCopyCoefficients(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.linear_model_copy_coefficients) {
    throwError(env, "Symbol linear_model_copy_coefficients is unavailable.");
    return env.Null();
  }
  if (info.Length() != 2 || !info[1].IsTypedArray()) {
    throwTypeError(env, "linearModelCopyCoefficients(handle, out) expects (BigInt, Float64Array).");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto out = info[1].As<Napi::Float64Array>();
  const std::uint8_t status = g_library.linear_model_copy_coefficients(handle, out.Data());
  return Napi::Number::New(env, status);
}

Napi::Value LinearModelGetIntercept(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.linear_model_get_intercept) {
    throwError(env, "Symbol linear_model_get_intercept is unavailable.");
    return env.Null();
  }
  if (info.Length() != 1) {
    throwTypeError(env, "linearModelGetIntercept(handle) expects one BigInt.");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  return Napi::Number::New(env, g_library.linear_model_get_intercept(handle));
}

Napi::Value LogisticModelCreate(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.logistic_model_create) {
    throwError(env, "Symbol logistic_model_create is unavailable.");
    return env.Null();
  }
  if (info.Length() != 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
    throwTypeError(env, "logisticModelCreate(nFeatures, fitIntercept) expects two numbers.");
    return env.Null();
  }
  const std::size_t n_features = static_cast<std::size_t>(info[0].As<Napi::Number>().Uint32Value());
  const std::uint8_t fit_intercept = static_cast<std::uint8_t>(info[1].As<Napi::Number>().Uint32Value());
  const NativeHandle handle = g_library.logistic_model_create(n_features, fit_intercept);
  return Napi::BigInt::New(env, static_cast<std::uint64_t>(handle));
}

Napi::Value LogisticModelDestroy(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.logistic_model_destroy) {
    throwError(env, "Symbol logistic_model_destroy is unavailable.");
    return env.Null();
  }
  if (info.Length() != 1) {
    throwTypeError(env, "logisticModelDestroy(handle) expects one BigInt.");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  g_library.logistic_model_destroy(handle);
  return env.Undefined();
}

Napi::Value LogisticModelFit(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.logistic_model_fit) {
    throwError(env, "Symbol logistic_model_fit is unavailable.");
    return env.Null();
  }
  if (info.Length() != 8 || !info[1].IsTypedArray() || !info[2].IsTypedArray()) {
    throwTypeError(env, "logisticModelFit(handle, x, y, nSamples, learningRate, l2, maxIter, tolerance) has invalid arguments.");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto x = info[1].As<Napi::Float64Array>();
  auto y = info[2].As<Napi::Float64Array>();
  const std::size_t n_samples = static_cast<std::size_t>(info[3].As<Napi::Number>().Uint32Value());
  const double learning_rate = info[4].As<Napi::Number>().DoubleValue();
  const double l2 = info[5].As<Napi::Number>().DoubleValue();
  const std::size_t max_iter = static_cast<std::size_t>(info[6].As<Napi::Number>().Uint32Value());
  const double tolerance = info[7].As<Napi::Number>().DoubleValue();
  const std::size_t epochs = g_library.logistic_model_fit(
      handle, x.Data(), y.Data(), n_samples, learning_rate, l2, max_iter, tolerance);
  return Napi::BigInt::New(env, static_cast<std::uint64_t>(epochs));
}

Napi::Value LogisticModelFitLbfgs(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.logistic_model_fit_lbfgs) {
    throwError(env, "Symbol logistic_model_fit_lbfgs is unavailable.");
    return env.Null();
  }
  if (info.Length() != 8 || !info[1].IsTypedArray() || !info[2].IsTypedArray()) {
    throwTypeError(env, "logisticModelFitLbfgs(handle, x, y, nSamples, maxIter, tolerance, l2, memory) has invalid arguments.");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto x = info[1].As<Napi::Float64Array>();
  auto y = info[2].As<Napi::Float64Array>();
  const std::size_t n_samples = static_cast<std::size_t>(info[3].As<Napi::Number>().Uint32Value());
  const std::size_t max_iter = static_cast<std::size_t>(info[4].As<Napi::Number>().Uint32Value());
  const double tolerance = info[5].As<Napi::Number>().DoubleValue();
  const double l2 = info[6].As<Napi::Number>().DoubleValue();
  const std::size_t memory = static_cast<std::size_t>(info[7].As<Napi::Number>().Uint32Value());
  const std::size_t epochs = g_library.logistic_model_fit_lbfgs(
      handle, x.Data(), y.Data(), n_samples, max_iter, tolerance, l2, memory);
  return Napi::BigInt::New(env, static_cast<std::uint64_t>(epochs));
}

Napi::Value LogisticModelCopyCoefficients(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.logistic_model_copy_coefficients) {
    throwError(env, "Symbol logistic_model_copy_coefficients is unavailable.");
    return env.Null();
  }
  if (info.Length() != 2 || !info[1].IsTypedArray()) {
    throwTypeError(env, "logisticModelCopyCoefficients(handle, out) expects (BigInt, Float64Array).");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto out = info[1].As<Napi::Float64Array>();
  const std::uint8_t status = g_library.logistic_model_copy_coefficients(handle, out.Data());
  return Napi::Number::New(env, status);
}

Napi::Value LogisticModelGetIntercept(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.logistic_model_get_intercept) {
    throwError(env, "Symbol logistic_model_get_intercept is unavailable.");
    return env.Null();
  }
  if (info.Length() != 1) {
    throwTypeError(env, "logisticModelGetIntercept(handle) expects one BigInt.");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  return Napi::Number::New(env, g_library.logistic_model_get_intercept(handle));
}

Napi::Value DecisionTreeModelCreate(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.decision_tree_model_create) {
    throwError(env, "Symbol decision_tree_model_create is unavailable.");
    return env.Null();
  }
  if (info.Length() != 8 || !info[0].IsNumber() || !info[1].IsNumber() || !info[2].IsNumber() ||
      !info[3].IsNumber() || !info[4].IsNumber() || !info[5].IsNumber() || !info[6].IsNumber() ||
      !info[7].IsNumber()) {
    throwTypeError(env, "decisionTreeModelCreate(maxDepth, minSamplesSplit, minSamplesLeaf, maxFeaturesMode, maxFeaturesValue, randomState, useRandomState, nFeatures) expects eight numbers.");
    return env.Null();
  }

  const std::size_t max_depth = static_cast<std::size_t>(info[0].As<Napi::Number>().Uint32Value());
  const std::size_t min_samples_split = static_cast<std::size_t>(info[1].As<Napi::Number>().Uint32Value());
  const std::size_t min_samples_leaf = static_cast<std::size_t>(info[2].As<Napi::Number>().Uint32Value());
  const std::uint8_t max_features_mode = static_cast<std::uint8_t>(info[3].As<Napi::Number>().Uint32Value());
  const std::size_t max_features_value = static_cast<std::size_t>(info[4].As<Napi::Number>().Uint32Value());
  const std::uint32_t random_state = static_cast<std::uint32_t>(info[5].As<Napi::Number>().Uint32Value());
  const std::uint8_t use_random_state = static_cast<std::uint8_t>(info[6].As<Napi::Number>().Uint32Value());
  const std::size_t n_features = static_cast<std::size_t>(info[7].As<Napi::Number>().Uint32Value());

  const NativeHandle handle = g_library.decision_tree_model_create(
      max_depth,
      min_samples_split,
      min_samples_leaf,
      max_features_mode,
      max_features_value,
      random_state,
      use_random_state,
      n_features);
  return Napi::BigInt::New(env, static_cast<std::uint64_t>(handle));
}

Napi::Value DecisionTreeModelDestroy(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.decision_tree_model_destroy) {
    throwError(env, "Symbol decision_tree_model_destroy is unavailable.");
    return env.Null();
  }
  if (info.Length() != 1) {
    throwTypeError(env, "decisionTreeModelDestroy(handle) expects one BigInt.");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  g_library.decision_tree_model_destroy(handle);
  return env.Undefined();
}

Napi::Value DecisionTreeModelFit(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.decision_tree_model_fit) {
    throwError(env, "Symbol decision_tree_model_fit is unavailable.");
    return env.Null();
  }
  if (info.Length() != 7 || !info[1].IsTypedArray() || !info[2].IsTypedArray() ||
      !info[3].IsNumber() || !info[4].IsNumber() || !info[5].IsTypedArray() || !info[6].IsNumber()) {
    throwTypeError(env, "decisionTreeModelFit(handle, x, y, nSamples, nFeatures, sampleIndices, sampleCount) has invalid arguments.");
    return env.Null();
  }

  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto x = info[1].As<Napi::Float64Array>();
  auto y = info[2].As<Napi::Uint8Array>();
  const std::size_t n_samples = static_cast<std::size_t>(info[3].As<Napi::Number>().Uint32Value());
  const std::size_t n_features = static_cast<std::size_t>(info[4].As<Napi::Number>().Uint32Value());
  auto sample_indices = info[5].As<Napi::Uint32Array>();
  const std::size_t sample_count = static_cast<std::size_t>(info[6].As<Napi::Number>().Uint32Value());

  const std::uint8_t status = g_library.decision_tree_model_fit(
      handle,
      x.Data(),
      y.Data(),
      n_samples,
      n_features,
      sample_indices.Data(),
      sample_count);
  return Napi::Number::New(env, status);
}

Napi::Value DecisionTreeModelPredict(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.decision_tree_model_predict) {
    throwError(env, "Symbol decision_tree_model_predict is unavailable.");
    return env.Null();
  }
  if (info.Length() != 5 || !info[1].IsTypedArray() || !info[2].IsNumber() || !info[3].IsNumber() ||
      !info[4].IsTypedArray()) {
    throwTypeError(env, "decisionTreeModelPredict(handle, x, nSamples, nFeatures, outLabels) has invalid arguments.");
    return env.Null();
  }

  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto x = info[1].As<Napi::Float64Array>();
  const std::size_t n_samples = static_cast<std::size_t>(info[2].As<Napi::Number>().Uint32Value());
  const std::size_t n_features = static_cast<std::size_t>(info[3].As<Napi::Number>().Uint32Value());
  auto out_labels = info[4].As<Napi::Uint8Array>();

  const std::uint8_t status = g_library.decision_tree_model_predict(
      handle,
      x.Data(),
      n_samples,
      n_features,
      out_labels.Data());
  return Napi::Number::New(env, status);
}

Napi::Value RandomForestClassifierModelCreate(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.random_forest_classifier_model_create) {
    throwError(env, "Symbol random_forest_classifier_model_create is unavailable.");
    return env.Null();
  }
  if (info.Length() != 10 || !info[0].IsNumber() || !info[1].IsNumber() || !info[2].IsNumber() ||
      !info[3].IsNumber() || !info[4].IsNumber() || !info[5].IsNumber() || !info[6].IsNumber() ||
      !info[7].IsNumber() || !info[8].IsNumber() || !info[9].IsNumber()) {
    throwTypeError(env, "randomForestClassifierModelCreate(nEstimators, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeaturesMode, maxFeaturesValue, bootstrap, randomState, useRandomState, nFeatures) expects ten numbers.");
    return env.Null();
  }

  const std::size_t n_estimators = static_cast<std::size_t>(info[0].As<Napi::Number>().Uint32Value());
  const std::size_t max_depth = static_cast<std::size_t>(info[1].As<Napi::Number>().Uint32Value());
  const std::size_t min_samples_split = static_cast<std::size_t>(info[2].As<Napi::Number>().Uint32Value());
  const std::size_t min_samples_leaf = static_cast<std::size_t>(info[3].As<Napi::Number>().Uint32Value());
  const std::uint8_t max_features_mode = static_cast<std::uint8_t>(info[4].As<Napi::Number>().Uint32Value());
  const std::size_t max_features_value = static_cast<std::size_t>(info[5].As<Napi::Number>().Uint32Value());
  const std::uint8_t bootstrap = static_cast<std::uint8_t>(info[6].As<Napi::Number>().Uint32Value());
  const std::uint32_t random_state = static_cast<std::uint32_t>(info[7].As<Napi::Number>().Uint32Value());
  const std::uint8_t use_random_state = static_cast<std::uint8_t>(info[8].As<Napi::Number>().Uint32Value());
  const std::size_t n_features = static_cast<std::size_t>(info[9].As<Napi::Number>().Uint32Value());

  const NativeHandle handle = g_library.random_forest_classifier_model_create(
      n_estimators,
      max_depth,
      min_samples_split,
      min_samples_leaf,
      max_features_mode,
      max_features_value,
      bootstrap,
      random_state,
      use_random_state,
      n_features);
  return Napi::BigInt::New(env, static_cast<std::uint64_t>(handle));
}

Napi::Value RandomForestClassifierModelDestroy(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.random_forest_classifier_model_destroy) {
    throwError(env, "Symbol random_forest_classifier_model_destroy is unavailable.");
    return env.Null();
  }
  if (info.Length() != 1) {
    throwTypeError(env, "randomForestClassifierModelDestroy(handle) expects one BigInt.");
    return env.Null();
  }
  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  g_library.random_forest_classifier_model_destroy(handle);
  return env.Undefined();
}

Napi::Value RandomForestClassifierModelFit(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.random_forest_classifier_model_fit) {
    throwError(env, "Symbol random_forest_classifier_model_fit is unavailable.");
    return env.Null();
  }
  if (info.Length() != 5 || !info[1].IsTypedArray() || !info[2].IsTypedArray() ||
      !info[3].IsNumber() || !info[4].IsNumber()) {
    throwTypeError(env, "randomForestClassifierModelFit(handle, x, y, nSamples, nFeatures) has invalid arguments.");
    return env.Null();
  }

  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto x = info[1].As<Napi::Float64Array>();
  auto y = info[2].As<Napi::Uint8Array>();
  const std::size_t n_samples = static_cast<std::size_t>(info[3].As<Napi::Number>().Uint32Value());
  const std::size_t n_features = static_cast<std::size_t>(info[4].As<Napi::Number>().Uint32Value());

  const std::uint8_t status = g_library.random_forest_classifier_model_fit(
      handle,
      x.Data(),
      y.Data(),
      n_samples,
      n_features);
  return Napi::Number::New(env, status);
}

Napi::Value RandomForestClassifierModelPredict(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  if (!isLibraryLoaded(env)) {
    return env.Null();
  }
  if (!g_library.random_forest_classifier_model_predict) {
    throwError(env, "Symbol random_forest_classifier_model_predict is unavailable.");
    return env.Null();
  }
  if (info.Length() != 5 || !info[1].IsTypedArray() || !info[2].IsNumber() || !info[3].IsNumber() ||
      !info[4].IsTypedArray()) {
    throwTypeError(env, "randomForestClassifierModelPredict(handle, x, nSamples, nFeatures, outLabels) has invalid arguments.");
    return env.Null();
  }

  const NativeHandle handle = handleFromBigInt(info[0], env);
  if (env.IsExceptionPending()) {
    return env.Null();
  }
  auto x = info[1].As<Napi::Float64Array>();
  const std::size_t n_samples = static_cast<std::size_t>(info[2].As<Napi::Number>().Uint32Value());
  const std::size_t n_features = static_cast<std::size_t>(info[3].As<Napi::Number>().Uint32Value());
  auto out_labels = info[4].As<Napi::Uint8Array>();

  const std::uint8_t status = g_library.random_forest_classifier_model_predict(
      handle,
      x.Data(),
      n_samples,
      n_features,
      out_labels.Data());
  return Napi::Number::New(env, status);
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("loadLibrary", Napi::Function::New(env, LoadNativeLibrary));
  exports.Set("unloadLibrary", Napi::Function::New(env, UnloadLibrary));
  exports.Set("loadedPath", Napi::Function::New(env, LoadedPath));
  exports.Set("abiVersion", Napi::Function::New(env, AbiVersion));

  exports.Set("linearModelCreate", Napi::Function::New(env, LinearModelCreate));
  exports.Set("linearModelDestroy", Napi::Function::New(env, LinearModelDestroy));
  exports.Set("linearModelFit", Napi::Function::New(env, LinearModelFit));
  exports.Set("linearModelCopyCoefficients", Napi::Function::New(env, LinearModelCopyCoefficients));
  exports.Set("linearModelGetIntercept", Napi::Function::New(env, LinearModelGetIntercept));

  exports.Set("logisticModelCreate", Napi::Function::New(env, LogisticModelCreate));
  exports.Set("logisticModelDestroy", Napi::Function::New(env, LogisticModelDestroy));
  exports.Set("logisticModelFit", Napi::Function::New(env, LogisticModelFit));
  exports.Set("logisticModelFitLbfgs", Napi::Function::New(env, LogisticModelFitLbfgs));
  exports.Set("logisticModelCopyCoefficients", Napi::Function::New(env, LogisticModelCopyCoefficients));
  exports.Set("logisticModelGetIntercept", Napi::Function::New(env, LogisticModelGetIntercept));

  exports.Set("decisionTreeModelCreate", Napi::Function::New(env, DecisionTreeModelCreate));
  exports.Set("decisionTreeModelDestroy", Napi::Function::New(env, DecisionTreeModelDestroy));
  exports.Set("decisionTreeModelFit", Napi::Function::New(env, DecisionTreeModelFit));
  exports.Set("decisionTreeModelPredict", Napi::Function::New(env, DecisionTreeModelPredict));
  exports.Set("randomForestClassifierModelCreate", Napi::Function::New(env, RandomForestClassifierModelCreate));
  exports.Set("randomForestClassifierModelDestroy", Napi::Function::New(env, RandomForestClassifierModelDestroy));
  exports.Set("randomForestClassifierModelFit", Napi::Function::New(env, RandomForestClassifierModelFit));
  exports.Set("randomForestClassifierModelPredict", Napi::Function::New(env, RandomForestClassifierModelPredict));

  return exports;
}

}  // namespace

NODE_API_MODULE(bun_scikit_node_addon, Init)
