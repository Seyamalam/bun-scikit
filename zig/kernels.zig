const std = @import("std");
const linear = @import("./src/linear.zig");
const logistic = @import("./src/logistic.zig");
const tree = @import("./src/tree.zig");

const ABI_VERSION: u32 = 2;
const Status = enum(u32) {
    ok = 1,
    invalid_handle = 2,
    invalid_shape = 3,
    allocation_failed = 4,
    fit_failed = 5,
    symbol_unavailable = 6,
};

pub export fn bun_scikit_abi_version() u32 {
    return ABI_VERSION;
}

pub export fn bun_scikit_status_ok() u32 {
    return @intFromEnum(Status.ok);
}

pub export fn bun_scikit_status_invalid_handle() u32 {
    return @intFromEnum(Status.invalid_handle);
}

pub export fn bun_scikit_status_invalid_shape() u32 {
    return @intFromEnum(Status.invalid_shape);
}

pub export fn bun_scikit_status_allocation_failed() u32 {
    return @intFromEnum(Status.allocation_failed);
}

pub export fn bun_scikit_status_fit_failed() u32 {
    return @intFromEnum(Status.fit_failed);
}

pub export fn bun_scikit_status_symbol_unavailable() u32 {
    return @intFromEnum(Status.symbol_unavailable);
}

pub export fn linear_model_create(n_features: usize, fit_intercept: u8) usize {
    return linear.linear_model_create(n_features, fit_intercept);
}

pub export fn linear_model_destroy(handle: usize) void {
    linear.linear_model_destroy(handle);
}

pub export fn linear_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    l2: f64,
) u8 {
    return linear.linear_model_fit(handle, x_ptr, y_ptr, n_samples, l2);
}

pub export fn linear_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    out_ptr: [*]f64,
) u8 {
    return linear.linear_model_predict(handle, x_ptr, n_samples, out_ptr);
}

pub export fn linear_model_copy_coefficients(handle: usize, out_ptr: [*]f64) u8 {
    return linear.linear_model_copy_coefficients(handle, out_ptr);
}

pub export fn linear_model_get_intercept(handle: usize) f64 {
    return linear.linear_model_get_intercept(handle);
}

pub export fn logistic_model_create(n_features: usize, fit_intercept: u8) usize {
    return logistic.logistic_model_create(n_features, fit_intercept);
}

pub export fn logistic_model_destroy(handle: usize) void {
    logistic.logistic_model_destroy(handle);
}

pub export fn logistic_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    learning_rate: f64,
    l2: f64,
    max_iters: usize,
    tolerance: f64,
) usize {
    return logistic.logistic_model_fit(
        handle,
        x_ptr,
        y_ptr,
        n_samples,
        learning_rate,
        l2,
        max_iters,
        tolerance,
    );
}

pub export fn logistic_model_fit_lbfgs(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    max_iters: usize,
    tolerance: f64,
    l2: f64,
    memory: usize,
) usize {
    return logistic.logistic_model_fit_lbfgs(
        handle,
        x_ptr,
        y_ptr,
        n_samples,
        max_iters,
        tolerance,
        l2,
        memory,
    );
}

pub export fn logistic_model_predict_proba(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    out_positive_ptr: [*]f64,
) u8 {
    return logistic.logistic_model_predict_proba(handle, x_ptr, n_samples, out_positive_ptr);
}

pub export fn logistic_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    out_labels_ptr: [*]u8,
) u8 {
    return logistic.logistic_model_predict(handle, x_ptr, n_samples, out_labels_ptr);
}

pub export fn logistic_model_copy_coefficients(handle: usize, out_ptr: [*]f64) u8 {
    return logistic.logistic_model_copy_coefficients(handle, out_ptr);
}

pub export fn logistic_model_get_intercept(handle: usize) f64 {
    return logistic.logistic_model_get_intercept(handle);
}

pub export fn decision_tree_model_create(
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features_mode: u8,
    max_features_value: usize,
    random_state: u32,
    use_random_state: u8,
    class_count: usize,
    n_features: usize,
) usize {
    return tree.decision_tree_model_create(
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features_mode,
        max_features_value,
        random_state,
        use_random_state,
        class_count,
        n_features,
    );
}

pub export fn decision_tree_model_destroy(handle: usize) void {
    tree.decision_tree_model_destroy(handle);
}

pub export fn decision_tree_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const u16,
    n_samples: usize,
    n_features: usize,
    sample_indices_ptr: [*]const u32,
    sample_count: usize,
) u8 {
    return tree.decision_tree_model_fit(
        handle,
        x_ptr,
        y_ptr,
        n_samples,
        n_features,
        sample_indices_ptr,
        sample_count,
    );
}

pub export fn decision_tree_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    out_labels_ptr: [*]u16,
) u8 {
    return tree.decision_tree_model_predict(handle, x_ptr, n_samples, n_features, out_labels_ptr);
}

pub export fn random_forest_classifier_model_create(
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features_mode: u8,
    max_features_value: usize,
    bootstrap: u8,
    random_state: u32,
    use_random_state: u8,
    class_count: usize,
    n_features: usize,
) usize {
    return tree.random_forest_classifier_model_create(
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features_mode,
        max_features_value,
        bootstrap,
        random_state,
        use_random_state,
        class_count,
        n_features,
    );
}

pub export fn random_forest_classifier_model_destroy(handle: usize) void {
    tree.random_forest_classifier_model_destroy(handle);
}

pub export fn random_forest_classifier_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const u16,
    n_samples: usize,
    n_features: usize,
) u8 {
    return tree.random_forest_classifier_model_fit(handle, x_ptr, y_ptr, n_samples, n_features);
}

pub export fn random_forest_classifier_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    out_labels_ptr: [*]u16,
) u8 {
    return tree.random_forest_classifier_model_predict(
        handle,
        x_ptr,
        n_samples,
        n_features,
        out_labels_ptr,
    );
}

pub export fn logistic_train_epoch(
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    weights_ptr: [*]f64,
    intercept_ptr: *f64,
    gradients_ptr: [*]f64,
    learning_rate: f64,
    l2: f64,
    fit_intercept: u8,
) f64 {
    return logistic.logistic_train_epoch(
        x_ptr,
        y_ptr,
        n_samples,
        n_features,
        weights_ptr,
        intercept_ptr,
        gradients_ptr,
        learning_rate,
        l2,
        fit_intercept,
    );
}

pub export fn logistic_train_epochs(
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    weights_ptr: [*]f64,
    intercept_ptr: *f64,
    gradients_ptr: [*]f64,
    learning_rate: f64,
    l2: f64,
    fit_intercept: u8,
    max_iters: usize,
    tolerance: f64,
) usize {
    return logistic.logistic_train_epochs(
        x_ptr,
        y_ptr,
        n_samples,
        n_features,
        weights_ptr,
        intercept_ptr,
        gradients_ptr,
        learning_rate,
        l2,
        fit_intercept,
        max_iters,
        tolerance,
    );
}
