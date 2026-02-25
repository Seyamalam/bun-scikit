const fit = @import("./tree/fit.zig");
const predict = @import("./tree/predict.zig");

pub fn decision_tree_model_create(
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features_mode: u8,
    max_features_value: usize,
    random_state: u32,
    use_random_state: u8,
    n_features: usize,
) usize {
    return fit.decision_tree_model_create(
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features_mode,
        max_features_value,
        random_state,
        use_random_state,
        n_features,
    );
}

pub fn decision_tree_model_destroy(handle: usize) void {
    fit.decision_tree_model_destroy(handle);
}

pub fn decision_tree_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    n_samples: usize,
    n_features: usize,
    sample_indices_ptr: [*]const u32,
    sample_count: usize,
) u8 {
    return fit.decision_tree_model_fit(
        handle,
        x_ptr,
        y_ptr,
        n_samples,
        n_features,
        sample_indices_ptr,
        sample_count,
    );
}

pub fn decision_tree_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    out_labels_ptr: [*]u8,
) u8 {
    return predict.decision_tree_model_predict(handle, x_ptr, n_samples, n_features, out_labels_ptr);
}

pub fn random_forest_classifier_model_create(
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features_mode: u8,
    max_features_value: usize,
    bootstrap: u8,
    random_state: u32,
    use_random_state: u8,
    n_features: usize,
) usize {
    return fit.random_forest_classifier_model_create(
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features_mode,
        max_features_value,
        bootstrap,
        random_state,
        use_random_state,
        n_features,
    );
}

pub fn random_forest_classifier_model_destroy(handle: usize) void {
    fit.random_forest_classifier_model_destroy(handle);
}

pub fn random_forest_classifier_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    n_samples: usize,
    n_features: usize,
) u8 {
    return fit.random_forest_classifier_model_fit(handle, x_ptr, y_ptr, n_samples, n_features);
}

pub fn random_forest_classifier_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    out_labels_ptr: [*]u8,
) u8 {
    return predict.random_forest_classifier_model_predict(handle, x_ptr, n_samples, n_features, out_labels_ptr);
}
