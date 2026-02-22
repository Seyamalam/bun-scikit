inline fn sigmoid(z: f64) f64 {
    if (z >= 0.0) {
        const exp_neg = @exp(-z);
        return 1.0 / (1.0 + exp_neg);
    }

    const exp_pos = @exp(z);
    return exp_pos / (1.0 + exp_pos);
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
    if (n_samples == 0 or n_features == 0) {
        return 0.0;
    }

    var j: usize = 0;
    while (j < n_features) : (j += 1) {
        gradients_ptr[j] = 0.0;
    }

    var intercept_gradient: f64 = 0.0;
    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const row_offset = i * n_features;
        var z = intercept_ptr.*;

        j = 0;
        while (j < n_features) : (j += 1) {
            z += x_ptr[row_offset + j] * weights_ptr[j];
        }

        const residual = sigmoid(z) - y_ptr[i];
        intercept_gradient += residual;

        j = 0;
        while (j < n_features) : (j += 1) {
            gradients_ptr[j] += residual * x_ptr[row_offset + j];
        }
    }

    const sample_scale = 1.0 / @as(f64, @floatFromInt(n_samples));
    var max_update: f64 = 0.0;
    j = 0;
    while (j < n_features) : (j += 1) {
        const l2_term = if (l2 > 0.0) l2 * weights_ptr[j] else 0.0;
        const delta = learning_rate * (sample_scale * gradients_ptr[j] + sample_scale * l2_term);
        weights_ptr[j] -= delta;
        const abs_delta = @abs(delta);
        if (abs_delta > max_update) {
            max_update = abs_delta;
        }
    }

    if (fit_intercept != 0) {
        const intercept_delta = learning_rate * sample_scale * intercept_gradient;
        intercept_ptr.* -= intercept_delta;
        const abs_intercept_delta = @abs(intercept_delta);
        if (abs_intercept_delta > max_update) {
            max_update = abs_intercept_delta;
        }
    }

    return max_update;
}
