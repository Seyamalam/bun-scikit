const std = @import("std");

const allocator = std.heap.page_allocator;

const LinearModel = struct {
    n_features: usize,
    fit_intercept: bool,
    coefficients: []f64,
    intercept: f64,
};

const LogisticModel = struct {
    n_features: usize,
    fit_intercept: bool,
    coefficients: []f64,
    gradients: []f64,
    intercept: f64,
};

inline fn sigmoid(z: f64) f64 {
    if (z >= 0.0) {
        const exp_neg = @exp(-z);
        return 1.0 / (1.0 + exp_neg);
    }

    const exp_pos = @exp(z);
    return exp_pos / (1.0 + exp_pos);
}

inline fn asLinearModel(handle: usize) ?*LinearModel {
    if (handle == 0) {
        return null;
    }
    return @as(*LinearModel, @ptrFromInt(handle));
}

inline fn asLogisticModel(handle: usize) ?*LogisticModel {
    if (handle == 0) {
        return null;
    }
    return @as(*LogisticModel, @ptrFromInt(handle));
}

fn solveSymmetricPositiveDefinite(
    dim: usize,
    gram: []const f64,
    rhs: []const f64,
    lower: []f64,
    forward: []f64,
    solution: []f64,
) bool {
    const epsilon = 1e-12;
    @memset(lower, 0.0);

    var i: usize = 0;
    while (i < dim) : (i += 1) {
        const row_offset_i = i * dim;
        var j: usize = 0;
        while (j <= i) : (j += 1) {
            const row_offset_j = j * dim;
            var sum = gram[row_offset_i + j];
            var k: usize = 0;
            while (k < j) : (k += 1) {
                sum -= lower[row_offset_i + k] * lower[row_offset_j + k];
            }

            if (i == j) {
                if (sum <= epsilon) {
                    return false;
                }
                lower[row_offset_i + j] = @sqrt(sum);
            } else {
                lower[row_offset_i + j] = sum / lower[row_offset_j + j];
            }
        }
    }

    i = 0;
    while (i < dim) : (i += 1) {
        const row_offset = i * dim;
        var sum = rhs[i];
        var k: usize = 0;
        while (k < i) : (k += 1) {
            sum -= lower[row_offset + k] * forward[k];
        }
        forward[i] = sum / lower[row_offset + i];
    }

    var reverse: usize = dim;
    while (reverse > 0) {
        const idx = reverse - 1;
        var sum = forward[idx];
        var k = idx + 1;
        while (k < dim) : (k += 1) {
            sum -= lower[k * dim + idx] * solution[k];
        }
        solution[idx] = sum / lower[idx * dim + idx];
        reverse -= 1;
    }

    return true;
}

pub export fn linear_model_create(n_features: usize, fit_intercept: u8) usize {
    if (n_features == 0) {
        return 0;
    }

    const model = allocator.create(LinearModel) catch return 0;
    errdefer allocator.destroy(model);
    const coefficients = allocator.alloc(f64, n_features) catch return 0;
    @memset(coefficients, 0.0);

    model.* = .{
        .n_features = n_features,
        .fit_intercept = fit_intercept != 0,
        .coefficients = coefficients,
        .intercept = 0.0,
    };
    return @intFromPtr(model);
}

pub export fn linear_model_destroy(handle: usize) void {
    const model = asLinearModel(handle) orelse return;
    allocator.free(model.coefficients);
    allocator.destroy(model);
}

pub export fn linear_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    l2: f64,
) u8 {
    const model = asLinearModel(handle) orelse return 0;
    if (n_samples == 0 or model.n_features == 0) {
        return 0;
    }

    const has_intercept: usize = if (model.fit_intercept) 1 else 0;
    const dim = model.n_features + has_intercept;
    const offset = has_intercept;

    const base_gram = allocator.alloc(f64, dim * dim) catch return 0;
    defer allocator.free(base_gram);
    const gram_attempt = allocator.alloc(f64, dim * dim) catch return 0;
    defer allocator.free(gram_attempt);
    const rhs = allocator.alloc(f64, dim) catch return 0;
    defer allocator.free(rhs);
    const lower = allocator.alloc(f64, dim * dim) catch return 0;
    defer allocator.free(lower);
    const forward = allocator.alloc(f64, dim) catch return 0;
    defer allocator.free(forward);
    const solution = allocator.alloc(f64, dim) catch return 0;
    defer allocator.free(solution);

    @memset(base_gram, 0.0);
    @memset(rhs, 0.0);

    var sample_index: usize = 0;
    while (sample_index < n_samples) : (sample_index += 1) {
        const row_offset = sample_index * model.n_features;
        const target = y_ptr[sample_index];

        var i: usize = 0;
        while (i < dim) : (i += 1) {
            const xi = if (model.fit_intercept and i == 0) 1.0 else x_ptr[row_offset + (i - offset)];
            rhs[i] += xi * target;

            const gram_row_offset = i * dim;
            var j: usize = 0;
            while (j <= i) : (j += 1) {
                const xj = if (model.fit_intercept and j == 0) 1.0 else x_ptr[row_offset + (j - offset)];
                base_gram[gram_row_offset + j] += xi * xj;
            }
        }
    }

    var row: usize = 1;
    while (row < dim) : (row += 1) {
        var col: usize = 0;
        while (col < row) : (col += 1) {
            base_gram[col * dim + row] = base_gram[row * dim + col];
        }
    }

    var regularization: f64 = if (l2 > 0.0) l2 else 1e-8;
    var attempt: usize = 0;
    while (attempt < 4) : (attempt += 1) {
        @memcpy(gram_attempt, base_gram);

        var diag: usize = 0;
        while (diag < dim) : (diag += 1) {
            if (model.fit_intercept and diag == 0) {
                continue;
            }
            gram_attempt[diag * dim + diag] += regularization;
        }

        if (solveSymmetricPositiveDefinite(dim, gram_attempt, rhs, lower, forward, solution)) {
            if (model.fit_intercept) {
                model.intercept = solution[0];
                var idx: usize = 0;
                while (idx < model.n_features) : (idx += 1) {
                    model.coefficients[idx] = solution[idx + 1];
                }
            } else {
                model.intercept = 0.0;
                @memcpy(model.coefficients, solution[0..model.n_features]);
            }
            return 1;
        }

        regularization *= 10.0;
    }

    return 0;
}

pub export fn linear_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    out_ptr: [*]f64,
) u8 {
    const model = asLinearModel(handle) orelse return 0;
    if (n_samples == 0 or model.n_features == 0) {
        return 0;
    }

    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const row_offset = i * model.n_features;
        var prediction = model.intercept;
        var j: usize = 0;
        while (j < model.n_features) : (j += 1) {
            prediction += x_ptr[row_offset + j] * model.coefficients[j];
        }
        out_ptr[i] = prediction;
    }

    return 1;
}

pub export fn linear_model_copy_coefficients(handle: usize, out_ptr: [*]f64) u8 {
    const model = asLinearModel(handle) orelse return 0;
    @memcpy(out_ptr[0..model.n_features], model.coefficients);
    return 1;
}

pub export fn linear_model_get_intercept(handle: usize) f64 {
    const model = asLinearModel(handle) orelse return 0.0;
    return model.intercept;
}

pub export fn logistic_model_create(n_features: usize, fit_intercept: u8) usize {
    if (n_features == 0) {
        return 0;
    }

    const model = allocator.create(LogisticModel) catch return 0;
    errdefer allocator.destroy(model);

    const coefficients = allocator.alloc(f64, n_features) catch return 0;
    errdefer allocator.free(coefficients);
    @memset(coefficients, 0.0);

    const gradients = allocator.alloc(f64, n_features) catch return 0;
    @memset(gradients, 0.0);

    model.* = .{
        .n_features = n_features,
        .fit_intercept = fit_intercept != 0,
        .coefficients = coefficients,
        .gradients = gradients,
        .intercept = 0.0,
    };
    return @intFromPtr(model);
}

pub export fn logistic_model_destroy(handle: usize) void {
    const model = asLogisticModel(handle) orelse return;
    allocator.free(model.coefficients);
    allocator.free(model.gradients);
    allocator.destroy(model);
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
    const model = asLogisticModel(handle) orelse return 0;
    if (n_samples == 0 or model.n_features == 0 or max_iters == 0) {
        return 0;
    }

    @memset(model.coefficients, 0.0);
    @memset(model.gradients, 0.0);
    model.intercept = 0.0;

    const sample_scale = 1.0 / @as(f64, @floatFromInt(n_samples));
    var iter: usize = 0;
    while (iter < max_iters) : (iter += 1) {
        @memset(model.gradients, 0.0);
        var intercept_gradient: f64 = 0.0;

        var i: usize = 0;
        while (i < n_samples) : (i += 1) {
            const row_offset = i * model.n_features;
            var z = model.intercept;
            var j: usize = 0;
            while (j < model.n_features) : (j += 1) {
                z += x_ptr[row_offset + j] * model.coefficients[j];
            }

            const residual = sigmoid(z) - y_ptr[i];
            intercept_gradient += residual;

            j = 0;
            while (j < model.n_features) : (j += 1) {
                model.gradients[j] += residual * x_ptr[row_offset + j];
            }
        }

        var max_update: f64 = 0.0;
        var j: usize = 0;
        while (j < model.n_features) : (j += 1) {
            const l2_term = if (l2 > 0.0) l2 * model.coefficients[j] else 0.0;
            const delta = learning_rate * (sample_scale * model.gradients[j] + sample_scale * l2_term);
            model.coefficients[j] -= delta;
            const abs_delta = @abs(delta);
            if (abs_delta > max_update) {
                max_update = abs_delta;
            }
        }

        if (model.fit_intercept) {
            const intercept_delta = learning_rate * sample_scale * intercept_gradient;
            model.intercept -= intercept_delta;
            const abs_intercept_delta = @abs(intercept_delta);
            if (abs_intercept_delta > max_update) {
                max_update = abs_intercept_delta;
            }
        }

        if (max_update < tolerance) {
            return iter + 1;
        }
    }

    return max_iters;
}

pub export fn logistic_model_predict_proba(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    out_positive_ptr: [*]f64,
) u8 {
    const model = asLogisticModel(handle) orelse return 0;
    if (n_samples == 0 or model.n_features == 0) {
        return 0;
    }

    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const row_offset = i * model.n_features;
        var z = model.intercept;
        var j: usize = 0;
        while (j < model.n_features) : (j += 1) {
            z += x_ptr[row_offset + j] * model.coefficients[j];
        }
        out_positive_ptr[i] = sigmoid(z);
    }

    return 1;
}

pub export fn logistic_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    out_labels_ptr: [*]u8,
) u8 {
    const model = asLogisticModel(handle) orelse return 0;
    if (n_samples == 0 or model.n_features == 0) {
        return 0;
    }

    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const row_offset = i * model.n_features;
        var z = model.intercept;
        var j: usize = 0;
        while (j < model.n_features) : (j += 1) {
            z += x_ptr[row_offset + j] * model.coefficients[j];
        }
        out_labels_ptr[i] = if (sigmoid(z) >= 0.5) 1 else 0;
    }

    return 1;
}

pub export fn logistic_model_copy_coefficients(handle: usize, out_ptr: [*]f64) u8 {
    const model = asLogisticModel(handle) orelse return 0;
    @memcpy(out_ptr[0..model.n_features], model.coefficients);
    return 1;
}

pub export fn logistic_model_get_intercept(handle: usize) f64 {
    const model = asLogisticModel(handle) orelse return 0.0;
    return model.intercept;
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
    if (max_iters == 0 or n_samples == 0 or n_features == 0) {
        return 0;
    }

    var iter: usize = 0;
    while (iter < max_iters) : (iter += 1) {
        const max_update = logistic_train_epoch(
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
        if (max_update < tolerance) {
            return iter + 1;
        }
    }

    return max_iters;
}
