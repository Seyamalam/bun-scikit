const std = @import("std");
const common = @import("./common.zig");

pub fn logistic_model_create(n_features: usize, fit_intercept: u8) usize {
    if (n_features == 0) {
        return 0;
    }

    const model = common.allocator.create(common.LogisticModel) catch return 0;
    errdefer common.allocator.destroy(model);

    const coefficients = common.allocator.alloc(f64, n_features) catch return 0;
    errdefer common.allocator.free(coefficients);
    @memset(coefficients, 0.0);

    const gradients = common.allocator.alloc(f64, n_features) catch return 0;
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

pub fn logistic_model_destroy(handle: usize) void {
    const model = common.asLogisticModel(handle) orelse return;
    common.allocator.free(model.coefficients);
    common.allocator.free(model.gradients);
    common.allocator.destroy(model);
}

fn logisticThetaLength(model: *const common.LogisticModel) usize {
    return model.n_features + @as(usize, if (model.fit_intercept) 1 else 0);
}

fn logisticLoadThetaFromModel(model: *const common.LogisticModel, theta: []f64) void {
    if (model.fit_intercept) {
        theta[0] = model.intercept;
        @memcpy(theta[1 .. 1 + model.n_features], model.coefficients);
    } else {
        @memcpy(theta[0..model.n_features], model.coefficients);
    }
}

fn logisticStoreThetaToModel(model: *common.LogisticModel, theta: []const f64) void {
    if (model.fit_intercept) {
        model.intercept = theta[0];
        @memcpy(model.coefficients, theta[1 .. 1 + model.n_features]);
    } else {
        model.intercept = 0.0;
        @memcpy(model.coefficients, theta[0..model.n_features]);
    }
}

fn dotProduct(a: []const f64, b: []const f64) f64 {
    var sum: f64 = 0.0;
    for (a, b) |av, bv| {
        sum += av * bv;
    }
    return sum;
}

fn maxAbs(values: []const f64) f64 {
    var max_value: f64 = 0.0;
    for (values) |value| {
        const abs_value = @abs(value);
        if (abs_value > max_value) {
            max_value = abs_value;
        }
    }
    return max_value;
}

fn logisticLossAndGradient(
    model: *const common.LogisticModel,
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    l2: f64,
    theta: []const f64,
    gradient: []f64,
) f64 {
    @memset(gradient, 0.0);

    const theta_offset: usize = if (model.fit_intercept) 1 else 0;
    var total_loss: f64 = 0.0;
    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const y_value = y_ptr[i];
        const row_offset = i * model.n_features;
        var z: f64 = if (model.fit_intercept) theta[0] else 0.0;
        var j: usize = 0;
        while (j < model.n_features) : (j += 1) {
            z += x_ptr[row_offset + j] * theta[theta_offset + j];
        }

        const prediction = common.sigmoid(z);
        const residual = prediction - y_value;
        if (model.fit_intercept) {
            gradient[0] += residual;
        }
        j = 0;
        while (j < model.n_features) : (j += 1) {
            gradient[theta_offset + j] += residual * x_ptr[row_offset + j];
        }

        if (z >= 0.0) {
            total_loss += (1.0 - y_value) * z + @log(1.0 + @exp(-z));
        } else {
            total_loss += -y_value * z + @log(1.0 + @exp(z));
        }
    }

    const sample_scale = 1.0 / @as(f64, @floatFromInt(n_samples));
    total_loss *= sample_scale;
    for (gradient) |*entry| {
        entry.* *= sample_scale;
    }

    if (l2 > 0.0) {
        var l2_sum: f64 = 0.0;
        var j: usize = 0;
        while (j < model.n_features) : (j += 1) {
            const weight = theta[theta_offset + j];
            l2_sum += weight * weight;
            gradient[theta_offset + j] += sample_scale * l2 * weight;
        }
        total_loss += 0.5 * sample_scale * l2 * l2_sum;
    }

    return total_loss;
}

pub fn logistic_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    learning_rate: f64,
    l2: f64,
    max_iters: usize,
    tolerance: f64,
) usize {
    const model = common.asLogisticModel(handle) orelse return 0;
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

            const residual = common.sigmoid(z) - y_ptr[i];
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

pub fn logistic_model_fit_lbfgs(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    max_iters: usize,
    tolerance: f64,
    l2: f64,
    memory: usize,
) usize {
    const model = common.asLogisticModel(handle) orelse return 0;
    if (n_samples == 0 or model.n_features == 0 or max_iters == 0) {
        return 0;
    }

    const theta_len = logisticThetaLength(model);
    const history_size = std.math.clamp(memory, 3, 20);

    const theta = common.allocator.alloc(f64, theta_len) catch return 0;
    defer common.allocator.free(theta);
    const theta_next = common.allocator.alloc(f64, theta_len) catch return 0;
    defer common.allocator.free(theta_next);
    const gradient = common.allocator.alloc(f64, theta_len) catch return 0;
    defer common.allocator.free(gradient);
    const gradient_next = common.allocator.alloc(f64, theta_len) catch return 0;
    defer common.allocator.free(gradient_next);
    const direction = common.allocator.alloc(f64, theta_len) catch return 0;
    defer common.allocator.free(direction);
    const q = common.allocator.alloc(f64, theta_len) catch return 0;
    defer common.allocator.free(q);
    const r = common.allocator.alloc(f64, theta_len) catch return 0;
    defer common.allocator.free(r);
    const s_history = common.allocator.alloc(f64, history_size * theta_len) catch return 0;
    defer common.allocator.free(s_history);
    const y_history = common.allocator.alloc(f64, history_size * theta_len) catch return 0;
    defer common.allocator.free(y_history);
    const rho_history = common.allocator.alloc(f64, history_size) catch return 0;
    defer common.allocator.free(rho_history);
    const alpha_history = common.allocator.alloc(f64, history_size) catch return 0;
    defer common.allocator.free(alpha_history);
    const index_history = common.allocator.alloc(usize, history_size) catch return 0;
    defer common.allocator.free(index_history);

    logisticLoadThetaFromModel(model, theta);
    @memset(theta_next, 0.0);
    @memset(gradient, 0.0);
    @memset(gradient_next, 0.0);
    @memset(direction, 0.0);
    @memset(q, 0.0);
    @memset(r, 0.0);

    var loss = logisticLossAndGradient(model, x_ptr, y_ptr, n_samples, l2, theta, gradient);
    if (!std.math.isFinite(loss)) {
        return 0;
    }

    var history_count: usize = 0;
    var history_head: usize = 0;

    var iter: usize = 0;
    while (iter < max_iters) : (iter += 1) {
        if (maxAbs(gradient) < tolerance) {
            logisticStoreThetaToModel(model, theta);
            return iter + 1;
        }

        @memcpy(q, gradient);
        var loop_count: usize = 0;
        while (loop_count < history_count) : (loop_count += 1) {
            const idx = (history_head + history_size - 1 - loop_count) % history_size;
            index_history[loop_count] = idx;
            const s = s_history[idx * theta_len .. (idx + 1) * theta_len];
            const y_vec = y_history[idx * theta_len .. (idx + 1) * theta_len];
            const alpha = rho_history[idx] * dotProduct(s, q);
            alpha_history[loop_count] = alpha;
            var t: usize = 0;
            while (t < theta_len) : (t += 1) {
                q[t] -= alpha * y_vec[t];
            }
        }

        var gamma: f64 = 1.0;
        if (history_count > 0) {
            const latest_idx = (history_head + history_size - 1) % history_size;
            const s_latest = s_history[latest_idx * theta_len .. (latest_idx + 1) * theta_len];
            const y_latest = y_history[latest_idx * theta_len .. (latest_idx + 1) * theta_len];
            const sy = dotProduct(s_latest, y_latest);
            const yy = dotProduct(y_latest, y_latest);
            if (yy > 1e-20 and sy > 0.0) {
                gamma = sy / yy;
            }
        }

        for (r, q) |*entry, q_value| {
            entry.* = gamma * q_value;
        }

        var backward = history_count;
        while (backward > 0) {
            const pos = backward - 1;
            const idx = index_history[pos];
            const s = s_history[idx * theta_len .. (idx + 1) * theta_len];
            const y_vec = y_history[idx * theta_len .. (idx + 1) * theta_len];
            const beta = rho_history[idx] * dotProduct(y_vec, r);
            const alpha = alpha_history[pos];
            var t: usize = 0;
            while (t < theta_len) : (t += 1) {
                r[t] += s[t] * (alpha - beta);
            }
            backward -= 1;
        }

        for (direction, r) |*entry, r_value| {
            entry.* = -r_value;
        }

        var directional_derivative = dotProduct(direction, gradient);
        if (directional_derivative >= -1e-20) {
            for (direction, gradient) |*entry, g_value| {
                entry.* = -g_value;
            }
            directional_derivative = -dotProduct(gradient, gradient);
        }

        var step: f64 = 1.0;
        const c1: f64 = 1e-4;
        const min_step: f64 = 1e-12;
        var candidate_loss: f64 = loss;
        var accepted = false;
        while (step >= min_step) {
            var t: usize = 0;
            while (t < theta_len) : (t += 1) {
                theta_next[t] = theta[t] + step * direction[t];
            }

            candidate_loss = logisticLossAndGradient(
                model,
                x_ptr,
                y_ptr,
                n_samples,
                l2,
                theta_next,
                gradient_next,
            );
            if (std.math.isFinite(candidate_loss) and
                candidate_loss <= loss + c1 * step * directional_derivative)
            {
                accepted = true;
                break;
            }
            step *= 0.5;
        }

        if (!accepted) {
            logisticStoreThetaToModel(model, theta);
            return iter + 1;
        }

        var max_step_update: f64 = 0.0;
        var t: usize = 0;
        while (t < theta_len) : (t += 1) {
            const delta = theta_next[t] - theta[t];
            const abs_delta = @abs(delta);
            if (abs_delta > max_step_update) {
                max_step_update = abs_delta;
            }
            direction[t] = delta;
            q[t] = gradient_next[t] - gradient[t];
        }

        const sy = dotProduct(direction, q);
        if (sy > 1e-12) {
            const idx = history_head;
            @memcpy(s_history[idx * theta_len .. (idx + 1) * theta_len], direction);
            @memcpy(y_history[idx * theta_len .. (idx + 1) * theta_len], q);
            rho_history[idx] = 1.0 / sy;
            history_head = (history_head + 1) % history_size;
            if (history_count < history_size) {
                history_count += 1;
            }
        }

        @memcpy(theta, theta_next);
        @memcpy(gradient, gradient_next);
        loss = candidate_loss;

        if (max_step_update < tolerance) {
            logisticStoreThetaToModel(model, theta);
            return iter + 1;
        }
    }

    logisticStoreThetaToModel(model, theta);
    return max_iters;
}

pub fn logistic_model_predict_proba(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    out_positive_ptr: [*]f64,
) u8 {
    const model = common.asLogisticModel(handle) orelse return 0;
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
        out_positive_ptr[i] = common.sigmoid(z);
    }

    return 1;
}

pub fn logistic_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    out_labels_ptr: [*]u8,
) u8 {
    const model = common.asLogisticModel(handle) orelse return 0;
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
        out_labels_ptr[i] = if (common.sigmoid(z) >= 0.5) 1 else 0;
    }

    return 1;
}

pub fn logistic_model_copy_coefficients(handle: usize, out_ptr: [*]f64) u8 {
    const model = common.asLogisticModel(handle) orelse return 0;
    @memcpy(out_ptr[0..model.n_features], model.coefficients);
    return 1;
}

pub fn logistic_model_get_intercept(handle: usize) f64 {
    const model = common.asLogisticModel(handle) orelse return 0.0;
    return model.intercept;
}

pub fn logistic_train_epoch(
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

        const residual = common.sigmoid(z) - y_ptr[i];
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

pub fn logistic_train_epochs(
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
