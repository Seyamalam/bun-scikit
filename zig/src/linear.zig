const common = @import("./common.zig");

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

pub fn linear_model_create(n_features: usize, fit_intercept: u8) usize {
    if (n_features == 0) {
        return 0;
    }

    const model = common.allocator.create(common.LinearModel) catch return 0;
    errdefer common.allocator.destroy(model);
    const coefficients = common.allocator.alloc(f64, n_features) catch return 0;
    @memset(coefficients, 0.0);

    model.* = .{
        .n_features = n_features,
        .fit_intercept = fit_intercept != 0,
        .coefficients = coefficients,
        .intercept = 0.0,
    };
    return @intFromPtr(model);
}

pub fn linear_model_destroy(handle: usize) void {
    const model = common.asLinearModel(handle) orelse return;
    common.allocator.free(model.coefficients);
    common.allocator.destroy(model);
}

pub fn linear_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const f64,
    n_samples: usize,
    l2: f64,
) u8 {
    const model = common.asLinearModel(handle) orelse return 0;
    if (n_samples == 0 or model.n_features == 0) {
        return 0;
    }

    const has_intercept: usize = if (model.fit_intercept) 1 else 0;
    const dim = model.n_features + has_intercept;
    const offset = has_intercept;

    const base_gram = common.allocator.alloc(f64, dim * dim) catch return 0;
    defer common.allocator.free(base_gram);
    const gram_attempt = common.allocator.alloc(f64, dim * dim) catch return 0;
    defer common.allocator.free(gram_attempt);
    const rhs = common.allocator.alloc(f64, dim) catch return 0;
    defer common.allocator.free(rhs);
    const lower = common.allocator.alloc(f64, dim * dim) catch return 0;
    defer common.allocator.free(lower);
    const forward = common.allocator.alloc(f64, dim) catch return 0;
    defer common.allocator.free(forward);
    const solution = common.allocator.alloc(f64, dim) catch return 0;
    defer common.allocator.free(solution);

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

pub fn linear_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    out_ptr: [*]f64,
) u8 {
    const model = common.asLinearModel(handle) orelse return 0;
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

pub fn linear_model_copy_coefficients(handle: usize, out_ptr: [*]f64) u8 {
    const model = common.asLinearModel(handle) orelse return 0;
    @memcpy(out_ptr[0..model.n_features], model.coefficients);
    return 1;
}

pub fn linear_model_get_intercept(handle: usize) f64 {
    const model = common.asLinearModel(handle) orelse return 0.0;
    return model.intercept;
}
