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

const TreeNode = struct {
    prediction: u8,
    feature_index: usize,
    threshold: f64,
    left_index: usize,
    right_index: usize,
    is_leaf: bool,
};

const DecisionTreeModel = struct {
    n_features: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features_mode: u8,
    max_features_value: usize,
    random_state: u32,
    use_random_state: bool,
    root_index: usize,
    has_root: bool,
    nodes: std.ArrayListUnmanaged(TreeNode),
};

const SplitResult = struct {
    threshold: f64,
    impurity: f64,
    left_indices: []usize,
    right_indices: []usize,
};

const Mulberry32 = struct {
    state: u32,

    fn init(seed: u32) Mulberry32 {
        return .{ .state = seed };
    }

    fn next(self: *Mulberry32) f64 {
        self.state +%= 0x6d2b79f5;
        var t = self.state ^ (self.state >> 15);
        t = @as(u32, @truncate(@as(u64, t) *% @as(u64, (1 | self.state))));
        t ^= t +% @as(u32, @truncate(@as(u64, (t ^ (t >> 7))) *% @as(u64, (61 | t))));
        return @as(f64, @floatFromInt(t ^ (t >> 14))) / 4294967296.0;
    }

    fn nextIndex(self: *Mulberry32, limit: usize) usize {
        if (limit <= 1) {
            return 0;
        }
        const value = self.next();
        const idx = @as(usize, @intFromFloat(@floor(value * @as(f64, @floatFromInt(limit)))));
        return if (idx >= limit) limit - 1 else idx;
    }
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

inline fn asDecisionTreeModel(handle: usize) ?*DecisionTreeModel {
    if (handle == 0) {
        return null;
    }
    return @as(*DecisionTreeModel, @ptrFromInt(handle));
}

inline fn giniImpurity(positive_count: usize, sample_count: usize) f64 {
    if (sample_count == 0) {
        return 0.0;
    }
    const p1 = @as(f64, @floatFromInt(positive_count)) / @as(f64, @floatFromInt(sample_count));
    const p0 = 1.0 - p1;
    return 1.0 - p1 * p1 - p0 * p0;
}

fn resolveMaxFeatures(model: *const DecisionTreeModel) usize {
    switch (model.max_features_mode) {
        0 => return model.n_features,
        1 => {
            const k = @as(usize, @intFromFloat(@floor(@sqrt(@as(f64, @floatFromInt(model.n_features))))));
            return if (k < 1) 1 else k;
        },
        2 => {
            const k = @as(usize, @intFromFloat(@floor(std.math.log2(@as(f64, @floatFromInt(model.n_features))))));
            return if (k < 1) 1 else k;
        },
        3 => return std.math.clamp(model.max_features_value, 1, model.n_features),
        else => return model.n_features,
    }
}

fn freeSplit(split: SplitResult) void {
    allocator.free(split.left_indices);
    allocator.free(split.right_indices);
}

fn selectCandidateFeatures(model: *const DecisionTreeModel, rng: *Mulberry32) ![]usize {
    const k = resolveMaxFeatures(model);
    if (k >= model.n_features) {
        const all_features = try allocator.alloc(usize, model.n_features);
        errdefer allocator.free(all_features);
        for (all_features, 0..) |*entry, idx| {
            entry.* = idx;
        }
        return all_features;
    }

    const shuffled = try allocator.alloc(usize, model.n_features);
    errdefer allocator.free(shuffled);
    for (shuffled, 0..) |*entry, idx| {
        entry.* = idx;
    }

    var i = model.n_features;
    while (i > 1) {
        i -= 1;
        const j = rng.nextIndex(i + 1);
        const tmp = shuffled[i];
        shuffled[i] = shuffled[j];
        shuffled[j] = tmp;
    }

    const selected = try allocator.alloc(usize, k);
    @memcpy(selected, shuffled[0..k]);
    allocator.free(shuffled);
    return selected;
}

fn findBestSplitForFeature(
    model: *const DecisionTreeModel,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    indices: []const usize,
    feature_index: usize,
) !?SplitResult {
    const sample_count = indices.len;
    if (sample_count < 2) {
        return null;
    }

    const sorted_indices = try allocator.alloc(usize, sample_count);
    defer allocator.free(sorted_indices);
    @memcpy(sorted_indices, indices);

    const SortContext = struct {
        x_ptr: [*]const f64,
        n_features: usize,
        feature_index: usize,
        fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.x_ptr[a * ctx.n_features + ctx.feature_index] <
                ctx.x_ptr[b * ctx.n_features + ctx.feature_index];
        }
    };
    std.sort.heap(usize, sorted_indices, SortContext{
        .x_ptr = x_ptr,
        .n_features = model.n_features,
        .feature_index = feature_index,
    }, SortContext.lessThan);

    var total_positive: usize = 0;
    for (sorted_indices) |sample_index| {
        total_positive += y_ptr[sample_index];
    }

    var left_count: usize = 0;
    var left_positive: usize = 0;
    var best_impurity = std.math.inf(f64);
    var best_threshold: f64 = 0.0;
    var best_split_index: usize = 0;
    var found = false;

    var i: usize = 1;
    while (i < sample_count) : (i += 1) {
        const previous_index = sorted_indices[i - 1];
        left_count += 1;
        left_positive += y_ptr[previous_index];
        const right_count = sample_count - left_count;

        if (left_count < model.min_samples_leaf or right_count < model.min_samples_leaf) {
            continue;
        }

        const left_value = x_ptr[previous_index * model.n_features + feature_index];
        const right_value = x_ptr[sorted_indices[i] * model.n_features + feature_index];
        if (left_value == right_value) {
            continue;
        }

        const right_positive = total_positive - left_positive;
        const impurity =
            (@as(f64, @floatFromInt(left_count)) / @as(f64, @floatFromInt(sample_count))) *
                giniImpurity(left_positive, left_count) +
            (@as(f64, @floatFromInt(right_count)) / @as(f64, @floatFromInt(sample_count))) *
                giniImpurity(right_positive, right_count);

        if (impurity < best_impurity) {
            best_impurity = impurity;
            best_threshold = (left_value + right_value) / 2.0;
            best_split_index = i;
            found = true;
        }
    }

    if (!found) {
        return null;
    }

    const left_indices = try allocator.alloc(usize, best_split_index);
    errdefer allocator.free(left_indices);
    const right_size = sample_count - best_split_index;
    const right_indices = try allocator.alloc(usize, right_size);
    errdefer allocator.free(right_indices);

    @memcpy(left_indices, sorted_indices[0..best_split_index]);
    @memcpy(right_indices, sorted_indices[best_split_index..]);

    return SplitResult{
        .threshold = best_threshold,
        .impurity = best_impurity,
        .left_indices = left_indices,
        .right_indices = right_indices,
    };
}

fn buildDecisionTreeNode(
    model: *DecisionTreeModel,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    indices: []const usize,
    depth: usize,
    rng: *Mulberry32,
) !usize {
    const sample_count = indices.len;
    var positive_count: usize = 0;
    for (indices) |sample_index| {
        positive_count += y_ptr[sample_index];
    }
    const prediction: u8 = if (positive_count * 2 >= sample_count) 1 else 0;

    const same_class = positive_count == 0 or positive_count == sample_count;
    const depth_stop = depth >= model.max_depth;
    const split_stop = sample_count < model.min_samples_split;
    if (same_class or depth_stop or split_stop) {
        const node_index = model.nodes.items.len;
        try model.nodes.append(allocator, TreeNode{
            .prediction = prediction,
            .feature_index = 0,
            .threshold = 0.0,
            .left_index = 0,
            .right_index = 0,
            .is_leaf = true,
        });
        return node_index;
    }

    const parent_impurity = giniImpurity(positive_count, sample_count);
    const candidate_features = try selectCandidateFeatures(model, rng);
    defer allocator.free(candidate_features);

    var best_feature: usize = 0;
    var best_split: ?SplitResult = null;
    var best_found = false;

    for (candidate_features) |feature_index| {
        const split_opt = try findBestSplitForFeature(model, x_ptr, y_ptr, indices, feature_index);
        if (split_opt) |split| {
            if (!best_found or split.impurity < best_split.?.impurity) {
                if (best_split) |previous| {
                    freeSplit(previous);
                }
                best_split = split;
                best_feature = feature_index;
                best_found = true;
            } else {
                freeSplit(split);
            }
        }
    }

    if (!best_found) {
        const node_index = model.nodes.items.len;
        try model.nodes.append(allocator, TreeNode{
            .prediction = prediction,
            .feature_index = 0,
            .threshold = 0.0,
            .left_index = 0,
            .right_index = 0,
            .is_leaf = true,
        });
        return node_index;
    }

    const split = best_split.?;
    defer freeSplit(split);
    if (split.impurity >= parent_impurity - 1e-12) {
        const node_index = model.nodes.items.len;
        try model.nodes.append(allocator, TreeNode{
            .prediction = prediction,
            .feature_index = 0,
            .threshold = 0.0,
            .left_index = 0,
            .right_index = 0,
            .is_leaf = true,
        });
        return node_index;
    }

    const node_index = model.nodes.items.len;
    try model.nodes.append(allocator, TreeNode{
        .prediction = prediction,
        .feature_index = 0,
        .threshold = 0.0,
        .left_index = 0,
        .right_index = 0,
        .is_leaf = false,
    });

    const left_index = try buildDecisionTreeNode(
        model,
        x_ptr,
        y_ptr,
        split.left_indices,
        depth + 1,
        rng,
    );
    const right_index = try buildDecisionTreeNode(
        model,
        x_ptr,
        y_ptr,
        split.right_indices,
        depth + 1,
        rng,
    );

    model.nodes.items[node_index] = TreeNode{
        .prediction = prediction,
        .feature_index = best_feature,
        .threshold = split.threshold,
        .left_index = left_index,
        .right_index = right_index,
        .is_leaf = false,
    };

    return node_index;
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

fn logisticThetaLength(model: *const LogisticModel) usize {
    return model.n_features + @as(usize, if (model.fit_intercept) 1 else 0);
}

fn logisticLoadThetaFromModel(model: *const LogisticModel, theta: []f64) void {
    if (model.fit_intercept) {
        theta[0] = model.intercept;
        @memcpy(theta[1 .. 1 + model.n_features], model.coefficients);
    } else {
        @memcpy(theta[0..model.n_features], model.coefficients);
    }
}

fn logisticStoreThetaToModel(model: *LogisticModel, theta: []const f64) void {
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
    model: *const LogisticModel,
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

        const prediction = sigmoid(z);
        const residual = prediction - y_value;
        if (model.fit_intercept) {
            gradient[0] += residual;
        }
        j = 0;
        while (j < model.n_features) : (j += 1) {
            gradient[theta_offset + j] += residual * x_ptr[row_offset + j];
        }

        // Stable binary cross-entropy evaluation.
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
    const model = asLogisticModel(handle) orelse return 0;
    if (n_samples == 0 or model.n_features == 0 or max_iters == 0) {
        return 0;
    }

    const theta_len = logisticThetaLength(model);
    const history_size = std.math.clamp(memory, 3, 20);

    const theta = allocator.alloc(f64, theta_len) catch return 0;
    defer allocator.free(theta);
    const theta_next = allocator.alloc(f64, theta_len) catch return 0;
    defer allocator.free(theta_next);
    const gradient = allocator.alloc(f64, theta_len) catch return 0;
    defer allocator.free(gradient);
    const gradient_next = allocator.alloc(f64, theta_len) catch return 0;
    defer allocator.free(gradient_next);
    const direction = allocator.alloc(f64, theta_len) catch return 0;
    defer allocator.free(direction);
    const q = allocator.alloc(f64, theta_len) catch return 0;
    defer allocator.free(q);
    const r = allocator.alloc(f64, theta_len) catch return 0;
    defer allocator.free(r);
    const s_history = allocator.alloc(f64, history_size * theta_len) catch return 0;
    defer allocator.free(s_history);
    const y_history = allocator.alloc(f64, history_size * theta_len) catch return 0;
    defer allocator.free(y_history);
    const rho_history = allocator.alloc(f64, history_size) catch return 0;
    defer allocator.free(rho_history);
    const alpha_history = allocator.alloc(f64, history_size) catch return 0;
    defer allocator.free(alpha_history);
    const index_history = allocator.alloc(usize, history_size) catch return 0;
    defer allocator.free(index_history);

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

pub export fn decision_tree_model_create(
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features_mode: u8,
    max_features_value: usize,
    random_state: u32,
    use_random_state: u8,
    n_features: usize,
) usize {
    if (n_features == 0 or max_depth == 0) {
        return 0;
    }

    const model = allocator.create(DecisionTreeModel) catch return 0;
    errdefer allocator.destroy(model);
    model.* = .{
        .n_features = n_features,
        .max_depth = max_depth,
        .min_samples_split = if (min_samples_split < 2) 2 else min_samples_split,
        .min_samples_leaf = if (min_samples_leaf < 1) 1 else min_samples_leaf,
        .max_features_mode = max_features_mode,
        .max_features_value = max_features_value,
        .random_state = random_state,
        .use_random_state = use_random_state != 0,
        .root_index = 0,
        .has_root = false,
        .nodes = .empty,
    };
    return @intFromPtr(model);
}

pub export fn decision_tree_model_destroy(handle: usize) void {
    const model = asDecisionTreeModel(handle) orelse return;
    model.nodes.deinit(allocator);
    allocator.destroy(model);
}

pub export fn decision_tree_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    n_samples: usize,
    n_features: usize,
    sample_indices_ptr: [*]const u32,
    sample_count: usize,
) u8 {
    const model = asDecisionTreeModel(handle) orelse return 0;
    if (n_samples == 0 or n_features == 0 or n_features != model.n_features) {
        return 0;
    }

    model.nodes.clearRetainingCapacity();
    model.has_root = false;

    const root_size = if (sample_count == 0) n_samples else sample_count;
    if (root_size == 0) {
        return 0;
    }

    const root_indices = allocator.alloc(usize, root_size) catch return 0;
    defer allocator.free(root_indices);

    if (sample_count == 0) {
        for (root_indices, 0..) |*entry, idx| {
            entry.* = idx;
        }
    } else {
        for (root_indices, 0..) |*entry, idx| {
            const sample_index = @as(usize, sample_indices_ptr[idx]);
            if (sample_index >= n_samples) {
                return 0;
            }
            entry.* = sample_index;
        }
    }

    const rng_seed: u32 = if (model.use_random_state)
        model.random_state
    else
        @as(u32, @truncate(@as(u64, @bitCast(std.time.microTimestamp()))));
    var rng = Mulberry32.init(rng_seed);
    const root_index = buildDecisionTreeNode(model, x_ptr, y_ptr, root_indices, 0, &rng) catch {
        model.nodes.clearRetainingCapacity();
        model.has_root = false;
        return 0;
    };
    model.root_index = root_index;
    model.has_root = true;
    return 1;
}

pub export fn decision_tree_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    out_labels_ptr: [*]u8,
) u8 {
    const model = asDecisionTreeModel(handle) orelse return 0;
    if (!model.has_root or n_samples == 0 or n_features != model.n_features) {
        return 0;
    }

    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const row_offset = i * model.n_features;
        var node_index = model.root_index;
        while (true) {
            const node = model.nodes.items[node_index];
            if (node.is_leaf) {
                out_labels_ptr[i] = node.prediction;
                break;
            }

            const value = x_ptr[row_offset + node.feature_index];
            node_index = if (value <= node.threshold) node.left_index else node.right_index;
        }
    }

    return 1;
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
