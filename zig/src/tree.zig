const std = @import("std");
const common = @import("./common.zig");

fn resolveMaxFeatures(model: *const common.DecisionTreeModel) usize {
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

fn selectCandidateFeatures(model: *common.DecisionTreeModel, rng: *common.Mulberry32) []const usize {
    for (model.feature_scratch, 0..) |*entry, idx| {
        entry.* = idx;
    }

    const k = resolveMaxFeatures(model);
    if (k >= model.n_features) {
        return model.feature_scratch[0..model.n_features];
    }

    var i: usize = 0;
    while (i < k) : (i += 1) {
        const remaining = model.n_features - i;
        const j = i + rng.nextIndex(remaining);
        const tmp = model.feature_scratch[i];
        model.feature_scratch[i] = model.feature_scratch[j];
        model.feature_scratch[j] = tmp;
    }

    return model.feature_scratch[0..k];
}

fn findBestSplitForFeature(
    model: *const common.DecisionTreeModel,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    indices: []const usize,
    feature_index: usize,
) ?common.SplitEvaluation {
    const sample_count = indices.len;
    if (sample_count < 2) {
        return null;
    }
    var min_value = std.math.inf(f64);
    var max_value = -std.math.inf(f64);
    var total_positive: usize = 0;
    for (indices) |sample_index| {
        const value = x_ptr[sample_index * model.n_features + feature_index];
        if (value < min_value) {
            min_value = value;
        }
        if (value > max_value) {
            max_value = value;
        }
        total_positive += y_ptr[sample_index];
    }

    if (!std.math.isFinite(min_value) or !std.math.isFinite(max_value) or min_value == max_value) {
        return null;
    }

    const dynamic_bins = @as(usize, @intFromFloat(@floor(@sqrt(@as(f64, @floatFromInt(sample_count))))));
    const bin_count = std.math.clamp(dynamic_bins, 16, common.MAX_THRESHOLD_BINS);
    var bin_totals: [common.MAX_THRESHOLD_BINS]usize = [_]usize{0} ** common.MAX_THRESHOLD_BINS;
    var bin_positives: [common.MAX_THRESHOLD_BINS]usize = [_]usize{0} ** common.MAX_THRESHOLD_BINS;
    const value_range = max_value - min_value;

    for (indices) |sample_index| {
        const value = x_ptr[sample_index * model.n_features + feature_index];
        var bin_index = @as(usize, @intFromFloat(@floor(((value - min_value) / value_range) * @as(f64, @floatFromInt(bin_count)))));
        if (bin_index >= bin_count) {
            bin_index = bin_count - 1;
        }
        bin_totals[bin_index] += 1;
        bin_positives[bin_index] += y_ptr[sample_index];
    }

    var left_count: usize = 0;
    var left_positive: usize = 0;
    var best_impurity = std.math.inf(f64);
    var best_threshold: f64 = 0.0;
    var found = false;

    var bin: usize = 0;
    while (bin + 1 < bin_count) : (bin += 1) {
        left_count += bin_totals[bin];
        left_positive += bin_positives[bin];
        const right_count = sample_count - left_count;
        if (left_count < model.min_samples_leaf or right_count < model.min_samples_leaf) {
            continue;
        }

        const right_positive = total_positive - left_positive;
        const impurity =
            (@as(f64, @floatFromInt(left_count)) / @as(f64, @floatFromInt(sample_count))) *
                common.giniImpurity(left_positive, left_count) +
            (@as(f64, @floatFromInt(right_count)) / @as(f64, @floatFromInt(sample_count))) *
                common.giniImpurity(right_positive, right_count);

        if (impurity < best_impurity) {
            best_impurity = impurity;
            best_threshold = min_value + (value_range * @as(f64, @floatFromInt(bin + 1))) / @as(f64, @floatFromInt(bin_count));
            found = true;
        }
    }

    if (!found) {
        return null;
    }

    return common.SplitEvaluation{
        .threshold = best_threshold,
        .impurity = best_impurity,
    };
}

fn partitionIndicesForThreshold(
    model: *const common.DecisionTreeModel,
    workspace: std.mem.Allocator,
    x_ptr: [*]const f64,
    indices: []const usize,
    feature_index: usize,
    threshold: f64,
) !?common.SplitPartition {
    var left_count: usize = 0;
    for (indices) |sample_index| {
        const value = x_ptr[sample_index * model.n_features + feature_index];
        if (value <= threshold) {
            left_count += 1;
        }
    }

    const right_count = indices.len - left_count;
    if (left_count < model.min_samples_leaf or right_count < model.min_samples_leaf) {
        return null;
    }

    const left_indices = try workspace.alloc(usize, left_count);
    const right_indices = try workspace.alloc(usize, right_count);

    var left_write: usize = 0;
    var right_write: usize = 0;
    for (indices) |sample_index| {
        const value = x_ptr[sample_index * model.n_features + feature_index];
        if (value <= threshold) {
            left_indices[left_write] = sample_index;
            left_write += 1;
        } else {
            right_indices[right_write] = sample_index;
            right_write += 1;
        }
    }

    return common.SplitPartition{
        .left_indices = left_indices,
        .right_indices = right_indices,
    };
}

fn buildDecisionTreeNode(
    model: *common.DecisionTreeModel,
    workspace: std.mem.Allocator,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    indices: []const usize,
    depth: usize,
    rng: *common.Mulberry32,
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
        try model.nodes.append(common.allocator, common.TreeNode{
            .prediction = prediction,
            .feature_index = 0,
            .threshold = 0.0,
            .left_index = 0,
            .right_index = 0,
            .is_leaf = true,
        });
        return node_index;
    }

    const parent_impurity = common.giniImpurity(positive_count, sample_count);
    const candidate_features = selectCandidateFeatures(model, rng);

    var best_feature: usize = 0;
    var best_split: ?common.SplitEvaluation = null;
    var best_found = false;

    for (candidate_features) |feature_index| {
        const split_opt = findBestSplitForFeature(model, x_ptr, y_ptr, indices, feature_index);
        if (split_opt) |split| {
            if (!best_found or split.impurity < best_split.?.impurity) {
                best_split = split;
                best_feature = feature_index;
                best_found = true;
            }
        }
    }

    if (!best_found) {
        const node_index = model.nodes.items.len;
        try model.nodes.append(common.allocator, common.TreeNode{
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
    if (split.impurity >= parent_impurity - 1e-12) {
        const node_index = model.nodes.items.len;
        try model.nodes.append(common.allocator, common.TreeNode{
            .prediction = prediction,
            .feature_index = 0,
            .threshold = 0.0,
            .left_index = 0,
            .right_index = 0,
            .is_leaf = true,
        });
        return node_index;
    }

    const partition = (try partitionIndicesForThreshold(
        model,
        workspace,
        x_ptr,
        indices,
        best_feature,
        split.threshold,
    )) orelse {
        const node_index = model.nodes.items.len;
        try model.nodes.append(common.allocator, common.TreeNode{
            .prediction = prediction,
            .feature_index = 0,
            .threshold = 0.0,
            .left_index = 0,
            .right_index = 0,
            .is_leaf = true,
        });
        return node_index;
    };
    const node_index = model.nodes.items.len;
    try model.nodes.append(common.allocator, common.TreeNode{
        .prediction = prediction,
        .feature_index = 0,
        .threshold = 0.0,
        .left_index = 0,
        .right_index = 0,
        .is_leaf = false,
    });

    const left_index = try buildDecisionTreeNode(
        model,
        workspace,
        x_ptr,
        y_ptr,
        partition.left_indices,
        depth + 1,
        rng,
    );
    const right_index = try buildDecisionTreeNode(
        model,
        workspace,
        x_ptr,
        y_ptr,
        partition.right_indices,
        depth + 1,
        rng,
    );

    model.nodes.items[node_index] = common.TreeNode{
        .prediction = prediction,
        .feature_index = best_feature,
        .threshold = split.threshold,
        .left_index = left_index,
        .right_index = right_index,
        .is_leaf = false,
    };

    return node_index;
}

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
    if (n_features == 0 or max_depth == 0) {
        return 0;
    }

    const model = common.allocator.create(common.DecisionTreeModel) catch return 0;
    errdefer common.allocator.destroy(model);
    const feature_scratch = common.allocator.alloc(usize, n_features) catch return 0;
    errdefer common.allocator.free(feature_scratch);
    for (feature_scratch, 0..) |*entry, idx| {
        entry.* = idx;
    }
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
        .feature_scratch = feature_scratch,
        .nodes = .empty,
    };
    return @intFromPtr(model);
}

pub fn decision_tree_model_destroy(handle: usize) void {
    const model = common.asDecisionTreeModel(handle) orelse return;
    common.allocator.free(model.feature_scratch);
    model.nodes.deinit(common.allocator);
    common.allocator.destroy(model);
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
    const model = common.asDecisionTreeModel(handle) orelse return 0;
    if (n_samples == 0 or n_features == 0 or n_features != model.n_features) {
        return 0;
    }

    model.nodes.clearRetainingCapacity();
    model.has_root = false;

    const root_size = if (sample_count == 0) n_samples else sample_count;
    if (root_size == 0) {
        return 0;
    }

    var arena = std.heap.ArenaAllocator.init(common.allocator);
    defer arena.deinit();
    const workspace = arena.allocator();

    const root_indices = workspace.alloc(usize, root_size) catch return 0;

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
    var rng = common.Mulberry32.init(rng_seed);
    const root_index = buildDecisionTreeNode(model, workspace, x_ptr, y_ptr, root_indices, 0, &rng) catch {
        model.nodes.clearRetainingCapacity();
        model.has_root = false;
        return 0;
    };
    model.root_index = root_index;
    model.has_root = true;
    return 1;
}

pub fn decision_tree_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    out_labels_ptr: [*]u8,
) u8 {
    const model = common.asDecisionTreeModel(handle) orelse return 0;
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

fn resetRandomForestClassifierModel(model: *common.RandomForestClassifierModel) void {
    var i: usize = 0;
    while (i < model.fitted_estimators) : (i += 1) {
        const tree_handle = model.tree_handles[i];
        if (tree_handle != 0) {
            decision_tree_model_destroy(tree_handle);
            model.tree_handles[i] = 0;
        }
    }
    model.fitted_estimators = 0;
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
    if (n_features == 0 or max_depth == 0 or n_estimators == 0) {
        return 0;
    }

    const model = common.allocator.create(common.RandomForestClassifierModel) catch return 0;
    errdefer common.allocator.destroy(model);
    const tree_handles = common.allocator.alloc(usize, n_estimators) catch return 0;
    errdefer common.allocator.free(tree_handles);
    @memset(tree_handles, 0);

    model.* = .{
        .n_features = n_features,
        .n_estimators = n_estimators,
        .max_depth = max_depth,
        .min_samples_split = if (min_samples_split < 2) 2 else min_samples_split,
        .min_samples_leaf = if (min_samples_leaf < 1) 1 else min_samples_leaf,
        .max_features_mode = max_features_mode,
        .max_features_value = max_features_value,
        .bootstrap = bootstrap != 0,
        .random_state = random_state,
        .use_random_state = use_random_state != 0,
        .tree_handles = tree_handles,
        .fitted_estimators = 0,
    };
    return @intFromPtr(model);
}

pub fn random_forest_classifier_model_destroy(handle: usize) void {
    const model = common.asRandomForestClassifierModel(handle) orelse return;
    resetRandomForestClassifierModel(model);
    common.allocator.free(model.tree_handles);
    common.allocator.destroy(model);
}

pub fn random_forest_classifier_model_fit(
    handle: usize,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    n_samples: usize,
    n_features: usize,
) u8 {
    const model = common.asRandomForestClassifierModel(handle) orelse return 0;
    if (n_samples == 0 or n_features == 0 or n_features != model.n_features) {
        return 0;
    }

    resetRandomForestClassifierModel(model);

    const sample_indices = common.allocator.alloc(u32, n_samples) catch return 0;
    defer common.allocator.free(sample_indices);

    const rng_seed: u32 = if (model.use_random_state)
        model.random_state
    else
        @as(u32, @truncate(@as(u64, @bitCast(std.time.microTimestamp()))));
    var rng = common.Mulberry32.init(rng_seed);

    var estimator_index: usize = 0;
    while (estimator_index < model.n_estimators) : (estimator_index += 1) {
        const tree_seed: u32 = if (model.use_random_state)
            model.random_state +% @as(u32, @truncate(estimator_index + 1))
        else
            rng.state +% @as(u32, @truncate(estimator_index + 1));
        const tree_handle = decision_tree_model_create(
            model.max_depth,
            model.min_samples_split,
            model.min_samples_leaf,
            model.max_features_mode,
            model.max_features_value,
            tree_seed,
            if (model.use_random_state) 1 else 0,
            model.n_features,
        );
        if (tree_handle == 0) {
            resetRandomForestClassifierModel(model);
            return 0;
        }

        if (model.bootstrap) {
            var i: usize = 0;
            while (i < n_samples) : (i += 1) {
                sample_indices[i] = @as(u32, @truncate(rng.nextIndex(n_samples)));
            }
        } else {
            for (sample_indices, 0..) |*entry, idx| {
                entry.* = @as(u32, @truncate(idx));
            }
        }

        const fit_status = decision_tree_model_fit(
            tree_handle,
            x_ptr,
            y_ptr,
            n_samples,
            n_features,
            sample_indices.ptr,
            n_samples,
        );
        if (fit_status != 1) {
            decision_tree_model_destroy(tree_handle);
            resetRandomForestClassifierModel(model);
            return 0;
        }

        model.tree_handles[estimator_index] = tree_handle;
        model.fitted_estimators = estimator_index + 1;
    }

    return 1;
}

pub fn random_forest_classifier_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    out_labels_ptr: [*]u8,
) u8 {
    const model = common.asRandomForestClassifierModel(handle) orelse return 0;
    if (model.fitted_estimators == 0 or n_samples == 0 or n_features != model.n_features) {
        return 0;
    }

    const tree_ptrs = common.allocator.alloc(*const common.DecisionTreeModel, model.fitted_estimators) catch return 0;
    defer common.allocator.free(tree_ptrs);
    var active_tree_count: usize = 0;
    var preload_index: usize = 0;
    while (preload_index < model.fitted_estimators) : (preload_index += 1) {
        const tree = common.asDecisionTreeModel(model.tree_handles[preload_index]) orelse continue;
        if (!tree.has_root) {
            continue;
        }
        tree_ptrs[active_tree_count] = tree;
        active_tree_count += 1;
    }
    if (active_tree_count == 0) {
        return 0;
    }

    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const row_offset = i * model.n_features;
        var positive_votes: usize = 0;
        var tree_index: usize = 0;
        while (tree_index < active_tree_count) : (tree_index += 1) {
            const tree = tree_ptrs[tree_index];
            var node_index = tree.root_index;
            while (true) {
                const node = tree.nodes.items[node_index];
                if (node.is_leaf) {
                    positive_votes += if (node.prediction == 1) 1 else 0;
                    break;
                }
                const value = x_ptr[row_offset + node.feature_index];
                node_index = if (value <= node.threshold) node.left_index else node.right_index;
            }
        }
        out_labels_ptr[i] = if (positive_votes * 2 >= active_tree_count) 1 else 0;
    }

    return 1;
}
