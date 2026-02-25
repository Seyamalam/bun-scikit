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

fn selectCandidateFeatures(
    model: *common.DecisionTreeModel,
    feature_scratch: []usize,
    rng: *common.Mulberry32,
) []const usize {
    for (feature_scratch, 0..) |*entry, idx| {
        entry.* = idx;
    }

    const k = resolveMaxFeatures(model);
    if (k >= model.n_features) {
        return feature_scratch[0..model.n_features];
    }

    var i: usize = 0;
    while (i < k) : (i += 1) {
        const remaining = model.n_features - i;
        const j = i + rng.nextIndex(remaining);
        const tmp = feature_scratch[i];
        feature_scratch[i] = feature_scratch[j];
        feature_scratch[j] = tmp;
    }

    return feature_scratch[0..k];
}

fn toTreeIndex(value: usize) !u32 {
    if (value > std.math.maxInt(u32)) {
        return error.TreeTooLarge;
    }
    return @as(u32, @intCast(value));
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
    const bin_count = std.math.clamp(dynamic_bins, 6, common.MAX_THRESHOLD_BINS);
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
    feature_scratch: []usize,
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
    const candidate_features = selectCandidateFeatures(model, feature_scratch, rng);

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
        feature_scratch,
        workspace,
        x_ptr,
        y_ptr,
        partition.left_indices,
        depth + 1,
        rng,
    );
    const right_index = try buildDecisionTreeNode(
        model,
        feature_scratch,
        workspace,
        x_ptr,
        y_ptr,
        partition.right_indices,
        depth + 1,
        rng,
    );

    model.nodes.items[node_index] = common.TreeNode{
        .prediction = prediction,
        .feature_index = try toTreeIndex(best_feature),
        .threshold = split.threshold,
        .left_index = try toTreeIndex(left_index),
        .right_index = try toTreeIndex(right_index),
        .is_leaf = false,
    };

    return node_index;
}

fn createDecisionTreeModelInternal(
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features_mode: u8,
    max_features_value: usize,
    random_state: u32,
    use_random_state: u8,
    n_features: usize,
) ?*common.DecisionTreeModel {
    if (n_features == 0 or max_depth == 0) {
        return null;
    }

    const model = common.allocator.create(common.DecisionTreeModel) catch return null;
    errdefer common.allocator.destroy(model);
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
    return model;
}

fn destroyDecisionTreeModelInternal(model: *common.DecisionTreeModel) void {
    model.nodes.deinit(common.allocator);
    common.allocator.destroy(model);
}

fn fitDecisionTreeModelWithWorkspace(
    model: *common.DecisionTreeModel,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    n_samples: usize,
    n_features: usize,
    sample_indices_opt: ?[]const usize,
    validate_sample_indices: bool,
    workspace: std.mem.Allocator,
) u8 {
    if (n_samples == 0 or n_features == 0 or n_features != model.n_features) {
        return 0;
    }

    const feature_scratch = workspace.alloc(usize, model.n_features) catch return 0;

    const root_indices = if (sample_indices_opt) |sample_indices| blk: {
        if (sample_indices.len == 0) {
            return 0;
        }
        if (validate_sample_indices) {
            for (sample_indices) |sample_index| {
                if (sample_index >= n_samples) {
                    return 0;
                }
            }
        }
        break :blk sample_indices;
    } else blk: {
        const indices = workspace.alloc(usize, n_samples) catch return 0;
        for (indices, 0..) |*entry, idx| {
            entry.* = idx;
        }
        break :blk @as([]const usize, indices);
    };

    const rng_seed: u32 = if (model.use_random_state)
        model.random_state
    else
        @as(u32, @truncate(@as(u64, @bitCast(std.time.microTimestamp()))));
    var rng = common.Mulberry32.init(rng_seed);
    const root_index = buildDecisionTreeNode(model, feature_scratch, workspace, x_ptr, y_ptr, root_indices, 0, &rng) catch {
        model.nodes.clearRetainingCapacity();
        model.has_root = false;
        return 0;
    };
    model.root_index = root_index;
    model.has_root = true;
    return 1;
}

fn fitDecisionTreeModelInternal(
    model: *common.DecisionTreeModel,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    n_samples: usize,
    n_features: usize,
    sample_indices_opt: ?[]const usize,
) u8 {
    if (n_samples == 0 or n_features == 0 or n_features != model.n_features) {
        return 0;
    }

    model.nodes.clearRetainingCapacity();
    model.has_root = false;

    var arena = std.heap.ArenaAllocator.init(common.allocator);
    defer arena.deinit();
    return fitDecisionTreeModelWithWorkspace(
        model,
        x_ptr,
        y_ptr,
        n_samples,
        n_features,
        sample_indices_opt,
        true,
        arena.allocator(),
    );
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
    const model = createDecisionTreeModelInternal(
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features_mode,
        max_features_value,
        random_state,
        use_random_state,
        n_features,
    ) orelse return 0;
    return @intFromPtr(model);
}

pub fn decision_tree_model_destroy(handle: usize) void {
    const model = common.asDecisionTreeModel(handle) orelse return;
    destroyDecisionTreeModelInternal(model);
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
    if (sample_count == 0) {
        return fitDecisionTreeModelInternal(model, x_ptr, y_ptr, n_samples, n_features, null);
    }

    const sample_indices = common.allocator.alloc(usize, sample_count) catch return 0;
    defer common.allocator.free(sample_indices);
    for (sample_indices, 0..) |*entry, idx| {
        entry.* = @as(usize, sample_indices_ptr[idx]);
    }

    return fitDecisionTreeModelInternal(
        model,
        x_ptr,
        y_ptr,
        n_samples,
        n_features,
        sample_indices,
    );
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

    const nodes = model.nodes.items;
    const feature_count = model.n_features;
    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const row_ptr = x_ptr + i * feature_count;
        var node_index = model.root_index;
        while (true) {
            const node = nodes[node_index];
            if (node.is_leaf) {
                out_labels_ptr[i] = node.prediction;
                break;
            }

            const feature_index = @as(usize, @intCast(node.feature_index));
            const value = row_ptr[feature_index];
            node_index = if (value <= node.threshold)
                @as(usize, @intCast(node.left_index))
            else
                @as(usize, @intCast(node.right_index));
        }
    }

    return 1;
}

fn resetRandomForestClassifierModel(model: *common.RandomForestClassifierModel) void {
    var i: usize = 0;
    while (i < model.n_estimators) : (i += 1) {
        const tree_model = model.tree_handles[i] orelse continue;
        tree_model.nodes.clearRetainingCapacity();
        tree_model.has_root = false;
    }
    model.active_tree_count = 0;
    model.fitted_estimators = 0;
}

fn destroyRandomForestClassifierTrees(model: *common.RandomForestClassifierModel) void {
    var i: usize = 0;
    while (i < model.n_estimators) : (i += 1) {
        if (model.tree_handles[i]) |tree_model| {
            destroyDecisionTreeModelInternal(tree_model);
            model.tree_handles[i] = null;
        }
    }
    model.active_tree_count = 0;
    model.fitted_estimators = 0;
}

const ForestFitTask = struct {
    model: *common.RandomForestClassifierModel,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    n_samples: usize,
    n_features: usize,
    start_index: usize,
    end_index: usize,
    base_seed: u32,
    status: *std.atomic.Value(u8),
};

fn fitForestRange(task: *const ForestFitTask) bool {
    const model = task.model;
    const sample_indices = common.allocator.alloc(usize, task.n_samples) catch return false;
    defer common.allocator.free(sample_indices);
    var arena = std.heap.ArenaAllocator.init(common.allocator);
    defer arena.deinit();

    if (!model.bootstrap) {
        for (sample_indices, 0..) |*entry, idx| {
            entry.* = idx;
        }
    }

    var estimator_index = task.start_index;
    while (estimator_index < task.end_index) : (estimator_index += 1) {
        if (task.status.load(.seq_cst) != 1) {
            return false;
        }

        const tree_model = model.tree_handles[estimator_index] orelse return false;
        const tree_seed = task.base_seed +% @as(u32, @truncate(estimator_index + 1));
        tree_model.random_state = tree_seed;
        tree_model.use_random_state = true;

        if (model.bootstrap) {
            var rng = common.Mulberry32.init(tree_seed ^ 0x9e3779b9);
            var i: usize = 0;
            while (i < task.n_samples) : (i += 1) {
                sample_indices[i] = rng.nextIndex(task.n_samples);
            }
        }

        _ = arena.reset(.retain_capacity);
        const fit_status = fitDecisionTreeModelWithWorkspace(
            tree_model,
            task.x_ptr,
            task.y_ptr,
            task.n_samples,
            task.n_features,
            sample_indices,
            false,
            arena.allocator(),
        );
        if (fit_status != 1) {
            return false;
        }
    }

    return true;
}

fn forestFitWorker(task: *ForestFitTask) void {
    if (!fitForestRange(task)) {
        task.status.store(0, .seq_cst);
    }
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
    const tree_handles = common.allocator.alloc(?*common.DecisionTreeModel, n_estimators) catch return 0;
    errdefer common.allocator.free(tree_handles);
    const active_tree_handles = common.allocator.alloc(*common.DecisionTreeModel, n_estimators) catch return 0;
    errdefer common.allocator.free(active_tree_handles);
    for (tree_handles, 0..) |*entry, estimator_index| {
        const tree_model = createDecisionTreeModelInternal(
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features_mode,
            max_features_value,
            random_state +% @as(u32, @truncate(estimator_index + 1)),
            1,
            n_features,
        ) orelse {
            var cleanup_index: usize = 0;
            while (cleanup_index < estimator_index) : (cleanup_index += 1) {
                if (tree_handles[cleanup_index]) |cleanup_tree| {
                    destroyDecisionTreeModelInternal(cleanup_tree);
                    tree_handles[cleanup_index] = null;
                }
            }
            return 0;
        };
        entry.* = tree_model;
    }

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
        .active_tree_handles = active_tree_handles,
        .active_tree_count = 0,
        .fitted_estimators = 0,
    };
    return @intFromPtr(model);
}

pub fn random_forest_classifier_model_destroy(handle: usize) void {
    const model = common.asRandomForestClassifierModel(handle) orelse return;
    destroyRandomForestClassifierTrees(model);
    common.allocator.free(model.active_tree_handles);
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
    const base_seed: u32 = if (model.use_random_state)
        model.random_state
    else
        @as(u32, @truncate(@as(u64, @bitCast(std.time.microTimestamp()))));

    var worker_count: usize = 1;
    if (model.n_estimators >= 256 and n_samples >= 10_000) {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const worker_cap: usize = @min(model.n_estimators, @as(usize, @intCast(cpu_count)));
        worker_count = if (worker_cap < 1) 1 else worker_cap;
    }

    var status = std.atomic.Value(u8).init(1);
    const tasks = common.allocator.alloc(ForestFitTask, worker_count) catch return 0;
    defer common.allocator.free(tasks);

    var threads: []std.Thread = &[_]std.Thread{};
    if (worker_count > 1) {
        threads = common.allocator.alloc(std.Thread, worker_count - 1) catch return 0;
    }
    defer if (worker_count > 1) common.allocator.free(threads);

    const chunk_size = @divFloor(model.n_estimators + worker_count - 1, worker_count);
    var worker_index: usize = 0;
    while (worker_index < worker_count) : (worker_index += 1) {
        const start_index = worker_index * chunk_size;
        const end_index = @min(start_index + chunk_size, model.n_estimators);
        tasks[worker_index] = .{
            .model = model,
            .x_ptr = x_ptr,
            .y_ptr = y_ptr,
            .n_samples = n_samples,
            .n_features = n_features,
            .start_index = start_index,
            .end_index = end_index,
            .base_seed = base_seed,
            .status = &status,
        };
    }

    var spawned_count: usize = 0;
    if (worker_count > 1) {
        worker_index = 1;
        while (worker_index < worker_count) : (worker_index += 1) {
            threads[spawned_count] = std.Thread.spawn(.{}, forestFitWorker, .{&tasks[worker_index]}) catch {
                status.store(0, .seq_cst);
                break;
            };
            spawned_count += 1;
        }
    }

    forestFitWorker(&tasks[0]);

    var join_index: usize = 0;
    while (join_index < spawned_count) : (join_index += 1) {
        threads[join_index].join();
    }

    if (status.load(.seq_cst) != 1) {
        resetRandomForestClassifierModel(model);
        return 0;
    }

    var active_tree_count: usize = 0;
    var estimator_index: usize = 0;
    while (estimator_index < model.n_estimators) : (estimator_index += 1) {
        const tree = model.tree_handles[estimator_index] orelse continue;
        if (!tree.has_root) {
            continue;
        }
        model.active_tree_handles[active_tree_count] = tree;
        active_tree_count += 1;
    }
    if (active_tree_count == 0) {
        resetRandomForestClassifierModel(model);
        return 0;
    }
    model.active_tree_count = active_tree_count;
    model.fitted_estimators = active_tree_count;
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
    if (model.active_tree_count == 0 or n_samples == 0 or n_features != model.n_features) {
        return 0;
    }

    const active_trees = model.active_tree_handles[0..model.active_tree_count];
    const feature_count = model.n_features;
    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        const row_ptr = x_ptr + i * feature_count;
        var positive_votes: usize = 0;
        for (active_trees) |tree| {
            const nodes = tree.nodes.items;
            var node_index = tree.root_index;
            while (true) {
                const node = nodes[node_index];
                if (node.is_leaf) {
                    positive_votes += if (node.prediction == 1) 1 else 0;
                    break;
                }
                const feature_index = @as(usize, @intCast(node.feature_index));
                const value = row_ptr[feature_index];
                node_index = if (value <= node.threshold)
                    @as(usize, @intCast(node.left_index))
                else
                    @as(usize, @intCast(node.right_index));
            }
        }
        out_labels_ptr[i] = if (positive_votes * 2 >= model.active_tree_count) 1 else 0;
    }

    return 1;
}
