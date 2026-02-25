const std = @import("std");
const common = @import("../common.zig");
const split = @import("./split.zig");

fn toTreeIndex(value: usize) !u32 {
    if (value > std.math.maxInt(u32)) {
        return error.TreeTooLarge;
    }
    return @as(u32, @intCast(value));
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
        try model.nodes.append(common.allocator, .{
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
    const candidate_features = split.selectCandidateFeatures(model, feature_scratch, rng);

    var best_feature: usize = 0;
    var best_split: ?common.SplitEvaluation = null;
    var best_found = false;

    for (candidate_features) |feature_index| {
        const split_opt = split.findBestSplitForFeature(model, x_ptr, y_ptr, indices, feature_index);
        if (split_opt) |entry| {
            if (!best_found or entry.impurity < best_split.?.impurity) {
                best_split = entry;
                best_feature = feature_index;
                best_found = true;
            }
        }
    }

    if (!best_found) {
        const node_index = model.nodes.items.len;
        try model.nodes.append(common.allocator, .{
            .prediction = prediction,
            .feature_index = 0,
            .threshold = 0.0,
            .left_index = 0,
            .right_index = 0,
            .is_leaf = true,
        });
        return node_index;
    }

    const best = best_split.?;
    if (best.impurity >= parent_impurity - 1e-12) {
        const node_index = model.nodes.items.len;
        try model.nodes.append(common.allocator, .{
            .prediction = prediction,
            .feature_index = 0,
            .threshold = 0.0,
            .left_index = 0,
            .right_index = 0,
            .is_leaf = true,
        });
        return node_index;
    }

    const partition = (try split.partitionIndicesForThreshold(
        model,
        workspace,
        x_ptr,
        indices,
        best_feature,
        best.threshold,
    )) orelse {
        const node_index = model.nodes.items.len;
        try model.nodes.append(common.allocator, .{
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
    try model.nodes.append(common.allocator, .{
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

    model.nodes.items[node_index] = .{
        .prediction = prediction,
        .feature_index = try toTreeIndex(best_feature),
        .threshold = best.threshold,
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

const ForestFitShared = struct {
    model: *common.RandomForestClassifierModel,
    x_ptr: [*]const f64,
    y_ptr: [*]const u8,
    n_samples: usize,
    n_features: usize,
    base_seed: u32,
    next_estimator: std.atomic.Value(usize),
    status: std.atomic.Value(u8),
};

const ForestFitWorkerContext = struct {
    shared: *ForestFitShared,
    worker_seed: u32,
};

fn fitSingleForestEstimator(
    shared: *ForestFitShared,
    worker_seed: u32,
    estimator_index: usize,
    sample_indices: []usize,
    full_indices_ready: *bool,
    arena: *std.heap.ArenaAllocator,
) bool {
    const model = shared.model;
    const tree_model = model.tree_handles[estimator_index] orelse return false;
    const tree_seed = shared.base_seed +% @as(u32, @truncate(estimator_index + 1));
    tree_model.random_state = tree_seed;
    tree_model.use_random_state = true;

    if (model.bootstrap) {
        var rng = common.Mulberry32.init(tree_seed ^ worker_seed ^ 0x9e3779b9);
        var i: usize = 0;
        while (i < shared.n_samples) : (i += 1) {
            sample_indices[i] = rng.nextIndex(shared.n_samples);
        }
    } else if (!full_indices_ready.*) {
        for (sample_indices, 0..) |*entry, idx| {
            entry.* = idx;
        }
        full_indices_ready.* = true;
    }

    _ = arena.reset(.retain_capacity);
    return fitDecisionTreeModelWithWorkspace(
        tree_model,
        shared.x_ptr,
        shared.y_ptr,
        shared.n_samples,
        shared.n_features,
        sample_indices,
        false,
        arena.allocator(),
    ) == 1;
}

fn forestFitWorker(context: *ForestFitWorkerContext) void {
    var arena = std.heap.ArenaAllocator.init(common.allocator);
    defer arena.deinit();
    const sample_indices = common.allocator.alloc(usize, context.shared.n_samples) catch {
        context.shared.status.store(0, .seq_cst);
        return;
    };
    defer common.allocator.free(sample_indices);
    var full_indices_ready = false;

    while (context.shared.status.load(.seq_cst) == 1) {
        const estimator_index = context.shared.next_estimator.fetchAdd(1, .seq_cst);
        if (estimator_index >= context.shared.model.n_estimators) {
            break;
        }
        if (!fitSingleForestEstimator(
            context.shared,
            context.worker_seed,
            estimator_index,
            sample_indices,
            &full_indices_ready,
            &arena,
        )) {
            context.shared.status.store(0, .seq_cst);
            break;
        }
    }
}

fn resolveForestFitWorkerCount(n_estimators: usize, n_samples: usize) usize {
    if (n_estimators < 16 or n_samples < 2048) {
        return 1;
    }
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const worker_cap = @min(n_estimators, @as(usize, @intCast(cpu_count)));
    return if (worker_cap < 1) 1 else worker_cap;
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

    var shared = ForestFitShared{
        .model = model,
        .x_ptr = x_ptr,
        .y_ptr = y_ptr,
        .n_samples = n_samples,
        .n_features = n_features,
        .base_seed = base_seed,
        .next_estimator = std.atomic.Value(usize).init(0),
        .status = std.atomic.Value(u8).init(1),
    };

    const worker_count = resolveForestFitWorkerCount(model.n_estimators, n_samples);
    const contexts = common.allocator.alloc(ForestFitWorkerContext, worker_count) catch return 0;
    defer common.allocator.free(contexts);
    for (contexts, 0..) |*context, idx| {
        context.* = .{
            .shared = &shared,
            .worker_seed = base_seed +% @as(u32, @truncate((idx + 1) * 0x9e37)),
        };
    }

    var threads: []std.Thread = &[_]std.Thread{};
    if (worker_count > 1) {
        threads = common.allocator.alloc(std.Thread, worker_count - 1) catch return 0;
    }
    defer if (worker_count > 1) common.allocator.free(threads);

    var spawned_count: usize = 0;
    if (worker_count > 1) {
        var worker_index: usize = 1;
        while (worker_index < worker_count) : (worker_index += 1) {
            threads[spawned_count] = std.Thread.spawn(.{}, forestFitWorker, .{&contexts[worker_index]}) catch {
                shared.status.store(0, .seq_cst);
                break;
            };
            spawned_count += 1;
        }
    }

    forestFitWorker(&contexts[0]);

    var join_index: usize = 0;
    while (join_index < spawned_count) : (join_index += 1) {
        threads[join_index].join();
    }

    if (shared.status.load(.seq_cst) != 1) {
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
