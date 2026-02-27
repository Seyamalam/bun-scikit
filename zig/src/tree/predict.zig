const std = @import("std");
const common = @import("../common.zig");

const SIMD_LANES: usize = 4;
const MIN_SIMD_TREE_ROWS: usize = 1024;
const RF_PREDICT_CHUNK_SIZE: usize = 128;

const ForestPredictTreeRef = struct {
    nodes: []const common.TreeNode,
    root_index: usize,
};

inline fn predictTreeLabelScalar(
    nodes: []const common.TreeNode,
    root_index: usize,
    row_ptr: [*]const f64,
) u16 {
    var node_index = root_index;
    while (true) {
        const node = nodes[node_index];
        if (node.is_leaf) {
            return node.prediction;
        }

        const feature_index = @as(usize, @intCast(node.feature_index));
        const value = row_ptr[feature_index];
        node_index = if (value <= node.threshold)
            @as(usize, @intCast(node.left_index))
        else
            @as(usize, @intCast(node.right_index));
    }
}

fn predictDecisionTreeBatchSimd(
    nodes: []const common.TreeNode,
    root_index: usize,
    x_ptr: [*]const f64,
    start_row: usize,
    lane_count: usize,
    feature_count: usize,
    out_labels_ptr: [*]u16,
) void {
    var node_indices: [SIMD_LANES]usize = [_]usize{root_index} ** SIMD_LANES;
    var done: [SIMD_LANES]bool = [_]bool{false} ** SIMD_LANES;
    var values_arr: [SIMD_LANES]f64 = [_]f64{0.0} ** SIMD_LANES;
    var thresholds_arr: [SIMD_LANES]f64 = [_]f64{0.0} ** SIMD_LANES;
    var left_indices: [SIMD_LANES]usize = [_]usize{root_index} ** SIMD_LANES;
    var right_indices: [SIMD_LANES]usize = [_]usize{root_index} ** SIMD_LANES;
    var has_branch_lane: [SIMD_LANES]bool = [_]bool{false} ** SIMD_LANES;

    while (true) {
        var any_active = false;
        for (0..lane_count) |lane| {
            if (done[lane]) {
                continue;
            }
            any_active = true;
            const node = nodes[node_indices[lane]];
            if (node.is_leaf) {
                out_labels_ptr[start_row + lane] = node.prediction;
                done[lane] = true;
                has_branch_lane[lane] = false;
                continue;
            }

            const row_ptr = x_ptr + (start_row + lane) * feature_count;
            const feature_index = @as(usize, @intCast(node.feature_index));
            values_arr[lane] = row_ptr[feature_index];
            thresholds_arr[lane] = node.threshold;
            left_indices[lane] = @as(usize, @intCast(node.left_index));
            right_indices[lane] = @as(usize, @intCast(node.right_index));
            has_branch_lane[lane] = true;
        }

        if (!any_active) {
            break;
        }

        const values_vec: @Vector(SIMD_LANES, f64) = values_arr;
        const threshold_vec: @Vector(SIMD_LANES, f64) = thresholds_arr;
        const cmp_vec = values_vec <= threshold_vec;

        for (0..lane_count) |lane| {
            if (done[lane] or !has_branch_lane[lane]) {
                continue;
            }
            node_indices[lane] = if (cmp_vec[lane]) left_indices[lane] else right_indices[lane];
        }
    }
}

pub fn decision_tree_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    out_labels_ptr: [*]u16,
) u8 {
    const model = common.asDecisionTreeModel(handle) orelse return 0;
    if (!model.has_root or n_samples == 0 or n_features != model.n_features) {
        return 0;
    }

    const nodes = model.nodes.items;
    const feature_count = model.n_features;
    var i: usize = 0;
    if (n_samples >= MIN_SIMD_TREE_ROWS) {
        while (i + SIMD_LANES <= n_samples) : (i += SIMD_LANES) {
            predictDecisionTreeBatchSimd(
                nodes,
                model.root_index,
                x_ptr,
                i,
                SIMD_LANES,
                feature_count,
                out_labels_ptr,
            );
        }
    }
    while (i < n_samples) : (i += 1) {
        const row_ptr = x_ptr + i * feature_count;
        out_labels_ptr[i] = predictTreeLabelScalar(nodes, model.root_index, row_ptr);
    }
    return 1;
}

const ForestPredictShared = struct {
    tree_refs: []const ForestPredictTreeRef,
    tree_count: usize,
    class_count: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    feature_count: usize,
    out_labels_ptr: [*]u16,
    next_row: std.atomic.Value(usize),
    status: std.atomic.Value(u8),
};

const ForestPredictWorkerContext = struct {
    shared: *ForestPredictShared,
};

fn predictForestRange(shared: *ForestPredictShared, start_row: usize, end_row: usize) bool {
    var row = start_row;
    while (row < end_row) : (row += 1) {
        var class_votes: [common.MAX_CLASS_COUNT]u16 = [_]u16{0} ** common.MAX_CLASS_COUNT;
        const row_ptr = shared.x_ptr + row * shared.feature_count;

        var tree_index: usize = 0;
        while (tree_index < shared.tree_count) : (tree_index += 1) {
            const tree_ref = shared.tree_refs[tree_index];
            const label = @as(usize, predictTreeLabelScalar(tree_ref.nodes, tree_ref.root_index, row_ptr));
            if (label >= shared.class_count) {
                return false;
            }
            class_votes[label] += 1;
        }

        var best_class: usize = 0;
        var best_votes: u16 = class_votes[0];
        var class_index: usize = 1;
        while (class_index < shared.class_count) : (class_index += 1) {
            if (class_votes[class_index] > best_votes) {
                best_votes = class_votes[class_index];
                best_class = class_index;
            }
        }
        shared.out_labels_ptr[row] = @as(u16, @intCast(best_class));
    }
    return true;
}

fn forestPredictWorker(context: *ForestPredictWorkerContext) void {
    while (context.shared.status.load(.seq_cst) == 1) {
        const start_row = context.shared.next_row.fetchAdd(RF_PREDICT_CHUNK_SIZE, .seq_cst);
        if (start_row >= context.shared.n_samples) {
            break;
        }
        const end_row = @min(start_row + RF_PREDICT_CHUNK_SIZE, context.shared.n_samples);
        if (!predictForestRange(context.shared, start_row, end_row)) {
            context.shared.status.store(0, .seq_cst);
            break;
        }
    }
}

fn resolveForestPredictWorkerCount(active_tree_count: usize, n_samples: usize) usize {
    if (active_tree_count < 24 or n_samples < 512) {
        return 1;
    }
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const per_sample_cap = n_samples / RF_PREDICT_CHUNK_SIZE + 1;
    const worker_cap = @min(@min(@as(usize, @intCast(cpu_count)), per_sample_cap), 4);
    return if (worker_cap < 1) 1 else worker_cap;
}

pub fn random_forest_classifier_model_predict(
    handle: usize,
    x_ptr: [*]const f64,
    n_samples: usize,
    n_features: usize,
    out_labels_ptr: [*]u16,
) u8 {
    const model = common.asRandomForestClassifierModel(handle) orelse return 0;
    if (model.active_tree_count == 0 or n_samples == 0 or n_features != model.n_features) {
        return 0;
    }

    const active_trees = model.active_tree_handles[0..model.active_tree_count];
    const active_tree_count = active_trees.len;
    if (active_tree_count == 0) {
        return 0;
    }

    var tree_refs = common.allocator.alloc(ForestPredictTreeRef, active_tree_count) catch return 0;
    defer common.allocator.free(tree_refs);
    for (active_trees, 0..) |tree, idx| {
        tree_refs[idx] = .{
            .nodes = tree.nodes.items,
            .root_index = tree.root_index,
        };
    }

    var shared = ForestPredictShared{
        .tree_refs = tree_refs,
        .tree_count = active_tree_count,
        .class_count = model.class_count,
        .x_ptr = x_ptr,
        .n_samples = n_samples,
        .feature_count = model.n_features,
        .out_labels_ptr = out_labels_ptr,
        .next_row = std.atomic.Value(usize).init(0),
        .status = std.atomic.Value(u8).init(1),
    };

    const worker_count = resolveForestPredictWorkerCount(shared.tree_count, n_samples);
    const contexts = common.allocator.alloc(ForestPredictWorkerContext, worker_count) catch return 0;
    defer common.allocator.free(contexts);
    for (contexts) |*context| {
        context.* = .{ .shared = &shared };
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
            threads[spawned_count] = std.Thread.spawn(.{}, forestPredictWorker, .{&contexts[worker_index]}) catch {
                shared.status.store(0, .seq_cst);
                break;
            };
            spawned_count += 1;
        }
    }

    forestPredictWorker(&contexts[0]);

    var join_index: usize = 0;
    while (join_index < spawned_count) : (join_index += 1) {
        threads[join_index].join();
    }

    return if (shared.status.load(.seq_cst) == 1) 1 else 0;
}
