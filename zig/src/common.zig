const std = @import("std");

pub const allocator = std.heap.page_allocator;

pub const LinearModel = struct {
    n_features: usize,
    fit_intercept: bool,
    coefficients: []f64,
    intercept: f64,
};

pub const LogisticModel = struct {
    n_features: usize,
    fit_intercept: bool,
    coefficients: []f64,
    gradients: []f64,
    intercept: f64,
};

pub const TreeNode = struct {
    prediction: u8,
    feature_index: usize,
    threshold: f64,
    left_index: usize,
    right_index: usize,
    is_leaf: bool,
};

pub const DecisionTreeModel = struct {
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

pub const RandomForestClassifierModel = struct {
    n_features: usize,
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features_mode: u8,
    max_features_value: usize,
    bootstrap: bool,
    random_state: u32,
    use_random_state: bool,
    tree_handles: []?*DecisionTreeModel,
    fitted_estimators: usize,
};

pub const SplitEvaluation = struct {
    threshold: f64,
    impurity: f64,
};

pub const SplitPartition = struct {
    left_indices: []usize,
    right_indices: []usize,
};

pub const MAX_THRESHOLD_BINS: usize = 24;

pub const Mulberry32 = struct {
    state: u32,

    pub fn init(seed: u32) Mulberry32 {
        return .{ .state = seed };
    }

    pub fn nextU32(self: *Mulberry32) u32 {
        self.state +%= 0x6d2b79f5;
        var t = self.state ^ (self.state >> 15);
        t = @as(u32, @truncate(@as(u64, t) *% @as(u64, (1 | self.state))));
        t ^= t +% @as(u32, @truncate(@as(u64, (t ^ (t >> 7))) *% @as(u64, (61 | t))));
        return t ^ (t >> 14);
    }

    pub fn next(self: *Mulberry32) f64 {
        return @as(f64, @floatFromInt(self.nextU32())) / 4294967296.0;
    }

    pub fn nextIndex(self: *Mulberry32, limit: usize) usize {
        if (limit <= 1) {
            return 0;
        }
        return @as(usize, @intCast(self.nextU32())) % limit;
    }
};

pub inline fn sigmoid(z: f64) f64 {
    if (z >= 0.0) {
        const exp_neg = @exp(-z);
        return 1.0 / (1.0 + exp_neg);
    }

    const exp_pos = @exp(z);
    return exp_pos / (1.0 + exp_pos);
}

pub inline fn asLinearModel(handle: usize) ?*LinearModel {
    if (handle == 0) {
        return null;
    }
    return @as(*LinearModel, @ptrFromInt(handle));
}

pub inline fn asLogisticModel(handle: usize) ?*LogisticModel {
    if (handle == 0) {
        return null;
    }
    return @as(*LogisticModel, @ptrFromInt(handle));
}

pub inline fn asDecisionTreeModel(handle: usize) ?*DecisionTreeModel {
    if (handle == 0) {
        return null;
    }
    return @as(*DecisionTreeModel, @ptrFromInt(handle));
}

pub inline fn asRandomForestClassifierModel(handle: usize) ?*RandomForestClassifierModel {
    if (handle == 0) {
        return null;
    }
    return @as(*RandomForestClassifierModel, @ptrFromInt(handle));
}

pub inline fn giniImpurity(positive_count: usize, sample_count: usize) f64 {
    if (sample_count == 0) {
        return 0.0;
    }
    const p1 = @as(f64, @floatFromInt(positive_count)) / @as(f64, @floatFromInt(sample_count));
    const p0 = 1.0 - p1;
    return 1.0 - p1 * p1 - p0 * p0;
}
