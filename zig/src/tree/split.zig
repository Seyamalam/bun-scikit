const std = @import("std");
const common = @import("../common.zig");

const EXACT_SPLIT_MAX_SAMPLES: usize = 88;

const FeatureSample = struct {
    value: f64,
    label: u8,
};

fn featureSampleLessThan(_: void, a: FeatureSample, b: FeatureSample) bool {
    return a.value < b.value;
}

pub fn resolveMaxFeatures(model: *const common.DecisionTreeModel) usize {
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

pub fn selectCandidateFeatures(
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

pub fn findBestSplitForFeature(
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

    if (sample_count <= EXACT_SPLIT_MAX_SAMPLES) {
        var sorted: [EXACT_SPLIT_MAX_SAMPLES]FeatureSample = undefined;
        for (indices, 0..) |sample_index, idx| {
            sorted[idx] = .{
                .value = x_ptr[sample_index * model.n_features + feature_index],
                .label = y_ptr[sample_index],
            };
        }
        const sorted_samples = sorted[0..sample_count];
        std.sort.pdq(FeatureSample, sorted_samples, {}, featureSampleLessThan);

        var left_count: usize = 0;
        var left_positive: usize = 0;
        var best_impurity = std.math.inf(f64);
        var best_threshold: f64 = 0.0;
        var found = false;

        var split_index: usize = 0;
        while (split_index + 1 < sample_count) : (split_index += 1) {
            left_count += 1;
            left_positive += sorted_samples[split_index].label;

            const current_value = sorted_samples[split_index].value;
            const next_value = sorted_samples[split_index + 1].value;
            if (current_value == next_value) {
                continue;
            }

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
                best_threshold = (current_value + next_value) * 0.5;
                found = true;
            }
        }

        if (!found) {
            return null;
        }
        return .{
            .threshold = best_threshold,
            .impurity = best_impurity,
        };
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

    return .{
        .threshold = best_threshold,
        .impurity = best_impurity,
    };
}

pub fn partitionIndicesForThreshold(
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

    return .{
        .left_indices = left_indices,
        .right_indices = right_indices,
    };
}
