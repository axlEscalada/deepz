const std = @import("std");
const math = @import("math.zig");
const Matrix = math.Matrix;
const Vector = math.Vector;
const LayerDense = math.LayerDense;
const SpiralData = @import("spiral_data.zig");

pub fn main() !void {
    const layer_outputs: Matrix(3, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 4.8, 1.21, 2.385 } },
        .{ .values = .{ 8.9, -1.81, 0.2 } },
        .{ .values = .{ 1.41, 1.051, 0.026 } },
    } };
    // layer_outputs.print();
    // std.debug.print("\n", .{});
    const max = layer_outputs.max(.{ .dimension = .row, .keep_dim = true });
    // max.print();
    // std.debug.print("\n", .{});
    const op = layer_outputs.subs(max);
    // op.print();
    // std.debug.print("\n", .{});
    const exp_values = op.exp();
    exp_values.print();
    const sum = exp_values.sum(.{ .dimension = .row, .keep_dim = true });
    sum.print();
    const probabilities = exp_values.div(sum);
    probabilities.print();
}
