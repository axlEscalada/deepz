const std = @import("std");
const math = @import("math.zig");
const Matrix = math.Matrix;
const Vector = math.Vector;
const LayerDense = math.LayerDense;
const SpiralData = @import("spiral_data.zig");

pub fn main() !void {
    // const inputs: Matrix(3, 4) = .{ .values = [_]Vector(4){
    //     .{ .values = .{ 1.0, 2.0, 3.0, 2.5 } },
    //     .{ .values = .{ 2.0, 5.0, -1.0, 2.0 } },
    //     .{ .values = .{ -1.5, 2.7, 3.3, -0.8 } },
    // } };
    // const weights: Matrix(3, 4) = .{ .values = [_]Vector(4){
    //     .{ .values = .{ 0.2, 0.8, -0.5, 1.0 } },
    //     .{ .values = .{ 0.5, -0.91, 0.26, -0.5 } },
    //     .{ .values = .{ -0.26, -0.27, 0.17, 0.87 } },
    // } };
    // const biases: Vector(3) = .{ .values = .{ 2.0, 3.0, 0.5 } };
    // const weights_2: Matrix(3, 3) = .{ .values = [_]Vector(3){
    //     .{ .values = .{ 0.1, -0.14, 0.5 } },
    //     .{ .values = .{ -0.5, 0.12, -0.33 } },
    //     .{ .values = .{ -0.44, 0.73, -0.13 } },
    // } };
    // const biases_2: Vector(3) = .{ .values = .{ -1.0, 2.0, -0.5 } };
    //
    // const layer1_output = inputs.product(weights.transpose()).plus(biases);
    // const layer2_output = layer1_output.product(weights_2.transpose()).plus(biases_2);
    // std.debug.print("Matrix {d}x{d}\n", .{ layer2_output.rows, layer2_output.cols });
    // layer2_output.print();
    //
    // const layer = math.LayerDense(2, 5).init();
    // std.debug.print("Matrix {d}x{d}\n", .{ layer.weights.rows, layer.weights.cols });
    // layer.weights.print();
    // std.debug.print("Vector {d}\n", .{layer.biases.values.len});
    // layer.biases.print();
    //

    //Chapter 4
    // const inputs = Vector(8){ .values = .{ 0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100 } };
    // const output = math.reluActivation(inputs);
    // output.print();

    //Chapter 4 ReLu
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();
    //
    // var prng = std.Random.DefaultPrng.init(42);
    // const random = prng.random();
    //
    // var spiral_data = try SpiralData.createSpiralData(allocator, 100, 3, random);
    // defer spiral_data.deinit(allocator);
    // const matrix = try spiral_data.toMatrix(300, 2);
    // var dense_1 = LayerDense(2, 3).init();
    // var output = dense_1.forward(matrix);
    // output.reluActivation().print();
    // output.print();

    // const layer_outputs = Vector(3){ .values = [_]f64{ 4.8, 1.21, 2.385 } };
    // var exp_values = layer_outputs.exp();
    // exp_values.print();
    // const norm_base = exp_values.sum();
    // for (layer_outputs, 0..) |o, i| {
    //     exp_values[i] = std.math.pow(f64, std.math.e, o);
    //     std.debug.print("{d}\n", .{exp_values[i]});
    // }

    // const norm_base: f64 = sum(&exp_values);
    //
    // var norm_values = [_]f64{0} ** 3;
    // for (exp_values, 0..) |o, i| {
    //     norm_values[i] = o / norm_base;
    // }
    // const norm_values_sum = sum(&norm_values);
    //
    // std.debug.print("[", .{});
    // for (norm_values) |i| {
    //     std.debug.print("{d}, ", .{i});
    // }
    // std.debug.print("]\n", .{});
    // std.debug.print("{d}\n", .{norm_values_sum});
    //
    const layer_outputs: Matrix(3, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 4.8, 1.21, 2.385 } },
        .{ .values = .{ 8.9, -1.81, 0.2 } },
        .{ .values = .{ 1.41, 1.051, 0.026 } },
    } };

    const sum_total_default = layer_outputs.sum(.{});
    sum_total_default.print();
    std.debug.print("\n", .{});
    const sum_total = layer_outputs.sum(.{ .dimension = .total });
    sum_total.print();
    std.debug.print("\n", .{});
    const sum_rows = layer_outputs.sum(.{ .dimension = .rows });
    sum_rows.print();
    std.debug.print("\n", .{});
    const sum_columns = layer_outputs.sum(.{ .dimension = .cols });
    sum_columns.print();
    std.debug.print("\n", .{});
    const sum_rows_keep_dim = layer_outputs.sum(.{ .dimension = .rows, .keep_dim = true });
    sum_rows_keep_dim.print();
    std.debug.print("\n", .{});
}
