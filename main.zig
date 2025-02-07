const std = @import("std");
const math = @import("math.zig");
const Matrix = math.Matrix;
const Vector = math.Vector;

pub fn main() void {
    const inputs: Matrix(3, 4) = .{ .values = [_]Vector(4){
        .{ .values = .{ 1.0, 2.0, 3.0, 2.5 } },
        .{ .values = .{ 2.0, 5.0, -1.0, 2.0 } },
        .{ .values = .{ -1.5, 2.7, 3.3, -0.8 } },
    } };
    const weights: Matrix(3, 4) = .{ .values = [_]Vector(4){
        .{ .values = .{ 0.2, 0.8, -0.5, 1.0 } },
        .{ .values = .{ 0.5, -0.91, 0.26, -0.5 } },
        .{ .values = .{ -0.26, -0.27, 0.17, 0.87 } },
    } };
    const biases: Vector(3) = .{ .values = .{ 2.0, 3.0, 0.5 } };
    const weights_2: Matrix(3, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 0.1, -0.14, 0.5 } },
        .{ .values = .{ -0.5, 0.12, -0.33 } },
        .{ .values = .{ -0.44, 0.73, -0.13 } },
    } };
    const biases_2: Vector(3) = .{ .values = .{ -1.0, 2.0, -0.5 } };

    const layer1_output = inputs.product(weights.transpose()).plus(biases);
    const layer2_output = layer1_output.product(weights_2.transpose()).plus(biases_2);
    std.debug.print("Matrix {d}x{d}\n", .{ layer2_output.rows, layer2_output.cols });
    layer2_output.print();

    const layer = math.LayerDense(2, 5).init();
    std.debug.print("Matrix {d}x{d}\n", .{ layer.weights.rows, layer.weights.cols });
    layer.weights.print();
    std.debug.print("Vector {d}\n", .{layer.biases.values.len});
    layer.biases.print();
}
