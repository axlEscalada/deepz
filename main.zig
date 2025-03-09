const std = @import("std");
const math = @import("math.zig");
const Matrix = math.Matrix;
const Vector = math.Vector;
const LayerDense = math.LayerDense;
const SpiralData = @import("spiral_data.zig");

pub fn main() !void {
    //Chapter 4 ReLu
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();
    //
    // var prng = std.Random.DefaultPrng.init(42);
    // const random = prng.random();
    // var spiral_data = try SpiralData.createSpiralData(allocator, 100, 3, random);
    // defer spiral_data.deinit(allocator);
    //
    // const matrix = try spiral_data.toMatrix(300, 2);
    // matrix.print();
    const matrix: Matrix(4, 2) = .{ .values = [_]Vector(2){
        .{ .values = .{ 0.00000000, 0.00000000 } },
        .{ .values = .{ 0.00073415, 0.01007430 } },
        .{ .values = .{ 0.00431511, 0.01973579 } },
        .{ .values = .{ 0.02011100, 0.02266763 } },
    } };

    var dense_1 = LayerDense(2, 3).init();
    var dense_2 = LayerDense(3, 3).init();
    var output_dense_1 = dense_1.forward(matrix);
    const output_relu = output_dense_1.reluActivation();
    var output_dense_2 = dense_2.forward(output_relu);
    const output = output_dense_2.softmaxActivation();
    output.print();
}
