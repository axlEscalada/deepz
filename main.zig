const std = @import("std");
const math = @import("math.zig");
const Matrix = math.Matrix;
const Vector = math.Vector;
const LayerDense = math.LayerDense;
const SpiralData = @import("spiral_data.zig");

pub fn main() !void {
    //Chapter 4 ReLu
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    var spiral_data = try SpiralData.createSpiralData(allocator, 100, 3, random);
    defer spiral_data.deinit(allocator);

    const matrix = try Matrix(300, 2).fromSlice(spiral_data.X);
    const y = try Vector(300).fromSlice(spiral_data.y);

    var dense_1 = LayerDense(2, 3).init();
    var dense_2 = LayerDense(3, 3).init();
    var output_dense_1 = dense_1.forward(matrix);
    const output_relu = output_dense_1.reluActivation();
    var output_dense_2 = dense_2.forward(output_relu);
    const output = output_dense_2.softmaxActivation();
    output.print(.{ .to = 5 });
    const loss = try output.calculateLoss(y);
    std.debug.print("\n Loss: {d} \n", .{loss});
}
