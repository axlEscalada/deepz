const std = @import("std");
const assert = std.debug.assert;

pub fn Matrix(comptime M: usize, comptime N: usize) type {
    return struct {
        values: [M]Vector(N),

        pub const Rows = M;
        pub const Cols = N;
        const Self = @This();

        pub fn dot(self: Self, inputs: [N]f64) Vector(M) {
            var layer_outputs: [M]f64 = [_]f64{0} ** M;

            for (self.values, 0..) |value, i| {
                layer_outputs[i] = value.dot(inputs);
            }

            return .{ .values = layer_outputs };
        }

        pub fn plus(self: Self, inputs: Vector(N)) Matrix(M, N) {
            var outputs = Matrix(N, M){ .values = [_]Vector(M){Vector(M){ .values = [_]f64{0} ** M }} ** N };

            for (0..M) |i| {
                outputs.values[i].values = inputs.plus(self.values[i]);
            }
            return outputs;
        }

        pub fn product(self: Self, matrix: anytype) Matrix(M, @TypeOf(matrix).Cols) {
            const MatrixType = @TypeOf(matrix);
            const P = MatrixType.Cols;
            assert(M == P);

            var outputs = Matrix(M, P){ .values = [_]Vector(P){Vector(P){ .values = [_]f64{0} ** P }} ** M };

            for (0..M) |i| {
                for (0..P) |j| {
                    var sum: f64 = 0;
                    for (0..N) |k| {
                        sum += self.values[i].values[k] * matrix.values[k].values[j];
                    }
                    outputs.values[i].values[j] = sum;
                }
            }

            return outputs;
        }

        pub fn transpose(self: Self) Matrix(N, M) {
            var outputs = Matrix(N, M){ .values = [_]Vector(M){Vector(M){ .values = [_]f64{0} ** M }} ** N };

            for (0..N) |i| {
                for (0..M) |j| {
                    outputs.values[i].values[j] = self.values[j].values[i];
                }
            }
            return outputs;
        }

        pub fn print(self: Self) void {
            std.debug.print("\nMatrix {}x{}:\n", .{ M, N });
            for (self.values) |row| {
                for (row.values) |value| {
                    std.debug.print("{d:.4} ", .{value});
                }
                std.debug.print("\n", .{});
            }
        }
    };
}

pub fn Vector(comptime M: usize) type {
    return struct {
        values: [M]f64,

        const Self = @This();

        pub fn plus(self: Self, vector: Vector(M)) [M]f64 {
            var output = [_]f64{0} ** M;

            for (self.values, 0..) |value, i| {
                output[i] = value + vector.values[i];
            }

            return output;
        }

        pub fn dot(self: Self, vector: [M]f64) f64 {
            var output: f64 = 0;

            for (self.values, 0..) |input, i| {
                output += input * vector[i];
            }
            return output;
        }
    };
}

test "matrix product" {
    const weights: Matrix(3, 4) = .{ .values = [_]Vector(4){
        .{ .values = .{ 0.2, 0.8, -0.5, 1.0 } },
        .{ .values = .{ 0.5, -0.91, 0.26, -0.5 } },
        .{ .values = .{ -0.26, -0.27, 0.17, 0.87 } },
    } };
    const inputs = [_]f64{ 1.0, 2.0, 3.0, 2.5 };
    var biases = [_]f64{ 2.0, 3.0, 0.5 };

    const output = weights.dot(inputs);
    const final = output.plus(biases[0..]);

    try std.testing.expectApproxEqAbs(final[0], 4.8, 0.0001);
    try std.testing.expectApproxEqAbs(final[1], 1.21, 0.0001);
    try std.testing.expectApproxEqAbs(final[2], 2.385, 0.0001);
}

test "vector product" {
    const weights: Matrix(3, 4) = .{ .values = [_]Vector(4){
        .{ .values = .{ 0.2, 0.8, -0.5, 1.0 } },
        .{ .values = .{ 0.5, -0.91, 0.26, -0.5 } },
        .{ .values = .{ -0.26, -0.27, 0.17, 0.87 } },
    } };
    const inputs = [_]f64{ 1.0, 2.0, 3.0, 2.5 };

    const final = [_]f64{ weights.values[0].dot(inputs), weights.values[1].dot(inputs), weights.values[2].dot(inputs) };

    try std.testing.expectApproxEqAbs(final[0], 2.8, 0.0001);
    try std.testing.expectApproxEqAbs(final[1], -1.79, 0.0001);
    try std.testing.expectApproxEqAbs(final[2], 1.885, 0.0001);
}

test "matrix multiplication" {
    const matrix_a: Matrix(2, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 1.0, 2.0, 3.0 } },
        .{ .values = .{ 4.0, 5.0, 6.0 } },
    } };

    const matrix_b: Matrix(3, 2) = .{ .values = [_]Vector(2){
        .{ .values = .{ 7.0, 8.0 } },
        .{ .values = .{ 9.0, 10.0 } },
        .{ .values = .{ 11.0, 12.0 } },
    } };

    const result = matrix_a.product(matrix_b);

    try std.testing.expectApproxEqAbs(result.values[0].values[0], 58.0, 0.0001);
    try std.testing.expectApproxEqAbs(result.values[0].values[1], 64.0, 0.0001);
    try std.testing.expectApproxEqAbs(result.values[1].values[0], 139.0, 0.0001);
    try std.testing.expectApproxEqAbs(result.values[1].values[1], 154.0, 0.0001);
}

test "matrix multiplication 2" {
    const matrix_a: Matrix(5, 4) = .{ .values = [_]Vector(4){
        .{ .values = .{ 0.49, 0.97, 0.53, 0.05 } },
        .{ .values = .{ 0.33, 0.65, 0.62, 0.51 } },
        .{ .values = .{ 1.00, 0.38, 0.61, 0.45 } },
        .{ .values = .{ 0.74, 0.27, 0.64, 0.17 } },
        .{ .values = .{ 0.36, 0.17, 0.96, 0.12 } },
    } };

    const matrix_b: Matrix(4, 5) = .{ .values = [_]Vector(5){
        .{ .values = .{ 0.79, 0.32, 0.68, 0.90, 0.77 } },
        .{ .values = .{ 0.18, 0.39, 0.12, 0.93, 0.09 } },
        .{ .values = .{ 0.87, 0.42, 0.60, 0.71, 0.12 } },
        .{ .values = .{ 0.45, 0.55, 0.40, 0.78, 0.81 } },
    } };

    const result = matrix_a.product(matrix_b);

    try std.testing.expectApproxEqAbs(result.values[0].values[0], 1.05, 0.01);
    try std.testing.expectApproxEqAbs(result.values[0].values[1], 0.79, 0.01);
    try std.testing.expectApproxEqAbs(result.values[0].values[2], 0.79, 0.01);
    try std.testing.expectApproxEqAbs(result.values[0].values[3], 1.76, 0.01);
    try std.testing.expectApproxEqAbs(result.values[0].values[4], 0.57, 0.01);
}

test "matrix multiplication 3" {
    const matrix_a: Matrix(1, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 1, 2, 3 } },
    } };

    const matrix_b: Matrix(3, 1) = .{ .values = [_]Vector(1){
        .{ .values = .{2} },
        .{ .values = .{3} },
        .{ .values = .{4} },
    } };

    const result = matrix_a.product(matrix_b);

    try std.testing.expectApproxEqAbs(result.values[0].values[0], 20, 0.01);
}

test "matrix transpose" {
    const matrix_a: Matrix(4, 5) = .{ .values = [_]Vector(5){
        .{ .values = .{ 0, 1, 2, 3, 4 } },
        .{ .values = .{ 5, 6, 7, 8, 9 } },
        .{ .values = .{ 10, 11, 12, 13, 14 } },
        .{ .values = .{ 15, 16, 17, 18, 19 } },
    } };

    const result = matrix_a.transpose();

    try std.testing.expectApproxEqAbs(result.values[0].values[0], 0, 0.01);
    try std.testing.expectApproxEqAbs(result.values[0].values[1], 5, 0.01);
    try std.testing.expectApproxEqAbs(result.values[0].values[2], 10, 0.01);
    try std.testing.expectApproxEqAbs(result.values[0].values[3], 15, 0.01);
}
