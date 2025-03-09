const std = @import("std");
const assert = std.debug.assert;

const MathType = enum {
    Matrix,
    Vector,
};

const Dimension = enum {
    total,
    row,
    col,
};

const Operation = enum {
    Subtract,
    Divide,
};

pub const DimensionOptions = struct {
    dimension: Dimension = .total,
    keep_dim: bool = false,
};

pub fn Matrix(comptime M: usize, comptime N: usize) type {
    return struct {
        values: [M]Vector(N),
        rows: usize = M,
        cols: usize = N,

        pub const Rows = M;
        pub const Cols = N;
        pub const Type = MathType.Matrix;
        const Self = @This();

        pub fn init() Matrix(M, N) {
            return Matrix(M, N){ .values = [_]Vector(N){Vector(N).init()} ** M };
        }

        pub fn dot(self: Self, inputs: [N]f64) Vector(M) {
            var layer_outputs: [M]f64 = [_]f64{0} ** M;

            for (self.values, 0..) |value, i| {
                layer_outputs[i] = value.dot(inputs);
            }

            return .{ .values = layer_outputs };
        }

        pub fn plus(self: Self, inputs: Vector(N)) Matrix(M, N) {
            var outputs = Matrix(M, N).init();

            for (0..M) |i| {
                outputs.values[i].values = inputs.plus(self.values[i]);
            }
            return outputs;
        }

        pub fn subs(self: Self, inputs: anytype) Matrix(M, N) {
            return self.applyOperation(inputs, .Subtract);
        }

        pub fn div(self: Self, inputs: anytype) Matrix(M, N) {
            return self.applyOperation(inputs, .Divide);
        }

        pub fn applyOperation(self: Self, inputs: anytype, op: Operation) Matrix(M, N) {
            const InputType = @TypeOf(inputs);
            var result = Matrix(M, N).init();

            switch (InputType.Type) {
                .Matrix => {
                    comptime assert(InputType.Rows == M and (InputType.Cols == 1 or InputType.Cols == N));

                    for (0..M) |i| {
                        for (0..N) |j| {
                            const idx = if (InputType.Cols == N) j else 0;
                            result.values[i].values[j] = switch (op) {
                                .Subtract => self.values[i].values[j] - inputs.values[i].values[idx],
                                .Divide => self.values[i].values[j] / inputs.values[i].values[idx],
                            };
                        }
                    }
                },
                .Vector => {
                    comptime assert(@TypeOf(inputs).Size == M);

                    for (0..M) |i| {
                        for (0..N) |j| {
                            result.values[i].values[j] = switch (op) {
                                .Subtract => self.values[i].values[j] - inputs.values[j],
                                .Divide => self.values[i].values[j] / inputs.values[j],
                            };
                        }
                    }
                },
            }

            return result;
        }

        pub fn product(self: Self, matrix: anytype) Matrix(M, @TypeOf(matrix).Cols) {
            const MatrixType = @TypeOf(matrix);
            const P = MatrixType.Cols;
            const J = MatrixType.Rows;
            assert(N == J);

            var outputs = Matrix(M, P).init();

            for (0..M) |i| {
                for (0..P) |j| {
                    var o: f64 = 0;
                    for (0..N) |k| {
                        o += self.values[i].values[k] * matrix.values[k].values[j];
                    }
                    outputs.values[i].values[j] = o;
                }
            }

            return outputs;
        }

        pub fn transpose(self: Self) Matrix(N, M) {
            var outputs = Matrix(N, M).init();

            for (0..N) |i| {
                for (0..M) |j| {
                    outputs.values[i].values[j] = self.values[j].values[i];
                }
            }
            return outputs;
        }

        pub fn exp(self: Self) Matrix(M, N) {
            var output = Matrix(M, N).init();

            for (self.values, 0..) |value, i| {
                output.values[i] = value.exp();
            }

            return output;
        }

        pub fn reluActivation(self: Self) Matrix(M, N) {
            var output = Matrix(M, N).init();
            for (0..M) |i| {
                for (0..N) |j| {
                    output.values[i].values[j] = @max(0, self.values[i].values[j]);
                }
            }

            return output;
        }

        pub fn softmaxActivation(self: Self) Matrix(M, N) {
            const exp_values = self.subs(self.max(.{ .dimension = .row, .keep_dim = true })).exp();
            const probabilities = exp_values.div(exp_values.sum(.{ .dimension = .row, .keep_dim = true }));
            return probabilities;
        }

        pub fn sum(self: Self, comptime options: DimensionOptions) switch (options.dimension) {
            .row => if (options.keep_dim) Matrix(M, 1) else Vector(M),
            .col => if (options.keep_dim) Matrix(1, N) else Vector(N),
            .total => if (options.keep_dim) Matrix(1, 1) else Vector(1),
        } {
            switch (options.dimension) {
                .row => {
                    if (options.keep_dim) {
                        var output = Matrix(M, 1).init();
                        for (0..M) |i| {
                            output.values[i].values[0] = self.values[i].sum();
                        }
                        return output;
                    }
                    var output = Vector(M).init();
                    for (0..M) |i| {
                        output.values[i] = self.values[i].sum();
                    }
                    return output;
                },
                .col => {
                    if (options.keep_dim) {
                        var output = Matrix(1, N).init();
                        for (0..N) |j| {
                            for (0..M) |i| {
                                output.values[0].values[j] += self.values[i].values[j];
                            }
                        }
                        return output;
                    }
                    var output = Vector(N).init();
                    for (0..N) |j| {
                        for (0..M) |i| {
                            output.values[j] += self.values[i].values[j];
                        }
                    }
                    return output;
                },
                .total => {
                    var total: f64 = 0.0;
                    for (self.values) |row| {
                        total += row.sum();
                    }

                    if (options.keep_dim) {
                        return Matrix(1, 1){ .values = [_]Vector(1){Vector(1){ .values = [_]f64{total} }} };
                    }
                    return Vector(1){ .values = [_]f64{total} };
                },
            }
        }

        pub fn max(self: Self, comptime options: DimensionOptions) switch (options.dimension) {
            .row => if (options.keep_dim) Matrix(M, 1) else Vector(M),
            .col => if (options.keep_dim) Matrix(1, N) else Vector(N),
            .total => if (options.keep_dim) Matrix(1, 1) else Vector(1),
        } {
            switch (options.dimension) {
                .row => {
                    if (options.keep_dim) {
                        var outputs = Matrix(M, 1).init();

                        for (0..M) |i| {
                            var o = std.math.floatMin(f64);
                            for (0..N) |j| {
                                o = @max(o, self.values[i].values[j]);
                            }
                            outputs.values[i].values[0] = o;
                        }
                        return outputs;
                    }
                    var outputs = Vector(M).init();
                    for (0..M) |i| {
                        var o = std.math.floatMin(f64);
                        for (0..N) |j| {
                            o = @max(o, self.values[i].values[j]);
                        }
                        outputs.values[i] = o;
                    }
                    return outputs;
                },
                .col => {
                    var outputs = Matrix(1, N).init();

                    for (0..N) |i| {
                        var o = std.math.floatMin(f64);
                        for (0..M) |j| {
                            o = @max(o, self.values[j].values[i]);
                        }
                        outputs.values[0].values[i] = o;
                    }
                    return outputs;
                },
                .total => {
                    var total: f64 = std.math.floatMin(f64);

                    for (0..N) |i| {
                        var o = std.math.floatMin(f64);
                        for (0..M) |j| {
                            o = @max(o, self.values[j].values[i]);
                        }
                        total = @max(total, o);
                    }
                    if (options.keep_dim) {
                        return Matrix(1, 1){ .values = [_]Vector(1){Vector(1){ .values = [_]f64{total} }} };
                    }
                    return Vector(1){ .values = [_]f64{total} };
                },
            }
        }

        pub fn print(self: Self) void {
            std.debug.print("[", .{});
            for (self.values, 0..) |row, i| {
                row.print();
                if (i < M - 1) {
                    std.debug.print(",", .{});
                } else {
                    std.debug.print("]", .{});
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
        pub const Size = M;
        pub const Type = MathType.Vector;

        pub fn init() Vector(M) {
            return Vector(M){ .values = [_]f64{0} ** M };
        }

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

        pub fn exp(self: Self) Vector(M) {
            var output = Vector(M).init();

            for (self.values, 0..) |value, i| {
                output.values[i] = @exp(value);
            }

            return output;
        }

        pub fn sum(self: Self) f64 {
            var norm_base: f64 = 0.0;
            for (self.values) |value| {
                norm_base += value;
            }
            return norm_base;
        }

        pub fn max(self: Self, comptime options: DimensionOptions) switch (options.dimension) {
            .row => if (options.keep_dim) Vector(1) else f64,
            .col => @compileError("Dimension .col is not suitable for max."),
            .total => if (options.keep_dim) Vector(1) else f64,
        } {
            switch (options.dimension) {
                .row, .total => {
                    if (options.keep_dim) {
                        var outputs = Vector(1).init();

                        var o = std.math.floatMin(f64);
                        for (0..M) |i| {
                            o = @max(o, self.values[i]);
                            outputs.values[0] = o;
                        }
                        return outputs;
                    }

                    var o = std.math.floatMin(f64);
                    for (0..M) |i| {
                        o = @max(o, self.values[i]);
                    }
                    return o;
                },
                .col => {
                    unreachable;
                },
            }
        }

        pub fn print(self: Self) void {
            // First check if we have any negative numbers
            var has_negative = false;
            for (self.values) |value| {
                if (value < 0) {
                    has_negative = true;
                    break;
                }
            }

            std.debug.print("[ ", .{});
            for (self.values, 0..) |value, i| {
                if (has_negative and value >= 0) {
                    std.debug.print(" ", .{});
                }
                std.debug.print("{d:.8}", .{value});
                if (i < M - 1) {
                    std.debug.print(", ", .{});
                }
            }
            std.debug.print(" ]", .{});
        }
    };
}

pub fn LayerDense(comptime M: usize, comptime N: usize) type {
    return struct {
        weights: Matrix(M, N),
        biases: Vector(N),

        const Self = @This();

        // pub fn init() LayerDense(M, N) {
        //     var m = Matrix(M, N).init();
        //     var b = Vector(N).init();
        //
        //     // Use fixed weights and biases for testing
        //     const fixed_weights = [_][N]f64{
        //         [_]f64{ 0.01, 0.02, 0.03 } ** (N / 3),
        //         [_]f64{ 0.04, 0.05, 0.06 } ** (N / 3),
        //     };
        //
        //     const fixed_biases = [_]f64{ 0.1, 0.2, 0.3 } ** (N / 3);
        //
        //     // Initialize with fixed values
        //     for (0..M) |i| {
        //         for (0..N) |j| {
        //             m.values[i].values[j] = fixed_weights[i % fixed_weights.len][j % fixed_weights[0].len];
        //         }
        //     }
        //
        //     for (0..N) |j| {
        //         b.values[j] = fixed_biases[j % fixed_biases.len];
        //     }
        //
        //     return .{
        //         .weights = m,
        //         .biases = b,
        //     };
        // }

        pub fn init() LayerDense(M, N) {
            var m = Matrix(M, N).init();
            var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
            var random = prng.random();

            const b = Vector(N).init();

            for (0..M) |i| {
                for (0..N) |j| {
                    m.values[i].values[j] = 0.01 * random.floatNorm(f32);
                }
            }

            return .{
                .weights = m,
                .biases = b,
            };
        }

        pub fn forward(self: *Self, inputs: anytype) Matrix(@TypeOf(inputs).Rows, N) {
            const MatrixType = @TypeOf(inputs);
            const P = MatrixType.Cols;
            comptime assert(M == P);
            return inputs.product(self.weights).plus(self.biases);
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
    const biases = Vector(3){ .values = [_]f64{ 2.0, 3.0, 0.5 } };

    const output = weights.dot(inputs);
    const final = output.plus(biases);

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

test "test exp function in Vector" {
    // Create a sample vector with 4 elements for testing
    var input_vector = Vector(4).init();
    input_vector.values = [_]f64{ 0.0, 1.0, -1.0, 2.0 };

    // Call the exp method on the vector
    const result_vector = input_vector.exp();

    // Define the expected output
    const expected_values = [_]f64{
        1.0, // exp(0) = 1.0
        2.718281828459045, // exp(1) = e^1
        0.36787944117144233, // exp(-1) = e^-1
        7.38905609893065, // exp(2) = e^2
    };

    // Allow for floating-point tolerance
    const tolerance = 0.0001;

    // Assert that the results match the expected values
    for (0..4) |i| {
        const diff = result_vector.values[i] - expected_values[i];
        assert(@abs(diff) <= tolerance);
    }
}

test "test sum function total keeping dim" {
    const matrix_a: Matrix(3, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 0, 1, 2 } },
        .{ .values = .{ 5, 8, 7 } },
        .{ .values = .{ 20, 11, 12 } },
    } };

    const result = matrix_a.sum(.{ .dimension = .total, .keep_dim = true });

    try std.testing.expectEqual(result.values[0].values[0], 66);
}

test "test sum function total not keeping dim" {
    const matrix_a: Matrix(3, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 0, 1, 2 } },
        .{ .values = .{ 5, 8, 7 } },
        .{ .values = .{ 20, 11, 12 } },
    } };

    const result = matrix_a.sum(.{ .dimension = .total, .keep_dim = false });

    try std.testing.expectEqual(result.values[0], 66);
}

test "test max function rows dimension and keeping dimension" {
    const matrix_a: Matrix(3, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 0, 1, 2 } },
        .{ .values = .{ 5, 8, 7 } },
        .{ .values = .{ 20, 11, 12 } },
    } };

    const result = matrix_a.max(.{ .dimension = .row, .keep_dim = true });

    try std.testing.expectEqual(result.values[0].values[0], 2);
    try std.testing.expectEqual(result.values[1].values[0], 8);
    try std.testing.expectEqual(result.values[2].values[0], 20);
}

test "test max function cols dimension and keeping dimension" {
    const matrix_a: Matrix(3, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 0, 1, 2 } },
        .{ .values = .{ 5, 8, 7 } },
        .{ .values = .{ 20, 11, 12 } },
    } };

    const result = matrix_a.max(.{ .dimension = .col, .keep_dim = true });

    try std.testing.expectEqual(result.values[0].values[0], 20);
    try std.testing.expectEqual(result.values[0].values[1], 11);
    try std.testing.expectEqual(result.values[0].values[2], 12);
}

test "test max function total keeping dim" {
    const matrix_a: Matrix(3, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 0, 1, 2 } },
        .{ .values = .{ 5, 8, 7 } },
        .{ .values = .{ 20, 11, 12 } },
    } };

    const result = matrix_a.max(.{ .dimension = .total, .keep_dim = true });

    try std.testing.expectEqual(result.values[0].values[0], 20);
}

test "test max function total not keeping dim" {
    const matrix_a: Matrix(3, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 0, 1, 2 } },
        .{ .values = .{ 5, 8, 7 } },
        .{ .values = .{ 20, 11, 12 } },
    } };

    const result = matrix_a.max(.{ .dimension = .total, .keep_dim = false });

    try std.testing.expectEqual(result.values[0], 20);
}

test "softmax activation test" {
    const matrix: Matrix(1, 3) = .{ .values = [_]Vector(3){
        .{ .values = .{ 1.0, 2.0, 3.0 } },
    } };

    const result = matrix.softmaxActivation();

    // Expected values after softmax:
    // exp(1) / (exp(1) + exp(2) + exp(3)) ≈ 0.09003
    // exp(2) / (exp(1) + exp(2) + exp(3)) ≈ 0.24473
    // exp(3) / (exp(1) + exp(2) + exp(3)) ≈ 0.66524
    try std.testing.expectApproxEqAbs(result.values[0].values[0], 0.09003, 0.0001);
    try std.testing.expectApproxEqAbs(result.values[0].values[1], 0.24473, 0.0001);
    try std.testing.expectApproxEqAbs(result.values[0].values[2], 0.66524, 0.0001);
}

test "test vector max function row not keeping dim" {
    const vector: Vector(3) = Vector(3){ .values = .{ 0, 1, 2 } };

    const result = vector.max(.{ .dimension = .row, .keep_dim = false });

    try std.testing.expectEqual(result, 2);
}

test "test vector max function row keeping dim" {
    const vector: Vector(3) = Vector(3){ .values = .{ 0, 1, 2 } };

    const result = vector.max(.{ .dimension = .row, .keep_dim = true });

    try std.testing.expectEqual(result.values[0], 2);
}

test "test vector max function total not keeping dim" {
    const vector: Vector(3) = Vector(3){ .values = .{ 0, 1, 2 } };

    const result = vector.max(.{ .dimension = .total, .keep_dim = false });

    try std.testing.expectEqual(result, 2);
}

test "test vector max function total keeping dim" {
    const vector: Vector(3) = Vector(3){ .values = .{ 0, 1, 2 } };

    const result = vector.max(.{ .dimension = .total, .keep_dim = true });

    try std.testing.expectEqual(result.values[0], 2);
}
