const std = @import("std");
const Random = std.Random;
const math = std.math;
const builtin = @import("builtin");

pub const SpiralData = struct {
    X: [][]f64, // Changed from f32
    y: []f64, // Changed from u8

    pub fn deinit(self: *SpiralData, allocator: std.mem.Allocator) void {
        for (self.X) |row| {
            allocator.free(row);
        }
        allocator.free(self.X);
        allocator.free(self.y);
    }

    // Update getDataBounds to use f64
    pub fn getDataBounds(self: SpiralData) struct { min_x: f64, max_x: f64, min_y: f64, max_y: f64 } {
        var min_x: f64 = self.X[0][0];
        var max_x: f64 = self.X[0][0];
        var min_y: f64 = self.X[0][1];
        var max_y: f64 = self.X[0][1];

        for (self.X) |point| {
            min_x = @min(min_x, point[0]);
            max_x = @max(max_x, point[0]);
            min_y = @min(min_y, point[1]);
            max_y = @max(max_y, point[1]);
        }

        return .{
            .min_x = min_x,
            .max_x = max_x,
            .min_y = min_y,
            .max_y = max_y,
        };
    }

    // Update ScaleFn to use f64
    const ScaleFn = struct {
        min_val: f64,
        max_val: f64,
        out_min: f64,
        out_max: f64,
        invert: bool = false,

        pub fn scale(self: ScaleFn, val: f64) f64 {
            const normalized = (val - self.min_val) / (self.max_val - self.min_val);
            const scaled = if (self.invert)
                (1 - normalized) * (self.out_max - self.out_min) + self.out_min
            else
                normalized * (self.out_max - self.out_min) + self.out_min;
            return scaled;
        }
    };

    // Update saveAsPPM to handle f64
    pub fn saveAsPPM(self: SpiralData, file_path: []const u8) !void {
        const width: u32 = 800;
        const height: u32 = 800;
        const padding: u32 = 50;
        const point_radius: u32 = 3;

        var image_data = try std.ArrayList(u8).initCapacity(std.heap.page_allocator, width * height * 3);
        defer image_data.deinit();

        // Fill with white background
        var i: usize = 0;
        while (i < width * height * 3) : (i += 3) {
            try image_data.appendSlice(&[_]u8{ 255, 255, 255 });
        }

        const bounds = self.getDataBounds();

        const scale_x = ScaleFn{
            .min_val = bounds.min_x,
            .max_val = bounds.max_x,
            .out_min = @floatFromInt(padding),
            .out_max = @floatFromInt(width - padding),
        };

        const scale_y = ScaleFn{
            .min_val = bounds.min_y,
            .max_val = bounds.max_y,
            .out_min = @floatFromInt(padding),
            .out_max = @floatFromInt(height - padding),
            .invert = true,
        };

        // Colors for each class
        const colors = [_][3]u8{
            [_]u8{ 255, 0, 0 }, // Red
            [_]u8{ 0, 255, 0 }, // Green
            [_]u8{ 0, 0, 255 }, // Blue
        };

        // Helper function to set a pixel
        const setPixel = struct {
            fn set(data: *std.ArrayList(u8), x: u32, y: u32, w: u32, color: [3]u8) !void {
                const idx = (y * w + x) * 3;
                if (idx + 2 < data.items.len) {
                    data.items[idx] = color[0];
                    data.items[idx + 1] = color[1];
                    data.items[idx + 2] = color[2];
                }
            }
        }.set;

        // Draw points
        for (self.X, self.y) |point, class| {
            const x = @as(u32, @intFromFloat(scale_x.scale(point[0])));
            const y = @as(u32, @intFromFloat(scale_y.scale(point[1])));

            var dy: i32 = -@as(i32, point_radius);
            while (dy <= point_radius) : (dy += 1) {
                var dx: i32 = -@as(i32, point_radius);
                while (dx <= point_radius) : (dx += 1) {
                    if (dx * dx + dy * dy <= point_radius * point_radius) {
                        const px = @as(i32, @intCast(x)) + dx;
                        const py = @as(i32, @intCast(y)) + dy;
                        if (px >= 0 and px < width and py >= 0 and py < height) {
                            try setPixel(&image_data, @intCast(px), @intCast(py), width, colors[@as(usize, @intFromFloat(class))]);
                        }
                    }
                }
            }
        }

        // Draw axes
        const black = [_]u8{ 0, 0, 0 };
        const origin_x = @as(u32, @intFromFloat(scale_x.scale(0)));
        const origin_y = @as(u32, @intFromFloat(scale_y.scale(0)));

        var x: u32 = padding;
        while (x < width - padding) : (x += 1) {
            try setPixel(&image_data, x, origin_y, width, black);
        }

        var y: u32 = padding;
        while (y < height - padding) : (y += 1) {
            try setPixel(&image_data, origin_x, y, width, black);
        }

        const file = try std.fs.cwd().createFile(file_path, .{});
        defer file.close();
        const writer = file.writer();

        try writer.print("P6\n{d} {d}\n255\n", .{ width, height });
        try writer.writeAll(image_data.items);
    }
};

pub fn createSpiralData(
    allocator: std.mem.Allocator,
    samples: usize,
    classes: usize,
    rng: Random,
) !SpiralData {
    const total_points = samples * classes;

    var X = try allocator.alloc([]f64, total_points);
    for (X) |*row| {
        row.* = try allocator.alloc(f64, 2);
    }
    var y = try allocator.alloc(f64, total_points);

    var class: usize = 0;
    while (class < classes) : (class += 1) {
        const start_idx = class * samples;

        var i: usize = 0;
        while (i < samples) : (i += 1) {
            const idx = start_idx + i;

            const r = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(samples - 1));
            const base_t = @as(f64, @floatFromInt(class * 4)) +
                (@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(samples))) * 4.0;
            const noise = (rng.float(f64) - 0.5) * 0.4;
            const t = base_t + noise;

            X[idx][0] = r * math.sin(t * 2.5);
            X[idx][1] = r * math.cos(t * 2.5);
            y[idx] = @floatFromInt(class);
        }
    }

    return SpiralData{ .X = X, .y = y };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Create spiral data
    var spiral_data = try createSpiralData(allocator, 100, 3, random);
    defer spiral_data.deinit(allocator);

    const file_path = "spiral_plot.ppm";

    // Save as PPM
    try spiral_data.saveAsPPM(file_path);

    // Open with default viewer
    const argv = [_][]const u8{
        switch (builtin.target.os.tag) {
            .windows => "start",
            .macos => "open",
            .linux => "xdg-open",
            else => @compileError("Unsupported OS"),
        },
        file_path,
    };

    var child_process = std.process.Child.init(&argv, allocator);
    _ = try child_process.spawnAndWait();

    std.debug.print("Plot saved as '{s}' and opened in default viewer\n", .{file_path});
}
