const std = @import("std");
const math = std.math;
const Random = std.Random;
const builtin = @import("builtin");

pub const SpiralData = struct {
    X: [][]f32,
    y: []u8,

    pub fn deinit(self: *SpiralData, allocator: std.mem.Allocator) void {
        for (self.X) |row| {
            allocator.free(row);
        }
        allocator.free(self.X);
        allocator.free(self.y);
    }

    // Find the min and max values for scaling
    pub fn getDataBounds(self: SpiralData) struct { min_x: f32, max_x: f32, min_y: f32, max_y: f32 } {
        var min_x: f32 = self.X[0][0];
        var max_x: f32 = self.X[0][0];
        var min_y: f32 = self.X[0][1];
        var max_y: f32 = self.X[0][1];

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

    const ScaleFn = struct {
        min_val: f32,
        max_val: f32,
        out_min: f32,
        out_max: f32,
        invert: bool = false,

        pub fn scale(self: ScaleFn, val: f32) f32 {
            const normalized = (val - self.min_val) / (self.max_val - self.min_val);
            const scaled = if (self.invert)
                (1 - normalized) * (self.out_max - self.out_min) + self.out_min
            else
                normalized * (self.out_max - self.out_min) + self.out_min;
            return scaled;
        }
    };

    // Generate SVG visualization
    pub fn saveAsSVG(self: SpiralData, file_path: []const u8) !void {
        const width: u32 = 800;
        const height: u32 = 800;
        const padding: u32 = 50;
        const point_radius: f32 = 3;

        // Get data bounds for scaling
        const bounds = self.getDataBounds();

        // Create scale functions
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

        // Open file for writing
        const file = try std.fs.cwd().createFile(file_path, .{});
        defer file.close();
        const writer = file.writer();

        // Write SVG header
        try writer.writeAll(
            \\<?xml version="1.0" encoding="UTF-8" standalone="no"?>
            \\<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
            \\
        );

        try writer.print(
            \\<svg width="{d}" height="{d}" xmlns="http://www.w3.org/2000/svg">
            \\<rect width="100%" height="100%" fill="white"/>
            \\
        , .{ width, height });

        // Draw points
        const colors = [_][]const u8{ "#ff0000", "#00ff00", "#0000ff" };
        for (self.X, self.y) |point, class| {
            const x = scale_x.scale(point[0]);
            const y = scale_y.scale(point[1]);
            try writer.print(
                \\<circle cx="{d:.2}" cy="{d:.2}" r="{d:.2}" fill="{s}" opacity="0.6"/>
                \\
            , .{ x, y, point_radius, colors[@as(usize, class)] });
        }

        // Draw axes
        const origin_x = scale_x.scale(0);
        const origin_y = scale_y.scale(0);

        // X-axis
        try writer.print(
            \\<line x1="{d}" y1="{d}" x2="{d}" y2="{d}" stroke="black" stroke-width="1"/>
            \\
        , .{ padding, origin_y, width - padding, origin_y });

        // Y-axis
        try writer.print(
            \\<line x1="{d}" y1="{d}" x2="{d}" y2="{d}" stroke="black" stroke-width="1"/>
            \\
        , .{ origin_x, padding, origin_x, height - padding });

        // Close SVG tag
        try writer.writeAll("</svg>\n");
    }
};

pub fn createSpiralData(
    allocator: std.mem.Allocator,
    samples: usize,
    classes: usize,
    rng: Random,
) !SpiralData {
    const total_points = samples * classes;

    var X = try allocator.alloc([]f32, total_points);
    for (X) |*row| {
        row.* = try allocator.alloc(f32, 2);
    }
    var y = try allocator.alloc(u8, total_points);

    var class: usize = 0;
    while (class < classes) : (class += 1) {
        const start_idx = class * samples;

        var i: usize = 0;
        while (i < samples) : (i += 1) {
            const idx = start_idx + i;

            const r = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(samples - 1));
            const base_t = @as(f32, @floatFromInt(class * 4)) +
                (@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(samples))) * 4.0;
            const noise = (rng.float(f32) - 0.5) * 0.4;
            const t = base_t + noise;

            X[idx][0] = r * math.sin(t * 2.5);
            X[idx][1] = r * math.cos(t * 2.5);
            y[idx] = @intCast(class);
        }
    }

    return SpiralData{ .X = X, .y = y };
}

// pub fn main() !void {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();
//     const allocator = gpa.allocator();
//
//     var prng = .Random.DefaultPrng.init(42);
//     const random = prng.random();
//
//     // Create spiral data
//     var spiral_data = try createSpiralData(allocator, 100, 3, random);
//     defer spiral_data.deinit(allocator);
//
//     // Save as SVG
//     try spiral_data.saveAsSVG("spiral_plot.svg");
//
//     std.debug.print("Plot saved as 'spiral_plot.svg'\n", .{});
// }

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Create spiral data
    var spiral_data = try createSpiralData(allocator, 100, 3, random);
    defer spiral_data.deinit(allocator);

    const file_path = "spiral_plot.svg";

    // Save as SVG
    try spiral_data.saveAsSVG(file_path);

    // Open the file with the system's default viewer
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
