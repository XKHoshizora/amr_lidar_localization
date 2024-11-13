#pragma once

namespace amr_lidar_localization {

struct Pose2D {
    double x{0}, y{0}, yaw{0};

    Pose2D() = default;
    Pose2D(double _x, double _y, double _yaw) : x(_x), y(_y), yaw(_yaw) {}

    // 计算两个位姿之间的差异
    Pose2D delta(const Pose2D& other) const {
        return Pose2D(other.x - x, other.y - y, other.yaw - yaw);
    }
};

struct WeightedPose {
    Pose2D pose;
    double weight{0.0};

    WeightedPose() = default;
    WeightedPose(const Pose2D& p, double w = 0.0) : pose(p), weight(w) {}
};

struct ParticleFilterStats {
    double effective_particles{0};
    double max_weight{0};
    double min_weight{0};
    double weight_variance{0};
};

} // namespace amr_lidar_localization