#pragma once

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "types.h"

namespace amr_lidar_localization {
namespace utils {

// 角度标准化到[-pi, pi]
inline double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

// 计算两个角度之间的最小差值
inline double angleDifference(double angle1, double angle2) {
    return std::abs(normalizeAngle(angle1 - angle2));
}

// 检查位置是否在边界内
inline bool isPositionInBounds(
    const Eigen::Vector2d& position,
    const Eigen::Vector2d& bounds_min,
    const Eigen::Vector2d& bounds_max) {
    return position.x() >= bounds_min.x() && position.x() <= bounds_max.x() &&
           position.y() >= bounds_min.y() && position.y() <= bounds_max.y();
}

// 计算协方差矩阵
Eigen::Matrix3d computeCovarianceMatrix(
    const std::vector<WeightedPose>& particles,
    const Pose2D& mean_pose);

} // namespace utils
} // namespace amr_lidar_localization