#include "amr_lidar_localization/utils.h"

namespace amr_lidar_localization {
namespace utils {

Eigen::Matrix3d computeCovarianceMatrix(
    const std::vector<WeightedPose>& particles,
    const Pose2D& mean_pose) {
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    double total_weight = 0;

    for (const auto& particle : particles) {
        double dx = particle.pose.x - mean_pose.x;
        double dy = particle.pose.y - mean_pose.y;
        double dyaw = normalizeAngle(particle.pose.yaw - mean_pose.yaw);

        Eigen::Vector3d diff(dx, dy, dyaw);
        covariance += particle.weight * (diff * diff.transpose());
        total_weight += particle.weight;
    }

    if (total_weight > 0) {
        covariance /= total_weight;
    }

    return covariance;
}

} // namespace utils
} // namespace amr_lidar_localization