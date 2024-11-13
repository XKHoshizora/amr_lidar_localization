#include "amr_lidar_localization/particle_filter.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include "amr_lidar_localization/utils.h"

namespace amr_lidar_localization {

ParticleFilter::ParticleFilter(int num_particles,
                             double resample_threshold,
                             const Eigen::Vector3d& motion_noise)
    : num_particles_(num_particles)
    , resample_threshold_(resample_threshold)
    , motion_noise_(motion_noise)
    , initialized_(false)
    , rng_(std::make_unique<ThreadSafeRNG>()) {

    if (num_particles <= 0) {
        throw std::invalid_argument("Number of particles must be positive");
    }
    if (resample_threshold <= 0 || resample_threshold > 1) {
        throw std::invalid_argument("Resample threshold must be in (0,1]");
    }
    if (motion_noise.minCoeff() < 0) {
        throw std::invalid_argument("Motion noise cannot be negative");
    }

    particles_.reserve(num_particles);
}

ParticleFilter::~ParticleFilter() = default;

void ParticleFilter::init(const Pose2D& initial_pose, const Eigen::Vector3d& noise) {
    try {
        // 验证噪声参数
        if (noise.minCoeff() < 0) {
            throw std::invalid_argument("Initialization noise cannot be negative");
        }

        particles_.clear();
        particles_.reserve(num_particles_);

        // 创建正态分布生成器
        std::normal_distribution<double> noise_x(0, noise(0));
        std::normal_distribution<double> noise_y(0, noise(1));
        std::normal_distribution<double> noise_yaw(0, noise(2));

        // 生成初始粒子
        for (int i = 0; i < num_particles_; ++i) {
            WeightedPose particle;
            particle.pose.x = initial_pose.x + rng_->generate(noise_x);
            particle.pose.y = initial_pose.y + rng_->generate(noise_y);
            particle.pose.yaw = normalizeAngle(initial_pose.yaw + rng_->generate(noise_yaw));
            particle.weight = 1.0 / num_particles_;
            particles_.push_back(particle);
        }

        initialized_ = true;
        ROS_INFO("Particle filter initialized with %d particles", num_particles_);
    }
    catch (const std::exception& e) {
        ROS_ERROR("Failed to initialize particle filter: %s", e.what());
        initialized_ = false;
        throw;
    }
}

void ParticleFilter::predict(const Pose2D& odom_delta) {
    if (!initialized_) {
        ROS_WARN_THROTTLE(1.0, "Particle filter not initialized");
        return;
    }

    // 创建运动噪声分布
    std::normal_distribution<double> noise_x(0, motion_noise_(0));
    std::normal_distribution<double> noise_y(0, motion_noise_(1));
    std::normal_distribution<double> noise_yaw(0, motion_noise_(2));

    #pragma omp parallel for
    for (size_t i = 0; i < particles_.size(); ++i) {
        // 添加噪声到里程计增量
        double noisy_dx = odom_delta.x + rng_->generate(noise_x);
        double noisy_dy = odom_delta.y + rng_->generate(noise_y);
        double noisy_dyaw = odom_delta.yaw + rng_->generate(noise_yaw);

        // 根据运动模型更新粒子位置
        double cos_yaw = cos(particles_[i].pose.yaw);
        double sin_yaw = sin(particles_[i].pose.yaw);

        particles_[i].pose.x += noisy_dx * cos_yaw - noisy_dy * sin_yaw;
        particles_[i].pose.y += noisy_dx * sin_yaw + noisy_dy * cos_yaw;
        particles_[i].pose.yaw = normalizeAngle(particles_[i].pose.yaw + noisy_dyaw);
    }
}

void ParticleFilter::update(const std::vector<float>& scan, const cv::Mat& distance_field) {
    if (!initialized_) {
        ROS_WARN_THROTTLE(1.0, "Particle filter not initialized");
        return;
    }

    double max_weight = -std::numeric_limits<double>::infinity();

    // 计算每个粒子的权重
    #pragma omp parallel for reduction(max:max_weight)
    for (size_t i = 0; i < particles_.size(); ++i) {
        // 计算观测似然
        double likelihood = computeScanLikelihood(particles_[i].pose, scan, distance_field);
        particles_[i].weight = likelihood;
        max_weight = std::max(max_weight, likelihood);
    }

    // 避免数值问题
    if (std::isinf(max_weight) || max_weight <= 0) {
        ROS_WARN("Invalid particle weights detected, resetting to uniform weights");
        for (auto& particle : particles_) {
            particle.weight = 1.0 / num_particles_;
        }
        return;
    }

    // 归一化权重
    normalizeWeights();

    // 计算有效粒子数
    double neff = computeEffectiveParticles();
    ROS_DEBUG("Effective particle ratio: %.2f", neff / num_particles_);

    // 如果有效粒子数太少，进行重采样
    if (neff < num_particles_ * resample_threshold_) {
        resample();
    }
}

void ParticleFilter::resample() {
    std::vector<WeightedPose> new_particles;
    new_particles.reserve(particles_.size());

    // 系统重采样算法
    std::uniform_real_distribution<double> dist(0, 1.0 / num_particles_);
    double r = rng_->generate(dist);
    double c = particles_[0].weight;
    size_t i = 0;

    for (size_t m = 0; m < particles_.size(); ++m) {
        double U = r + m * (1.0 / num_particles_);
        while (U > c && i < particles_.size() - 1) {
            i++;
            c += particles_[i].weight;
        }
        new_particles.push_back(particles_[i]);
        new_particles.back().weight = 1.0 / num_particles_;
    }

    particles_ = std::move(new_particles);
}

Pose2D ParticleFilter::getEstimatedPose() const {
    if (!initialized_ || particles_.empty()) {
        ROS_WARN("Cannot estimate pose: particle filter not initialized or empty");
        return Pose2D();
    }

    Pose2D mean_pose;
    double weight_sum = 0;
    double cos_sum = 0, sin_sum = 0;

    for (const auto& particle : particles_) {
        mean_pose.x += particle.weight * particle.pose.x;
        mean_pose.y += particle.weight * particle.pose.y;
        cos_sum += particle.weight * cos(particle.pose.yaw);
        sin_sum += particle.weight * sin(particle.pose.yaw);
        weight_sum += particle.weight;
    }

    if (weight_sum > 0) {
        mean_pose.x /= weight_sum;
        mean_pose.y /= weight_sum;
        mean_pose.yaw = atan2(sin_sum, cos_sum);
    }

    return mean_pose;
}

void ParticleFilter::addRandomParticles(int count) {
    if (count <= 0 || particles_.empty()) {
        return;
    }

    // 基于当前最优粒子创建随机粒子
    const WeightedPose& best_particle =
        *std::max_element(particles_.begin(), particles_.end(),
            [](const WeightedPose& a, const WeightedPose& b) {
                return a.weight < b.weight;
            });

    // 创建较大范围的高斯分布
    std::normal_distribution<double> noise_x(0, motion_noise_(0) * 3);
    std::normal_distribution<double> noise_y(0, motion_noise_(1) * 3);
    std::normal_distribution<double> noise_yaw(0, motion_noise_(2) * 3);

    for (int i = 0; i < count; ++i) {
        WeightedPose particle;
        particle.pose.x = best_particle.pose.x + rng_->generate(noise_x);
        particle.pose.y = best_particle.pose.y + rng_->generate(noise_y);
        particle.pose.yaw = normalizeAngle(
            best_particle.pose.yaw + rng_->generate(noise_yaw));
        particle.weight = 1.0 / (particles_.size() + count);
        particles_.push_back(particle);
    }

    // 重新归一化权重
    normalizeWeights();
}

double ParticleFilter::normalizeAngle(double angle) const {
    return utils::normalizeAngle(angle);
}

double ParticleFilter::computeEffectiveParticles() const {
    double sum_squared_weights = 0.0;
    for (const auto& particle : particles_) {
        sum_squared_weights += particle.weight * particle.weight;
    }
    return 1.0 / sum_squared_weights;
}

void ParticleFilter::normalizeWeights() {
    double sum_weights = 0.0;
    for (const auto& particle : particles_) {
        sum_weights += particle.weight;
    }

    if (sum_weights > 0) {
        for (auto& particle : particles_) {
            particle.weight /= sum_weights;
        }
    } else {
        double uniform_weight = 1.0 / particles_.size();
        for (auto& particle : particles_) {
            particle.weight = uniform_weight;
        }
    }
}

bool ParticleFilter::detectLocalizationFailure() const {
    // 检查有效粒子数是否过低
    double neff = computeEffectiveParticles();
    if (neff < num_particles_ * 0.1) {  // 阈值可配置
        return true;
    }

    // 检查粒子分布是否过于分散
    if (getParticleDiversity() > 2.0) {  // 阈值可配置
        return true;
    }

    return false;
}

double ParticleFilter::getParticleDiversity() const {
    if (particles_.empty()) {
        return 0.0;
    }

    // 计算粒子位置的标准差
    Pose2D mean = getEstimatedPose();
    double var_x = 0.0, var_y = 0.0;

    for (const auto& particle : particles_) {
        double dx = particle.pose.x - mean.x;
        double dy = particle.pose.y - mean.y;
        var_x += particle.weight * dx * dx;
        var_y += particle.weight * dy * dy;
    }

    // 返回位置标准差的平均值
    return std::sqrt((var_x + var_y) / 2.0);
}

double ParticleFilter::computeScanLikelihood(const Pose2D& pose,
                               const std::vector<float>& scan,
                               const cv::Mat& distance_field) const {
    if (scan.empty() || distance_field.empty()) {
        return 0.0;
    }

    double total_likelihood = 0.0;
    int valid_points = 0;
    const double angle_increment = M_PI / 180.0;  // 1度，应从参数读取
    const double max_range = 20.0;  // 应从参数读取
    const double resolution = 0.05;  // 地图分辨率，应从参数读取

    for (size_t i = 0; i < scan.size(); ++i) {
        const float range = scan[i];
        if (!std::isfinite(range) || range > max_range) {
            continue;
        }

        // 计算激光点在地图坐标系下的位置
        double angle = pose.yaw + i * angle_increment - M_PI;
        double px = pose.x + range * cos(angle);
        double py = pose.y + range * sin(angle);

        // 将世界坐标转换为图像坐标
        int map_x = static_cast<int>(px / resolution);
        int map_y = static_cast<int>(py / resolution);

        // 检查点是否在地图范围内，留出1像素边界用于插值
        if (map_x >= 1 && map_x < distance_field.cols - 2 &&
            map_y >= 1 && map_y < distance_field.rows - 2) {

            // 计算插值系数
            double x_ratio = (px / resolution) - map_x;
            double y_ratio = (py / resolution) - map_y;

            // 双线性插值
            double d00 = distance_field.at<float>(map_y, map_x);
            double d01 = distance_field.at<float>(map_y, map_x + 1);
            double d10 = distance_field.at<float>(map_y + 1, map_x);
            double d11 = distance_field.at<float>(map_y + 1, map_x + 1);

            double dist = d00 * (1 - x_ratio) * (1 - y_ratio) +
                         d01 * x_ratio * (1 - y_ratio) +
                         d10 * (1 - x_ratio) * y_ratio +
                         d11 * x_ratio * y_ratio;

            // 计算似然
            const double sigma = 0.2;
            total_likelihood += exp(-dist * dist / (2 * sigma * sigma));
            valid_points++;
        }
    }

    // 返回平均似然值
    return valid_points > 0 ? total_likelihood / valid_points : 0.0;
}

ParticleFilterStats ParticleFilter::getStats() const {
    ParticleFilterStats stats;
    stats.effective_particles = computeEffectiveParticles();

    auto minmax = std::minmax_element(particles_.begin(), particles_.end(),
        [](const WeightedPose& a, const WeightedPose& b) {
            return a.weight < b.weight;
        });

    stats.min_weight = minmax.first->weight;
    stats.max_weight = minmax.second->weight;

    // 计算权重方差
    double mean_weight = 1.0 / num_particles_;
    double sum_squared_diff = 0.0;
    for (const auto& particle : particles_) {
        double diff = particle.weight - mean_weight;
        sum_squared_diff += diff * diff;
    }
    stats.weight_variance = sum_squared_diff / num_particles_;

    return stats;
}

} // namespace amr_lidar_localization