#pragma once

#include <vector>
#include <random>
#include <memory>
#include <mutex>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include "types.h"
#include "utils.h"

namespace amr_lidar_localization {

class ParticleFilter {
public:
    ParticleFilter(int num_particles,
                  double resample_threshold,
                  const Eigen::Vector3d& motion_noise);
    ~ParticleFilter();

    // 主要功能函数
    void init(const Pose2D& initial_pose, const Eigen::Vector3d& noise);
    void predict(const Pose2D& odom_delta);
    void update(const std::vector<float>& scan, const cv::Mat& distance_field);
    void resample();

    // 获取状态信息
    Pose2D getEstimatedPose() const;
    const std::vector<WeightedPose>& getParticles() const { return particles_; }
    ParticleFilterStats getStats() const;
    bool isInitialized() const { return initialized_; }

    // 恢复机制
    void addRandomParticles(int count);

private:
    // 线程安全的随机数生成器
    class ThreadSafeRNG {
    private:
        mutable std::mutex mutex_;
        mutable std::mt19937 rng_;

    public:
        ThreadSafeRNG() : rng_(std::random_device()()) {}

        template<typename Distribution>
        typename Distribution::result_type generate(Distribution& dist) const {
            std::lock_guard<std::mutex> lock(mutex_);
            return dist(rng_);
        }
    };

    const int num_particles_;              // 粒子数量
    const double resample_threshold_;      // 重采样阈值
    const Eigen::Vector3d motion_noise_;   // 运动噪声

    bool initialized_{false};              // 初始化标志
    std::vector<WeightedPose> particles_; // 粒子集合
    std::unique_ptr<ThreadSafeRNG> rng_;  // 随机数生成器

    // 辅助函数
    double normalizeAngle(double angle) const;                // 角度归一化
    double computeEffectiveParticles() const;                // 计算有效粒子数
    void normalizeWeights();                                 // 权重归一化
    bool detectLocalizationFailure() const;                  // 检测定位失败
    double getParticleDiversity() const;                     // 获取粒子多样性
    double computeScanLikelihood(const Pose2D& pose,         // 计算扫描似然
                               const std::vector<float>& scan,
                               const cv::Mat& distance_field) const;
};

} // namespace amr_lidar_localization