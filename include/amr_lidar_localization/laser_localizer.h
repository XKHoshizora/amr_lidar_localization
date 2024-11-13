#pragma once

#include <mutex>
#include <memory>
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <diagnostic_updater/diagnostic_updater.h>
#include "map_processor.h"
#include "particle_filter.h"

namespace amr_lidar_localization {

class LaserLocalizer {
public:
    LaserLocalizer(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~LaserLocalizer() = default;

    bool start();  // 启动定位
    void stop();   // 停止定位
    bool isRunning() const { return running_; }

private:
    // 互斥锁保护
    std::mutex map_mutex_;
    std::mutex pose_mutex_;

    // ROS handles
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // 订阅者和发布者
    ros::Subscriber map_sub_;
    ros::Subscriber scan_sub_;
    ros::Subscriber initial_pose_sub_;
    ros::Publisher particle_pub_;
    ros::Publisher likelihood_pub_;
    ros::Publisher pose_pub_;
    ros::Publisher diagnostics_pub_;

    // TF相关
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // 诊断更新器
    diagnostic_updater::Updater diagnostics_;

    // 定时器
    ros::Timer diagnostics_timer_;

    // 核心组件
    std::unique_ptr<ParticleFilter> particle_filter_;
    std::unique_ptr<MapProcessor> map_processor_;

    // 参数结构
    struct Parameters {
        // 框架ID
        std::string base_frame{"base_footprint"};
        std::string odom_frame{"odom"};
        std::string map_frame{"map"};
        std::string laser_frame{"laser"};

        // 时间和更新阈值
        double transform_tolerance{0.1};
        double min_update_distance{0.1};
        double min_update_angle{0.1};

        // 激光雷达参数
        double max_range{20.0};
        double min_range{0.1};
        double angle_increment{0.25};
        double angle_min{-M_PI};
        double angle_max{M_PI};

        // 激光雷达扫描匹配参数
        double scan_sigma{0.2};           // 扫描匹配的标准差
        double scan_resolution{0.05};    // 扫描匹配的分辨率
        double scan_max_range{20.0};     // 扫描匹配的最大范围
        double scan_angle_increment{0.25}; // 扫描匹配的角度增量

        // 粒子滤波参数
        int num_particles{1000};
        double resample_threshold{0.5};
        Eigen::Vector3d initial_noise{0.5, 0.5, 0.2};
        Eigen::Vector3d motion_noise{0.1, 0.1, 0.05};

        // 发布选项
        bool publish_particles{true};
        bool publish_likelihood{true};
        double publish_rate{10.0};
    } params_;

    // 状态变量
    struct State {
        bool initialized{false};
        bool map_received{false};
        Pose2D last_pose;
        ros::Time last_update_time;

        // 性能统计
        struct Statistics {
            double update_time{0};              // 更新时间
            int valid_scans{0};                 // 有效激光点数
            int total_scans{0};                 // 总激光点数
            double effective_particles_ratio{0}; // 有效粒子比例
        } stats;
    } state_;

    // 运行状态监控
    struct HealthMonitor {
        int consecutive_failures{0};
        ros::Time last_successful_update;
        bool is_degraded{false};

        static constexpr int MAX_FAILURES = 3;
        static constexpr double RECOVERY_TIMEOUT = 5.0;
    } health_;

    // 初始化函数
    void initializeParameters();
    void initializeSubscribers();
    void initializePublishers();
    bool initializeTransforms();

    // 回调函数
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
    void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg);
    void diagnosticsCallback(const ros::TimerEvent& event);

    // 辅助函数
    bool needsUpdate(const Pose2D& current_pose) const;
    bool getTransform(const std::string& target_frame,
                     const std::string& source_frame,
                     const ros::Time& time,
                     geometry_msgs::TransformStamped& transform) const;
    void publishResults(const Pose2D& pose, const ros::Time& timestamp);
    bool validatePose(const Pose2D& pose) const;
    void updateDiagnostics(diagnostic_updater::DiagnosticStatusWrapper& stat);
    void publishParticles(const std::vector<WeightedPose>& particles);
    void publishPose(const Pose2D& pose, const ros::Time& timestamp);

    // 状态监控和恢复
    void checkHealth();
    void enterDegradedMode();
    void exitDegradedMode();
    void reset();

    // 状态标志
    bool running_{false};

    // 坐标变换辅助函数
    Pose2D transformToPose2D(const geometry_msgs::TransformStamped& transform) const;
    Pose2D predictPose(const ros::Time& target_time);
    geometry_msgs::Transform pose2DToTransform(const Pose2D& pose) const;
};

} // namespace amr_lidar_localization