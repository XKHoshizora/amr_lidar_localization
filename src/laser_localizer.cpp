// === src/laser_localizer.cpp ===
#include "amr_lidar_localization/laser_localizer.h"
#include <tf2/utils.h>

namespace amr_lidar_localization {

using namespace utils;

LaserLocalizer::LaserLocalizer(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh)
    , pnh_(pnh)
    , diagnostics_(nh, pnh)
{
    try {
        // 初始化组件
        initializeParameters();

        // 初始化TF组件
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>();

        // 初始化核心组件
        particle_filter_ = std::make_unique<ParticleFilter>(
            params_.num_particles,
            params_.resample_threshold,
            params_.motion_noise);

        map_processor_ = std::make_unique<MapProcessor>();

        // 初始化ROS接口
        initializeSubscribers();
        initializePublishers();

        // 初始化诊断
        diagnostics_.setHardwareID("laser_localizer");
        diagnostics_.add("localizer_status", this, &LaserLocalizer::updateDiagnostics);

        // 启动诊断定时器
        diagnostics_timer_ = nh_.createTimer(
            ros::Duration(1.0),
            &LaserLocalizer::diagnosticsCallback,
            this);

        if (!initializeTransforms()) {
            ROS_ERROR("Failed to initialize transforms");
            throw std::runtime_error("Transform initialization failed");
        }

        ROS_INFO("Laser localizer initialized successfully");
    }
    catch (const std::exception& e) {
        ROS_ERROR("Failed to initialize laser localizer: %s", e.what());
        throw;
    }
}

void LaserLocalizer::initializeParameters() {
    // 加载框架ID参数
    pnh_.param<std::string>("base_frame", params_.base_frame, "base_footprint");
    pnh_.param<std::string>("odom_frame", params_.odom_frame, "odom");
    pnh_.param<std::string>("map_frame", params_.map_frame, "map");
    pnh_.param<std::string>("laser_frame", params_.laser_frame, "laser");

    // 加载时间和更新阈值参数
    pnh_.param<double>("transform_tolerance", params_.transform_tolerance, 0.1);
    pnh_.param<double>("update_min_d", params_.min_update_distance, 0.1);
    pnh_.param<double>("update_min_a", params_.min_update_angle, 0.1);

    // 加载激光雷达参数
    pnh_.param<double>("max_laser_range", params_.max_range, 20.0);
    pnh_.param<double>("min_laser_range", params_.min_range, 0.1);
    pnh_.param<double>("angle_increment", params_.angle_increment, 0.25);
    pnh_.param<double>("angle_min", params_.angle_min, -M_PI);
    pnh_.param<double>("angle_max", params_.angle_max, M_PI);

    // 加载粒子滤波参数
    pnh_.param<int>("num_particles", params_.num_particles, 1000);
    pnh_.param<double>("resample_threshold", params_.resample_threshold, 0.5);

    // 加载噪声参数
    double temp;
    pnh_.param<double>("initial_noise/x", temp, 0.5);
    params_.initial_noise(0) = temp;
    pnh_.param<double>("initial_noise/y", temp, 0.5);
    params_.initial_noise(1) = temp;
    pnh_.param<double>("initial_noise/yaw", temp, 0.2);
    params_.initial_noise(2) = temp;

    pnh_.param<double>("motion_noise/x", temp, 0.1);
    params_.motion_noise(0) = temp;
    pnh_.param<double>("motion_noise/y", temp, 0.1);
    params_.motion_noise(1) = temp;
    pnh_.param<double>("motion_noise/yaw", temp, 0.05);
    params_.motion_noise(2) = temp;

    // 加载发布选项
    pnh_.param<bool>("publish_particles", params_.publish_particles, true);
    pnh_.param<bool>("publish_likelihood", params_.publish_likelihood, true);
    pnh_.param<double>("publish_rate", params_.publish_rate, 10.0);

    // 加载扫描匹配参数
    pnh_.param<double>("particle_filter/scan_matching/sigma", params_.scan_sigma, 0.2);
    pnh_.param<double>("particle_filter/scan_matching/resolution", params_.scan_resolution, 0.05);
    pnh_.param<double>("particle_filter/scan_matching/max_range", params_.scan_max_range, 20.0);
    pnh_.param<double>("particle_filter/scan_matching/angle_increment",
                      params_.scan_angle_increment, 0.25);

    // 加载OpenMP参数
    bool enable_openmp = true;
    int thread_count = 4;
    pnh_.param<bool>("parallel/enable_openmp", enable_openmp, true);
    pnh_.param<int>("parallel/thread_count", thread_count, 4);

    if (enable_openmp) {
        #ifdef _OPENMP
            if (thread_count > 0) {
                omp_set_num_threads(thread_count);
            }
            ROS_INFO("OpenMP enabled with %d threads", omp_get_max_threads());
        #else
            ROS_WARN("OpenMP support not available");
        #endif
    }
}

void LaserLocalizer::initializeSubscribers() {
    map_sub_ = nh_.subscribe("map", 1, &LaserLocalizer::mapCallback, this);
    scan_sub_ = nh_.subscribe("scan", 1, &LaserLocalizer::scanCallback, this);
    initial_pose_sub_ = nh_.subscribe("initialpose", 1,
                                    &LaserLocalizer::initialPoseCallback, this);
}

void LaserLocalizer::initializePublishers() {
    particle_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("particles", 1);
    likelihood_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("likelihood_field", 1);
    pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose", 1);
    diagnostics_pub_ = nh_.advertise<diagnostic_msgs::DiagnosticArray>("diagnostics", 1);
}

bool LaserLocalizer::initializeTransforms() {
    try {
        // 等待必要的TF转换可用
        if (!tf_buffer_->canTransform(params_.base_frame, params_.laser_frame,
                                    ros::Time(0), ros::Duration(5.0))) {
            ROS_ERROR("Cannot transform from %s to %s",
                     params_.base_frame.c_str(), params_.laser_frame.c_str());
            return false;
        }

        return true;
    }
    catch (const tf2::TransformException& ex) {
        ROS_ERROR("Transform initialization failed: %s", ex.what());
        return false;
    }
}

void LaserLocalizer::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    try {
        map_processor_->updateMap(*msg);
        state_.map_received = true;

        if (params_.publish_likelihood) {
            nav_msgs::OccupancyGrid likelihood_map;
            likelihood_map.header = msg->header;
            likelihood_map.info = msg->info;

            const cv::Mat& distance_field = map_processor_->getDistanceField();
            likelihood_map.data.resize(distance_field.rows * distance_field.cols);

            for (int i = 0; i < distance_field.rows; ++i) {
                for (int j = 0; j < distance_field.cols; ++j) {
                    float value = distance_field.at<float>(i, j);
                    likelihood_map.data[i * distance_field.cols + j] =
                        static_cast<int8_t>(value * 100);
                }
            }

            likelihood_pub_.publish(likelihood_map);
        }

        ROS_INFO("Map updated: %dx%d cells", msg->info.width, msg->info.height);
    }
    catch (const std::exception& e) {
        ROS_ERROR("Error processing map: %s", e.what());
    }
}

void LaserLocalizer::scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    if (!state_.map_received) {
        ROS_WARN_THROTTLE(1.0, "No map received yet");
        return;
    }

    std::lock_guard<std::mutex> map_lock(map_mutex_);
    std::lock_guard<std::mutex> pose_lock(pose_mutex_);

    try {
        ros::Time start_time = ros::Time::now();

        // 获取激光雷达到基座的转换
        geometry_msgs::TransformStamped laser_to_base;
        if (!getTransform(params_.base_frame, params_.laser_frame, msg->header.stamp,
                         laser_to_base)) {
            return;
        }

        // 获取里程计增量
        geometry_msgs::TransformStamped odom_tf;
        if (!getTransform(params_.odom_frame, params_.base_frame, msg->header.stamp,
                         odom_tf)) {
            return;
        }

        // 转换为2D位姿增量
        Pose2D current_odom = transformToPose2D(odom_tf);

        // 检查是否需要更新
        if (!needsUpdate(current_odom)) {
            return;
        }

        // 预处理激光数据
        std::vector<float> valid_ranges;
        valid_ranges.reserve(msg->ranges.size());

        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            const float range = msg->ranges[i];
            if (std::isfinite(range) && range >= params_.min_range &&
                range <= params_.max_range) {
                valid_ranges.push_back(range);
                state_.stats.valid_scans++;
            }
        }
        state_.stats.total_scans += msg->ranges.size();

        if (valid_ranges.empty()) {
            ROS_WARN_THROTTLE(1.0, "No valid laser measurements");
            return;
        }

        // 更新粒子滤波器
        particle_filter_->predict(current_odom);
        particle_filter_->update(valid_ranges, map_processor_->getDistanceField());

        // 获取估计位姿
        Pose2D estimated_pose = particle_filter_->getEstimatedPose();

        // 验证位姿有效性
        if (!validatePose(estimated_pose)) {
            ROS_WARN("Invalid pose estimate, skipping update");
            return;
        }

        // 发布结果
        publishResults(estimated_pose, msg->header.stamp);

        // 更新状态
        state_.last_pose = estimated_pose;
        state_.last_update_time = msg->header.stamp;

        // 更新统计信息
        auto pf_stats = particle_filter_->getStats();
        state_.stats.effective_particles_ratio =
            pf_stats.effective_particles / params_.num_particles;
        state_.stats.update_time = (ros::Time::now() - start_time).toSec();

        ROS_DEBUG("Update completed in %.3f seconds, effective particles ratio: %.2f",
                 state_.stats.update_time, state_.stats.effective_particles_ratio);

        // 更新成功
        health_.consecutive_failures = 0;
        health_.last_successful_update = msg->header.stamp;
        if (health_.is_degraded) {
            exitDegradedMode();
        }
    }
    catch (const std::exception& e) {
        health_.consecutive_failures++;
        checkHealth();
        ROS_ERROR_STREAM("Scan processing failed: " << e.what());
    }
}

void LaserLocalizer::initialPoseCallback(
    const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {

    std::lock_guard<std::mutex> pose_lock(pose_mutex_);

    try {
        // 转换初始位姿到2D
        Pose2D initial_pose;
        initial_pose.x = msg->pose.pose.position.x;
        initial_pose.y = msg->pose.pose.position.y;

        // 从四元数转换为yaw角
        tf2::Quaternion q;
        tf2::fromMsg(msg->pose.pose.orientation, q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        initial_pose.yaw = yaw;

        // 初始化粒子滤波器
        particle_filter_->init(initial_pose, params_.initial_noise);

        // 更新状态
        state_.initialized = true;
        state_.last_pose = initial_pose;
        state_.last_update_time = msg->header.stamp;

        ROS_INFO("Initialized pose at (%.2f, %.2f, %.2f)",
                initial_pose.x, initial_pose.y, initial_pose.yaw);
    }
    catch (const std::exception& e) {
        ROS_ERROR("Error in initial pose processing: %s", e.what());
    }
}

bool LaserLocalizer::needsUpdate(const Pose2D& current_pose) const {
    if (!state_.initialized) {
        return false;
    }

    if ((ros::Time::now() - state_.last_update_time).toSec() <
        params_.transform_tolerance) {
        return false;
    }

    double dx = current_pose.x - state_.last_pose.x;
    double dy = current_pose.y - state_.last_pose.y;
    double dist = std::hypot(dx, dy);

    double angle_diff = std::abs(normalizeAngle(
        current_pose.yaw - state_.last_pose.yaw));

    return dist > params_.min_update_distance ||
           angle_diff > params_.min_update_angle;
}

bool LaserLocalizer::getTransform(const std::string& target_frame,
                                const std::string& source_frame,
                                const ros::Time& time,
                                geometry_msgs::TransformStamped& transform) const {
    try {
        transform = tf_buffer_->lookupTransform(
            target_frame, source_frame, time,
            ros::Duration(params_.transform_tolerance));
        return true;
    }
    catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(1.0, "Failed to get transform %s -> %s: %s",
                         source_frame.c_str(), target_frame.c_str(), ex.what());
        return false;
    }
}

void LaserLocalizer::publishResults(const Pose2D& pose, const ros::Time& timestamp) {
    if (params_.publish_particles) {
        publishParticles(particle_filter_->getParticles());
    }

    publishPose(pose, timestamp);

    // 发布TF转换
    geometry_msgs::TransformStamped transform;
    transform.header.stamp = timestamp;
    transform.header.frame_id = params_.map_frame;
    transform.child_frame_id = params_.odom_frame;
    transform.transform = pose2DToTransform(pose);

    tf_broadcaster_->sendTransform(transform);
}

void LaserLocalizer::publishParticles(const std::vector<amr_lidar_localization::WeightedPose>& particles) {
    visualization_msgs::MarkerArray marker_array;

    // 创建粒子点标记
    visualization_msgs::Marker particle_points;
    particle_points.header.frame_id = params_.map_frame;
    particle_points.header.stamp = ros::Time::now();
    particle_points.ns = "particles";
    particle_points.id = 0;
    particle_points.type = visualization_msgs::Marker::POINTS;
    particle_points.action = visualization_msgs::Marker::ADD;
    particle_points.scale.x = 0.05;
    particle_points.scale.y = 0.05;

    // 创建方向标记
    visualization_msgs::Marker particle_arrows;
    particle_arrows.header = particle_points.header;
    particle_arrows.ns = "particle_directions";
    particle_arrows.id = 1;
    particle_arrows.type = visualization_msgs::Marker::LINE_LIST;
    particle_arrows.action = visualization_msgs::Marker::ADD;
    particle_arrows.scale.x = 0.02;

    for (const auto& particle : particles) {
        // 添加粒子位置
        geometry_msgs::Point p;
        p.x = particle.pose.x;
        p.y = particle.pose.y;
        p.z = 0.0;

        // 设置颜色（根据权重）
        std_msgs::ColorRGBA color;
        color.r = particle.weight;
        color.b = 1.0 - particle.weight;
        color.a = 0.5;

        particle_points.points.push_back(p);
        particle_points.colors.push_back(color);

        // 添加方向线
        geometry_msgs::Point p2;
        p2.x = p.x + 0.2 * cos(particle.pose.yaw);
        p2.y = p.y + 0.2 * sin(particle.pose.yaw);
        p2.z = 0.0;

        particle_arrows.points.push_back(p);
        particle_arrows.points.push_back(p2);
    }

    marker_array.markers.push_back(particle_points);
    marker_array.markers.push_back(particle_arrows);
    particle_pub_.publish(marker_array);
}

void LaserLocalizer::publishPose(const Pose2D& pose, const ros::Time& timestamp) {
    geometry_msgs::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = timestamp;
    pose_msg.header.frame_id = params_.map_frame;

    // 设置位置和方向
    pose_msg.pose.pose.position.x = pose.x;
    pose_msg.pose.pose.position.y = pose.y;
    pose_msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, pose.yaw);
    pose_msg.pose.pose.orientation = tf2::toMsg(q);

    // 计算协方差矩阵
    // 使用粒子分布来估计位姿不确定性
    const std::vector<amr_lidar_localization::WeightedPose>& particles = particle_filter_->getParticles();
    Eigen::Matrix3d covariance = utils::computeCovarianceMatrix(particles, pose);

    // 将3x3协方差矩阵转换为ROS消息中的6x6协方差矩阵
    // ROS中的协方差矩阵顺序是: [x, y, z, rot_x, rot_y, rot_z]
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (i < 2 && j < 2) {
                // x, y位置的协方差
                pose_msg.pose.covariance[i * 6 + j] = covariance(i, j);
            } else if (i == 5 && j == 5) {
                // yaw角度的协方差
                pose_msg.pose.covariance[i * 6 + j] = covariance(2, 2);
            } else if (i == j) {
                // z, roll, pitch的协方差设为小值
                pose_msg.pose.covariance[i * 6 + j] = 0.01;
            } else {
                // 其他交叉项设为0
                pose_msg.pose.covariance[i * 6 + j] = 0.0;
            }
        }
    }

    pose_pub_.publish(pose_msg);

    // 可选：发布协方差椭圆可视化
    if (params_.publish_particles) {
        visualization_msgs::Marker covariance_marker;
        covariance_marker.header = pose_msg.header;
        covariance_marker.ns = "pose_covariance";
        covariance_marker.id = 0;
        covariance_marker.type = visualization_msgs::Marker::SPHERE;
        covariance_marker.action = visualization_msgs::Marker::ADD;

        // 设置椭圆中心
        covariance_marker.pose = pose_msg.pose.pose;

        // 计算椭圆尺寸 (使用2σ置信区间)
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(
            covariance.block<2,2>(0,0));
        Eigen::Vector2d eigen_values = eigen_solver.eigenvalues();

        // 设置椭圆尺寸
        covariance_marker.scale.x = 2 * std::sqrt(std::abs(eigen_values(0)));
        covariance_marker.scale.y = 2 * std::sqrt(std::abs(eigen_values(1)));
        covariance_marker.scale.z = 0.01;  // 很薄的椭圆

        // 设置颜色和透明度
        covariance_marker.color.r = 0.0;
        covariance_marker.color.g = 0.7;
        covariance_marker.color.b = 0.7;
        covariance_marker.color.a = 0.3;

        // 发布协方差可视化
        particle_pub_.publish(covariance_marker);
    }
}

void LaserLocalizer::diagnosticsCallback(const ros::TimerEvent& event) {
    diagnostics_.update();
}

void LaserLocalizer::updateDiagnostics(diagnostic_updater::DiagnosticStatusWrapper& stat) {
    if (!state_.initialized) {
        stat.summary(diagnostic_msgs::DiagnosticStatus::WARN, "Not initialized");
        return;
    }

    if (!state_.map_received) {
        stat.summary(diagnostic_msgs::DiagnosticStatus::WARN, "No map received");
        return;
    }

    if (state_.stats.update_time > params_.transform_tolerance) {
        stat.summary(diagnostic_msgs::DiagnosticStatus::WARN, "Update too slow");
    } else {
        stat.summary(diagnostic_msgs::DiagnosticStatus::OK, "Running normally");
    }

    stat.add("Update time (s)", state_.stats.update_time);
    stat.add("Valid scans ratio",
             static_cast<double>(state_.stats.valid_scans) / state_.stats.total_scans);
    stat.add("Effective particles ratio", state_.stats.effective_particles_ratio);
}

Pose2D LaserLocalizer::transformToPose2D(
    const geometry_msgs::TransformStamped& transform) const {

    amr_lidar_localization::Pose2D pose;
    pose.x = transform.transform.translation.x;
    pose.y = transform.transform.translation.y;

    tf2::Quaternion q;
    tf2::fromMsg(transform.transform.rotation, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    pose.yaw = yaw;

    return pose;
}

geometry_msgs::Transform LaserLocalizer::pose2DToTransform(const Pose2D& pose) const {
    geometry_msgs::Transform transform;

    transform.translation.x = pose.x;
    transform.translation.y = pose.y;
    transform.translation.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, pose.yaw);
    transform.rotation = tf2::toMsg(q);

    return transform;
}

bool LaserLocalizer::validatePose(const Pose2D& pose) const {
    // 检查位置是否在地图范围内
    if (!map_processor_->isInside(pose.x, pose.y)) {
        return false;
    }

    // 检查角度是否正常
    if (std::isnan(pose.yaw) || std::abs(pose.yaw) > M_PI) {
        return false;
    }

    // 检查与上一次位姿的差异是否合理
    if (state_.initialized) {
        double dx = pose.x - state_.last_pose.x;
        double dy = pose.y - state_.last_pose.y;
        double dist = std::hypot(dx, dy);

        if (dist > params_.max_range) {
            return false;
        }

        double angle_diff = std::abs(normalizeAngle(pose.yaw - state_.last_pose.yaw));
        if (angle_diff > M_PI_2) {
            return false;
        }
    }

    return true;
}

void LaserLocalizer::checkHealth() {
    if (health_.consecutive_failures >= HealthMonitor::MAX_FAILURES) {
        enterDegradedMode();
    }

    if (health_.is_degraded) {
        double time_since_last_success =
            (ros::Time::now() - health_.last_successful_update).toSec();
        if (time_since_last_success > HealthMonitor::RECOVERY_TIMEOUT) {
            reset();
        }
    }
}

void LaserLocalizer::enterDegradedMode() {
    if (!health_.is_degraded) {
        health_.is_degraded = true;
        ROS_WARN("Entering degraded mode due to consecutive failures");
        // 增加随机粒子以提高恢复概率
        particle_filter_->addRandomParticles(200);
    }
}

void LaserLocalizer::exitDegradedMode() {
    if (health_.is_degraded) {
        health_.is_degraded = false;
        health_.consecutive_failures = 0;
        ROS_INFO("Exiting degraded mode");
    }
}

void LaserLocalizer::reset() {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    std::lock_guard<std::mutex> pose_lock(pose_mutex_);

    state_.initialized = false;
    health_.consecutive_failures = 0;
    health_.is_degraded = false;

    // 重新初始化粒子滤波器
    if (state_.map_received) {
        Pose2D initial_pose;  // 使用上一次的位姿或默认位姿
        particle_filter_->init(initial_pose, params_.initial_noise);
    }

    ROS_INFO("Localizer reset completed");
}

Pose2D LaserLocalizer::predictPose(const ros::Time& target_time) {
    if (!state_.initialized) {
        return Pose2D();
    }

    try {
        // 获取当前到目标时间的里程计增量
        geometry_msgs::TransformStamped transform;
        if (!getTransform(params_.odom_frame, params_.base_frame,
                         target_time, transform)) {
            return state_.last_pose;
        }

        Pose2D odom_delta = transformToPose2D(transform);
        Pose2D predicted_pose = state_.last_pose;

        // 应用运动模型
        predicted_pose.x += odom_delta.x * cos(predicted_pose.yaw) -
                           odom_delta.y * sin(predicted_pose.yaw);
        predicted_pose.y += odom_delta.x * sin(predicted_pose.yaw) +
                           odom_delta.y * cos(predicted_pose.yaw);
        predicted_pose.yaw = normalizeAngle(predicted_pose.yaw + odom_delta.yaw);

        return predicted_pose;
    }
    catch (const tf2::TransformException& ex) {
        ROS_WARN_STREAM("Failed to predict pose: " << ex.what());
        return state_.last_pose;
    }
}

bool LaserLocalizer::start() {
    if (running_) {
        return true;
    }

    if (!initializeTransforms()) {
        ROS_ERROR("Failed to initialize transforms");
        return false;
    }

    running_ = true;
    ROS_INFO("Laser localizer started");
    return true;
}

void LaserLocalizer::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    ROS_INFO("Laser localizer stopped");
}

} // namespace amr_lidar_localization