#include <memory>
#include <thread>
#include <ros/ros.h>
#include "amr_lidar_localization/laser_localizer.h"

namespace amr_lidar_localization {

class LocalizationNode {
public:
    LocalizationNode() {
        // 获取节点句柄
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");

        // 加载参数
        loadParameters(pnh);

        // 初始化日志级别
        if (debug_) {
            if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                ros::console::levels::Debug)) {
                ros::console::notifyLoggerLevelsChanged();
            }
        }

        try {
            // 创建定位器实例
            localizer_ = std::make_unique<LaserLocalizer>(nh, pnh);

            // 启动定位器
            if (!localizer_->start()) {
                throw std::runtime_error("Failed to start localizer");
            }

            ROS_INFO("Laser localization node initialized successfully");
        }
        catch (const std::exception& e) {
            ROS_FATAL("Failed to initialize laser localization: %s", e.what());
            ros::shutdown();
            throw;
        }
    }

    void run() {
        // 设置线程数
        int thread_count = thread_count_ <= 0 ?
                          std::thread::hardware_concurrency() : thread_count_;
        ROS_INFO("Starting laser localization with %d threads", thread_count);

        // 创建异步spinner
        ros::AsyncSpinner spinner(thread_count);
        spinner.start();

        // 等待关闭
        ros::waitForShutdown();

        // 停止定位器
        if (localizer_) {
            localizer_->stop();
        }
    }

private:
    std::unique_ptr<LaserLocalizer> localizer_;
    bool debug_{false};
    int thread_count_{4};

    void loadParameters(ros::NodeHandle& pnh) {
        pnh.param<bool>("debug", debug_, false);
        pnh.param<int>("thread_count", thread_count_, 4);
    }
};

} // namespace amr_lidar_localization

int main(int argc, char** argv) {
    try {
        // 初始化ROS节点
        ros::init(argc, argv, "laser_localization");

        // 设置日志输出格式
        ros::console::initialize();

        // 创建并运行节点
        amr_lidar_localization::LocalizationNode node;
        node.run();

        return 0;
    }
    catch (const std::exception& e) {
        ROS_FATAL("Laser localization failed: %s", e.what());
        return 1;
    }
}