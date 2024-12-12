#ifndef LIDAR_LOC_H
#define LIDAR_LOC_H

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <sensor_msgs/LaserScan.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>
#include <vector>
#include <deque>
#include <cmath>

class LidarLocator {
public:
    LidarLocator();
    ~LidarLocator() = default;

private:
    // ROS相关
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    ros::Subscriber map_sub_;
    ros::Subscriber scan_sub_;
    ros::Subscriber initial_pose_sub_;
    ros::ServiceClient clear_costmaps_client_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // 参数
    std::string base_frame_;
    std::string odom_frame_;
    std::string laser_frame_;
    std::string laser_topic_;

    // 数据存储
    nav_msgs::OccupancyGrid map_msg_;
    cv::Mat map_cropped_;
    cv::Mat map_temp_;
    cv::Mat map_match_;
    sensor_msgs::RegionOfInterest map_roi_info_;
    std::vector<cv::Point2f> scan_points_;
    std::deque<std::tuple<float, float, float>> data_queue_;

    // 状态变量
    float lidar_x_, lidar_y_, lidar_yaw_;
    const float deg_to_rad_;
    int scan_count_;
    int clear_countdown_;
    static constexpr size_t max_queue_size_ = 10;
    bool map_received_;

    // 回调函数
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
    void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg);

    // 辅助函数
    void processLidarData();
    void cropMap();
    void processMap();
    bool check(float x, float y, float yaw);
    cv::Mat createGradientMask(int size);
    void poseTF();
    bool waitForTransform(const std::string& target_frame, const std::string& source_frame,
                         const ros::Time& time, const ros::Duration& timeout);
};

#endif // LIDAR_LOC_H