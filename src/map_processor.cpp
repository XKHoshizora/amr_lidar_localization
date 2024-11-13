#include "amr_lidar_localization/map_processor.h"
#include <ros/ros.h>

namespace amr_lidar_localization {

void MapProcessor::updateMap(const nav_msgs::OccupancyGrid& map) {
    try {
        if (map.data.empty()) {
            throw std::runtime_error("Empty map data received");
        }

        map_info_ = map.info;

        // 转换栅格地图为OpenCV格式
        occupancy_grid_ = cv::Mat(map.info.height, map.info.width, CV_8UC1);
        for (size_t i = 0; i < map.data.size(); ++i) {
            int value = map.data[i];
            if (value == -1) {
                occupancy_grid_.data[i] = 127;  // 未知区域
            } else if (value > 50) {  // 假设50%概率为阈值
                occupancy_grid_.data[i] = 255;  // 障碍物
            } else {
                occupancy_grid_.data[i] = 0;    // 自由空间
            }
        }

        // 计算距离变换
        computeDistanceField();

        ROS_INFO("Map updated: %dx%d pixels, resolution: %.3f",
                 map.info.width, map.info.height, map.info.resolution);
    }
    catch (const std::exception& e) {
        ROS_ERROR("Failed to update map: %s", e.what());
        throw;
    }
}

void MapProcessor::computeDistanceField() {
    try {
        cv::Mat binary;
        cv::threshold(occupancy_grid_, binary, 254, 255, cv::THRESH_BINARY);

        // 计算距离变换
        cv::distanceTransform(binary, distance_field_, cv::DIST_L2, cv::DIST_MASK_PRECISE);

        // 归一化到[0,1]范围
        double min_val, max_val;
        cv::minMaxLoc(distance_field_, &min_val, &max_val);
        if (max_val > 0) {
            distance_field_ /= max_val;
        }

        ROS_DEBUG("Distance field computed. Range: [%.3f, %.3f]", min_val, max_val);
    }
    catch (const cv::Exception& e) {
        ROS_ERROR("OpenCV error in distance field computation: %s", e.what());
        throw;
    }
}

double MapProcessor::getLikelihood(const Pose2D& pose, const std::vector<float>& scan) const {
    if (distance_field_.empty()) {
        ROS_WARN_THROTTLE(1.0, "Distance field is empty");
        return 0.0;
    }

    double likelihood = 0.0;
    int valid_points = 0;
    const double angle_increment = M_PI / 180.0;  // 1度,应从参数读取
    const double max_range = 20.0;  // 应从参数读取

    for (size_t i = 0; i < scan.size(); ++i) {
        if (!std::isfinite(scan[i]) || scan[i] > max_range) {
            continue;
        }

        // 计算激光点在地图坐标系下的位置
        double angle = pose.yaw + i * angle_increment - M_PI;
        double px = pose.x + scan[i] * cos(angle);
        double py = pose.y + scan[i] * sin(angle);

        // 转换到地图坐标系
        cv::Point2d map_point = worldToMap(px, py);

        // 检查点是否在地图范围内
        if (map_point.x >= 0 && map_point.x < distance_field_.cols - 1 &&
            map_point.y >= 0 && map_point.y < distance_field_.rows - 1) {

            // 使用双线性插值获取距离场值
            double dist = bilinearInterpolation(distance_field_, map_point);
            likelihood += dist;
            valid_points++;
        }
    }

    return valid_points > 0 ? likelihood / valid_points : 0.0;
}

cv::Point2d MapProcessor::worldToMap(double x, double y) const {
    // 将世界坐标转换为地图像素坐标
    cv::Point2d map_point;
    map_point.x = (x - map_info_.origin.position.x) / map_info_.resolution;
    map_point.y = (y - map_info_.origin.position.y) / map_info_.resolution;
    return map_point;
}

double MapProcessor::bilinearInterpolation(const cv::Mat& img, const cv::Point2d& pt) const {
    int x = static_cast<int>(pt.x);
    int y = static_cast<int>(pt.y);

    double a = pt.x - x;
    double b = pt.y - y;

    double value = (1-a)*(1-b)*img.at<float>(y,x) +
                  a*(1-b)*img.at<float>(y,x+1) +
                  (1-a)*b*img.at<float>(y+1,x) +
                  a*b*img.at<float>(y+1,x+1);

    return value;
}

bool MapProcessor::isInside(double x, double y) const {
    cv::Point2d pt = worldToMap(x, y);
    return pt.x >= 0 && pt.x < distance_field_.cols &&
           pt.y >= 0 && pt.y < distance_field_.rows;
}

} // namespace amr_lidar_localization