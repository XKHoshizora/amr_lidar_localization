#pragma once

#include <opencv2/opencv.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include "types.h"

namespace amr_lidar_localization {

class MapProcessor {
public:
    MapProcessor() = default;
    ~MapProcessor() = default;

    void updateMap(const nav_msgs::OccupancyGrid& map);
    double getLikelihood(const Pose2D& pose, const std::vector<float>& scan) const;
    const cv::Mat& getDistanceField() const { return distance_field_; }
    const nav_msgs::MapMetaData& getMapInfo() const { return map_info_; }
    bool isInside(double x, double y) const;

private:
    cv::Mat occupancy_grid_;       // 栅格地图
    cv::Mat distance_field_;       // 距离变换场
    nav_msgs::MapMetaData map_info_;  // 地图元数据

    void computeDistanceField();  // 计算距离变换
    cv::Point2d worldToMap(double x, double y) const;  // 世界坐标到地图坐标转换
    cv::Point2d worldToMap(const Pose2D& pose) const {
        return worldToMap(pose.x, pose.y);
    }
    double bilinearInterpolation(const cv::Mat& img, const cv::Point2d& pt) const;  // 双线性插值
};

} // namespace amr_lidar_localization