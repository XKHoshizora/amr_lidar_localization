#include <amr_lidar_localization/lidar_locator.h>

LidarLocator::LidarLocator()
    : private_nh_("~")
    , tf_listener_(tf_buffer_)
    , deg_to_rad_(M_PI / 180.0)
    , lidar_x_(250)
    , lidar_y_(250)
    , lidar_yaw_(0)
    , scan_count_(0)
    , clear_countdown_(-1)
    , map_received_(false)
{
    // 获取参数
    private_nh_.param<std::string>("base_frame", base_frame_, "base_footprint");
    private_nh_.param<std::string>("odom_frame", odom_frame_, "odom");
    private_nh_.param<std::string>("laser_frame", laser_frame_, "laser");
    private_nh_.param<std::string>("laser_topic", laser_topic_, "scan");

    // 初始化订阅者和服务客户端
    map_sub_ = nh_.subscribe("map", 1, &LidarLocator::mapCallback, this);
    scan_sub_ = nh_.subscribe(laser_topic_, 1, &LidarLocator::scanCallback, this);
    initial_pose_sub_ = nh_.subscribe("initialpose", 1, &LidarLocator::initialPoseCallback, this);
    clear_costmaps_client_ = nh_.serviceClient<std_srvs::Empty>("move_base/clear_costmaps");

    ROS_INFO("Lidar localization node initialized");
}

void LidarLocator::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    try {
        ROS_INFO("Received map message, size: %dx%d", msg->info.width, msg->info.height);

        if (msg->data.empty()) {
            ROS_ERROR("Received empty map data");
            return;
        }

        map_msg_ = *msg;
        cropMap();
        processMap();
        map_received_ = true;
        ROS_INFO("Map processed successfully");
    } catch (const std::exception& e) {
        ROS_ERROR("Error in mapCallback: %s", e.what());
    }
}

void LidarLocator::initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
    try {
        if (!waitForTransform(base_frame_, laser_frame_, ros::Time(0), ros::Duration(1.0))) {
            ROS_WARN("Transform not available yet");
            return;
        }

        geometry_msgs::TransformStamped transformStamped =
            tf_buffer_.lookupTransform(base_frame_, laser_frame_, ros::Time(0));

        geometry_msgs::PoseStamped base_pose, laser_pose;
        base_pose.header = msg->header;
        base_pose.pose = msg->pose.pose;

        tf2::doTransform(base_pose, laser_pose, transformStamped);

        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;

        tf2::Quaternion q(
            laser_pose.pose.orientation.x,
            laser_pose.pose.orientation.y,
            laser_pose.pose.orientation.z,
            laser_pose.pose.orientation.w);

        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        if (map_received_) {
            lidar_x_ = (x - map_msg_.info.origin.position.x) / map_msg_.info.resolution - map_roi_info_.x_offset;
            lidar_y_ = (y - map_msg_.info.origin.position.y) / map_msg_.info.resolution - map_roi_info_.y_offset;
            lidar_yaw_ = -yaw;
            clear_countdown_ = 30;
            ROS_INFO("Initial pose set to: x=%.2f, y=%.2f, yaw=%.2f", lidar_x_, lidar_y_, lidar_yaw_);
        } else {
            ROS_WARN("Map not received yet, cannot set initial pose");
        }
    } catch (const tf2::TransformException& ex) {
        ROS_WARN("Transform failed: %s", ex.what());
    } catch (const std::exception& e) {
        ROS_ERROR("Error in initialPoseCallback: %s", e.what());
    }
}

void LidarLocator::scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    try {
        if (!map_received_) {
            ROS_DEBUG_THROTTLE(1.0, "Waiting for map data...");
            return;
        }

        scan_points_.clear();
        double angle = msg->angle_min;
        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            if (msg->ranges[i] >= msg->range_min && msg->ranges[i] <= msg->range_max) {
                float x = msg->ranges[i] * cos(angle) / map_msg_.info.resolution;
                float y = -msg->ranges[i] * sin(angle) / map_msg_.info.resolution;
                scan_points_.push_back(cv::Point2f(x, y));
            }
            angle += msg->angle_increment;
        }

        if (scan_count_ == 0) {
            scan_count_++;
        }

        processLidarData();

        if (clear_countdown_ > -1) {
            clear_countdown_--;
        }
        if (clear_countdown_ == 0) {
            std_srvs::Empty srv;
            if (clear_costmaps_client_.call(srv)) {
                ROS_INFO("Costmaps cleared successfully");
            } else {
                ROS_WARN("Failed to clear costmaps");
            }
        }

    } catch (const std::exception& e) {
        ROS_ERROR("Error in scanCallback: %s", e.what());
    }
}

void LidarLocator::processLidarData()
{
    if (map_cropped_.empty() || scan_points_.empty()) {
        return;
    }

    while (ros::ok()) {
        std::vector<cv::Point2f> transform_points, clockwise_points, counter_points;
        int max_sum = 0;
        float best_dx = 0, best_dy = 0, best_dyaw = 0;

        // 计算三种旋转情况下的点
        for (const auto& point : scan_points_) {
            // 原始角度
            float rotated_x = point.x * cos(lidar_yaw_) - point.y * sin(lidar_yaw_);
            float rotated_y = point.x * sin(lidar_yaw_) + point.y * cos(lidar_yaw_);
            transform_points.push_back(cv::Point2f(rotated_x + lidar_x_, lidar_y_ - rotated_y));

            // 顺时针旋转
            float clockwise_yaw = lidar_yaw_ + deg_to_rad_;
            rotated_x = point.x * cos(clockwise_yaw) - point.y * sin(clockwise_yaw);
            rotated_y = point.x * sin(clockwise_yaw) + point.y * cos(clockwise_yaw);
            clockwise_points.push_back(cv::Point2f(rotated_x + lidar_x_, lidar_y_ - rotated_y));

            // 逆时针旋转
            float counter_yaw = lidar_yaw_ - deg_to_rad_;
            rotated_x = point.x * cos(counter_yaw) - point.y * sin(counter_yaw);
            rotated_y = point.x * sin(counter_yaw) + point.y * cos(counter_yaw);
            counter_points.push_back(cv::Point2f(rotated_x + lidar_x_, lidar_y_ - rotated_y));
        }

        // 计算最佳匹配
        std::vector<cv::Point2f> offsets = {{0,0}, {1,0}, {-1,0}, {0,1}, {0,-1}};
        std::vector<std::vector<cv::Point2f>> point_sets = {transform_points, clockwise_points, counter_points};
        std::vector<float> yaw_offsets = {0, deg_to_rad_, -deg_to_rad_};

        for (size_t i = 0; i < offsets.size(); ++i) {
            for (size_t j = 0; j < point_sets.size(); ++j) {
                int sum = 0;
                for (const auto& point : point_sets[j]) {
                    float px = point.x + offsets[i].x;
                    float py = point.y + offsets[i].y;
                    if (px >= 0 && px < map_temp_.cols && py >= 0 && py < map_temp_.rows) {
                        sum += map_temp_.at<uchar>(py, px);
                    }
                }
                if (sum > max_sum) {
                    max_sum = sum;
                    best_dx = offsets[i].x;
                    best_dy = offsets[i].y;
                    best_dyaw = yaw_offsets[j];
                }
            }
        }

        // 更新位置和角度
        lidar_x_ += best_dx;
        lidar_y_ += best_dy;
        lidar_yaw_ += best_dyaw;

        if (check(lidar_x_, lidar_y_, lidar_yaw_)) {
            break;
        }
    }

    // 发布转换
    poseTF();
}

void LidarLocator::cropMap()
{
    ROS_INFO("Starting cropMap");

    if (map_msg_.data.empty()) {
        ROS_ERROR("Empty map data in cropMap");
        return;
    }

    try {
        nav_msgs::MapMetaData info = map_msg_.info;
        int xMax, xMin, yMax, yMin;
        xMax = xMin = info.width/2;
        yMax = yMin = info.height/2;
        bool bFirstPoint = true;

        cv::Mat map_raw(info.height, info.width, CV_8UC1, cv::Scalar(128));

        for(int y = 0; y < info.height; y++) {
            for(int x = 0; x < info.width; x++) {
                int index = y * info.width + x;
                map_raw.at<uchar>(y, x) = static_cast<uchar>(map_msg_.data[index]);

                if(map_msg_.data[index] == 100) {
                    if(bFirstPoint) {
                        xMax = xMin = x;
                        yMax = yMin = y;
                        bFirstPoint = false;
                        continue;
                    }
                    xMin = std::min(xMin, x);
                    xMax = std::max(xMax, x);
                    yMin = std::min(yMin, y);
                    yMax = std::max(yMax, y);
                }
            }
        }

        int cen_x = (xMin + xMax)/2;
        int cen_y = (yMin + yMax)/2;

        int new_half_width = abs(xMax - xMin)/2 + 50;
        int new_half_height = abs(yMax - yMin)/2 + 50;
        int new_origin_x = cen_x - new_half_width;
        int new_origin_y = cen_y - new_half_height;
        int new_width = new_half_width*2;
        int new_height = new_half_height*2;

        new_origin_x = std::max(0, new_origin_x);
        new_width = std::min(new_width, static_cast<int>(info.width - new_origin_x));
        new_origin_y = std::max(0, new_origin_y);
        new_height = std::min(new_height, static_cast<int>(info.height - new_origin_y));

        cv::Rect roi(new_origin_x, new_origin_y, new_width, new_height);
        map_cropped_ = map_raw(roi).clone();

        map_roi_info_.x_offset = new_origin_x;
        map_roi_info_.y_offset = new_origin_y;
        map_roi_info_.width = new_width;
        map_roi_info_.height = new_height;

        ROS_INFO("Map cropped successfully");
        ROS_INFO("Map cropped: origin(%d,%d) size(%d,%d)",
                 new_origin_x, new_origin_y, new_width, new_height);
    } catch (const std::exception& e) {
        ROS_ERROR("Error in cropMap: %s", e.what());
    }
}

cv::Mat LidarLocator::createGradientMask(int size)
{
    try {
        ROS_INFO("Creating gradient mask of size %dx%d", size, size);
        cv::Mat mask(size, size, CV_8UC1);
        int center = size / 2;

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double distance = std::hypot(x - center, y - center);
                int value = cv::saturate_cast<uchar>(255 * std::max(0.0, 1.0 - distance / center));
                mask.at<uchar>(y, x) = value;
            }
        }

        ROS_INFO("Gradient mask created successfully");
        return mask;
    } catch (const cv::Exception& e) {
        ROS_ERROR("OpenCV error in createGradientMask: %s", e.what());
        return cv::Mat();
    } catch (const std::exception& e) {
        ROS_ERROR("Error in createGradientMask: %s", e.what());
        return cv::Mat();
    }
}

void LidarLocator::processMap()
{
    ROS_INFO("Starting processMap");

    if (map_cropped_.empty()) {
        ROS_ERROR("Empty cropped map in processMap");
        return;
    }

    try {
        ROS_INFO("Creating zero matrix of size %dx%d", map_cropped_.rows, map_cropped_.cols);
        map_temp_ = cv::Mat::zeros(map_cropped_.size(), CV_8UC1);

        ROS_INFO("Creating gradient mask");
        cv::Mat gradient_mask = createGradientMask(101);
        if (gradient_mask.empty()) {
            ROS_ERROR("Failed to create gradient mask");
            return;
        }

        ROS_INFO("Processing map with dimensions: %dx%d", map_cropped_.rows, map_cropped_.cols);

        for (int y = 0; y < map_cropped_.rows; y++) {
            for (int x = 0; x < map_cropped_.cols; x++) {
                if (map_cropped_.at<uchar>(y, x) == 100) {
                    // 计算ROI的边界，确保不会越界
                    int left = std::max(0, x - 50);
                    int top = std::max(0, y - 50);
                    int right = std::min(map_cropped_.cols - 1, x + 50);
                    int bottom = std::min(map_cropped_.rows - 1, y + 50);

                    if (right < left || bottom < top) {
                        ROS_ERROR("Invalid ROI boundaries: left=%d, right=%d, top=%d, bottom=%d",
                                 left, right, top, bottom);
                        continue;
                    }

                    cv::Rect roi(left, top, right - left + 1, bottom - top + 1);
                    if (!roi.width || !roi.height) {
                        ROS_ERROR("Invalid ROI size: %dx%d", roi.width, roi.height);
                        continue;
                    }

                    // 检查ROI是否在图像范围内
                    if ((roi & cv::Rect(0, 0, map_temp_.cols, map_temp_.rows)) != roi) {
                        ROS_ERROR("ROI outside image boundaries");
                        continue;
                    }

                    cv::Mat region = map_temp_(roi);

                    int mask_left = 50 - (x - left);
                    int mask_top = 50 - (y - top);

                    // 检查mask_roi的有效性
                    if (mask_left < 0 || mask_top < 0 ||
                        mask_left + roi.width > gradient_mask.cols ||
                        mask_top + roi.height > gradient_mask.rows) {
                        ROS_ERROR("Invalid mask ROI parameters");
                        continue;
                    }

                    cv::Rect mask_roi(mask_left, mask_top, roi.width, roi.height);
                    cv::Mat mask = gradient_mask(mask_roi);

                    cv::max(region, mask, region);
                }
            }
        }

        ROS_INFO("Map processing completed successfully");
    } catch (const cv::Exception& e) {
        ROS_ERROR("OpenCV error in processMap: %s", e.what());
    } catch (const std::exception& e) {
        ROS_ERROR("Error in processMap: %s", e.what());
    }
}

bool LidarLocator::check(float x, float y, float yaw)
{
    if (x == 0 && y == 0 && yaw == 0) {
        data_queue_.clear();
        return true;
    }

    data_queue_.push_back(std::make_tuple(x, y, yaw));

    if (data_queue_.size() > max_queue_size_) {
        data_queue_.pop_front();
    }

    if (data_queue_.size() == max_queue_size_) {
        auto& first = data_queue_.front();
        auto& last = data_queue_.back();

        float dx = std::abs(std::get<0>(last) - std::get<0>(first));
        float dy = std::abs(std::get<1>(last) - std::get<1>(first));
        float dyaw = std::abs(std::get<2>(last) - std::get<2>(first));

        if (dx < 5 && dy < 5 && dyaw < 5*deg_to_rad_) {
            data_queue_.clear();
            return true;
        }
    }
    return false;
}

void LidarLocator::poseTF()
{
    if (scan_count_ == 0 || !map_received_) {
        return;
    }

    try {
        // 等待必要的转换可用
        if (!waitForTransform(odom_frame_, laser_frame_, ros::Time(0), ros::Duration(0.1))) {
            return;
        }

        // 计算实际米制坐标
        double x_meters = (lidar_x_ + map_roi_info_.x_offset) * map_msg_.info.resolution;
        double y_meters = (lidar_y_ + map_roi_info_.y_offset) * map_msg_.info.resolution;

        // 考虑地图原点偏移
        x_meters += map_msg_.info.origin.position.x;
        y_meters += map_msg_.info.origin.position.y;

        // 处理yaw角度并转换为四元数
        double yaw_ros = -lidar_yaw_;
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw_ros);

        // 计算base在map中的位置
        double base_x = x_meters;
        double base_y = y_meters;

        // 获取odom到base的转换
        geometry_msgs::TransformStamped odom_to_base =
            tf_buffer_.lookupTransform(odom_frame_, laser_frame_, ros::Time(0));

        // 计算map到odom的转换
        tf2::Transform map_to_base, odom_to_base_tf2;
        map_to_base.setOrigin(tf2::Vector3(base_x, base_y, 0));
        map_to_base.setRotation(q);

        tf2::fromMsg(odom_to_base.transform, odom_to_base_tf2);
        tf2::Transform map_to_odom = map_to_base * odom_to_base_tf2.inverse();

        // 发布转换
        geometry_msgs::TransformStamped map_to_odom_msg;
        map_to_odom_msg.header.stamp = ros::Time::now();
        map_to_odom_msg.header.frame_id = "map";
        map_to_odom_msg.child_frame_id = odom_frame_;
        map_to_odom_msg.transform = tf2::toMsg(map_to_odom);

        tf_broadcaster_.sendTransform(map_to_odom_msg);

    } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(1.0, "Transform failure: %s", ex.what());
    } catch (const std::exception& e) {
        ROS_ERROR("Error in poseTF: %s", e.what());
    }
}

bool LidarLocator::waitForTransform(const std::string& target_frame,
                               const std::string& source_frame,
                               const ros::Time& time,
                               const ros::Duration& timeout)
{
    try {
        return tf_buffer_.canTransform(target_frame, source_frame, time, timeout);
    } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(1.0, "waitForTransform failed: %s", ex.what());
        return false;
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_locator");

    try {
        LidarLocator lidar_locator;
        ros::Rate rate(30);

        // 给其他节点一些启动时间
        ros::Duration(1.0).sleep();

        while (ros::ok()) {
            try {
                ros::spinOnce();
                rate.sleep();
            } catch (const std::exception& e) {
                ROS_ERROR("Error in main loop: %s", e.what());
            }
        }
    } catch (const std::exception& e) {
        ROS_FATAL("Fatal error in lidar_locator: %s", e.what());
        return 1;
    }

    return 0;
}