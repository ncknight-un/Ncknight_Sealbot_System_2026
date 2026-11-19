// Make Isaac ROS AprilTag output look exactly like apriltag_ros, so everything
// downstream (panel_calibration, tag_zero_frames, sealbot_viz) works unchanged
// with either detector:
//
//   isaac_ros_apriltag_interfaces/AprilTagDetectionArray (detections_in)
//   + rectified mono image (image) + camera_info
//     -> /tf                    camera_frame -> <tag_prefix><id>   (as apriltag_ros)
//     -> apriltag_msgs/AprilTagDetectionArray (detections_out)
//
// Isaac (cuAprilTags) is only used to *find* tags. Its pose is ignored: the
// corners are sub-pixel refined on the rectified image and the pose is solved
// with PnP using the same tag model and corner order as apriltag_ros
// (pose_estimation_method: pnp), so poses and tag axes match the CPU method.
//
// If a CPU-vs-Isaac calibration comparison shows a constant rotation, Isaac's
// corner order differs from apriltag's:
//   90/180/270 deg about tag Z      -> corner_shift 1/2/3
//   Z flipped (180 deg about X/Y)   -> corner_reverse true (maybe plus a shift)
//   small leftover                  -> tag_rotation_correction [x, y, z, w]

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "apriltag_msgs/msg/april_tag_detection_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"

namespace sealbot
{

using IsaacArray = isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
using RosArray = apriltag_msgs::msg::AprilTagDetectionArray;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;

class IsaacTagBridge : public rclcpp::Node
{
public:
  explicit IsaacTagBridge(const rclcpp::NodeOptions & options)
  : Node("isaac_tag_bridge", options)
  {
    tag_prefix_    = declare_parameter<std::string>("tag_prefix", "tag_");
    camera_frame_  = declare_parameter<std::string>("camera_frame", "");  // empty = image frame
    publish_tf_    = declare_parameter<bool>("publish_tf", true);
    size_          = declare_parameter<double>("size", 0.0085);  // black-border edge [m], as apriltag_ros
    refine_        = declare_parameter<bool>("refine_corners", true);
    corner_shift_  = declare_parameter<int>("corner_shift", 0);
    corner_reverse_ = declare_parameter<bool>("corner_reverse", false);
    max_buffer_    = declare_parameter<int>("image_buffer", 30);
    const auto ids = declare_parameter<std::vector<int64_t>>("tag_ids", std::vector<int64_t>{});
    tag_ids_ = std::set<int64_t>(ids.begin(), ids.end());  // empty = accept all
    const auto corr = declare_parameter<std::vector<double>>(
      "tag_rotation_correction", {0.0, 0.0, 0.0, 1.0});
    if (corr.size() != 4) {
      throw std::invalid_argument("tag_rotation_correction must be [x, y, z, w]");
    }
    correction_ = tf2::Quaternion(corr[0], corr[1], corr[2], corr[3]).normalized();

    // Same object points as apriltag_ros pnp: corner i of the apriltag library
    // (counter-clockwise, starting bottom-left in tag coordinates).
    const double s = size_ / 2.0;
    object_points_ = {{-s, -s, 0.0}, {s, -s, 0.0}, {s, s, 0.0}, {-s, s, 0.0}};

    pub_ = create_publisher<RosArray>("detections_out", rclcpp::SensorDataQoS());
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    info_sub_ = create_subscription<CameraInfo>(
      "camera_info", rclcpp::SensorDataQoS(),
      [this](CameraInfo::ConstSharedPtr msg) {info_ = msg;});
    image_sub_ = create_subscription<Image>(
      "image", rclcpp::SensorDataQoS(),
      std::bind(&IsaacTagBridge::image_callback, this, std::placeholders::_1));
    det_sub_ = create_subscription<IsaacArray>(
      "detections_in", rclcpp::SensorDataQoS(),
      std::bind(&IsaacTagBridge::detections_callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "Bridging %s -> %s, TF frames '%s<id>', tag size %.4f m, refine %s, PnP on %s, "
      "corner_shift %d%s, %s",
      det_sub_->get_topic_name(), pub_->get_topic_name(), tag_prefix_.c_str(), size_,
      refine_ ? "on" : "off", image_sub_->get_topic_name(), corner_shift_,
      corner_reverse_ ? " (reversed)" : "",
      tag_ids_.empty() ? "all tag ids" : (std::to_string(tag_ids_.size()) + " tag ids").c_str());
  }

private:
  void image_callback(Image::ConstSharedPtr msg)
  {
    images_.push_back(msg);
    while (static_cast<int>(images_.size()) > max_buffer_) {
      images_.pop_front();
    }
  }

  // Grayscale view of the buffered image with exactly this stamp
  bool find_image(const builtin_interfaces::msg::Time & stamp, cv::Mat & gray)
  {
    for (auto it = images_.rbegin(); it != images_.rend(); ++it) {
      const auto & img = *it;
      if (img->header.stamp.sec != stamp.sec || img->header.stamp.nanosec != stamp.nanosec) {
        continue;
      }
      auto * data = const_cast<uint8_t *>(img->data.data());
      if (img->encoding == "mono8") {
        gray = cv::Mat(img->height, img->width, CV_8UC1, data, img->step);
      } else if (img->encoding == "rgb8" || img->encoding == "bgr8") {
        cv::Mat color(img->height, img->width, CV_8UC3, data, img->step);
        cv::cvtColor(color, gray, img->encoding == "rgb8" ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY);
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "Unsupported image encoding '%s' (need mono8/rgb8/bgr8)", img->encoding.c_str());
        return false;
      }
      return true;
    }
    return false;
  }

  void detections_callback(const IsaacArray::SharedPtr msg)
  {
    RosArray out;
    out.header = msg->header;
    if (!camera_frame_.empty()) {
      out.header.frame_id = camera_frame_;
    }

    if (!info_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for camera_info");
      return;
    }
    cv::Mat gray;
    if (!msg->detections.empty() && !find_image(msg->header.stamp, gray)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "No rectified image with a matching stamp (%d.%09u) in the last %zu images; "
        "check the 'image' remap", msg->header.stamp.sec, msg->header.stamp.nanosec,
        images_.size());
      return;
    }

    // Rectified image: use the projection matrix and no distortion (as apriltag_ros)
    const auto & P = info_->p;
    const cv::Matx33d K(P[0], 0.0, P[2], 0.0, P[5], P[6], 0.0, 0.0, 1.0);

    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    for (const auto & det : msg->detections) {
      if (!tag_ids_.empty() && !tag_ids_.count(det.id)) {
        continue;
      }

      // Isaac corner order -> apriltag corner order
      std::vector<cv::Point2f> corners(4);
      for (int i = 0; i < 4; ++i) {
        const int src = corner_reverse_ ?
          ((corner_shift_ - i) % 4 + 4) % 4 : ((i + corner_shift_) % 4 + 4) % 4;
        corners[i] = cv::Point2f(det.corners[src].x, det.corners[src].y);
      }

      if (refine_) {
        // Window well inside one tag cell (black border = 8 cells across)
        double min_side = 1e9;
        for (int i = 0; i < 4; ++i) {
          min_side = std::min(min_side, cv::norm(corners[i] - corners[(i + 1) % 4]));
        }
        const int half = std::clamp(static_cast<int>(std::lround(min_side / 8.0 * 0.6)), 1, 5);
        cv::cornerSubPix(gray, corners, cv::Size(half, half), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
      }

      cv::Vec3d rvec, tvec;
      std::vector<cv::Point2d> image_points(corners.begin(), corners.end());
      if (!cv::solvePnP(object_points_, image_points, K, cv::noArray(), rvec, tvec,
                        false, cv::SOLVEPNP_IPPE))
      {
        continue;
      }
      cv::Matx33d R;
      cv::Rodrigues(rvec, R);
      tf2::Quaternion q;
      tf2::Matrix3x3(R(0, 0), R(0, 1), R(0, 2),
                     R(1, 0), R(1, 1), R(1, 2),
                     R(2, 0), R(2, 1), R(2, 2)).getRotation(q);
      q = (q * correction_).normalized();

      apriltag_msgs::msg::AprilTagDetection d;
      d.family = det.family;
      d.id = det.id;
      d.centre.x = det.center.x;
      d.centre.y = det.center.y;
      for (int i = 0; i < 4; ++i) {
        d.corners[i].x = corners[i].x;
        d.corners[i].y = corners[i].y;
      }
      out.detections.push_back(d);

      geometry_msgs::msg::TransformStamped tf;
      tf.header = out.header;
      tf.child_frame_id = tag_prefix_ + std::to_string(det.id);
      tf.transform.translation.x = tvec[0];
      tf.transform.translation.y = tvec[1];
      tf.transform.translation.z = tvec[2];
      tf.transform.rotation.x = q.x();
      tf.transform.rotation.y = q.y();
      tf.transform.rotation.z = q.z();
      tf.transform.rotation.w = q.w();
      transforms.push_back(tf);
    }

    pub_->publish(out);
    if (publish_tf_ && !transforms.empty()) {
      tf_->sendTransform(transforms);
    }
  }

  std::string tag_prefix_, camera_frame_;
  bool publish_tf_, refine_, corner_reverse_;
  double size_;
  int corner_shift_, max_buffer_;
  std::set<int64_t> tag_ids_;
  tf2::Quaternion correction_;
  std::vector<cv::Point3d> object_points_;

  std::deque<Image::ConstSharedPtr> images_;
  CameraInfo::ConstSharedPtr info_;

  rclcpp::Publisher<RosArray>::SharedPtr pub_;
  rclcpp::Subscription<IsaacArray>::SharedPtr det_sub_;
  rclcpp::Subscription<Image>::SharedPtr image_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr info_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

}  // namespace sealbot

RCLCPP_COMPONENTS_REGISTER_NODE(sealbot::IsaacTagBridge)
