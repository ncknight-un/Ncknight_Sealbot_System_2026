// Make Isaac ROS AprilTag output look exactly like apriltag_ros, so everything
// downstream (panel_calibration, tag_zero_frames, sealbot_viz) works unchanged
// with either detector:
//
//   isaac_ros_apriltag_interfaces/AprilTagDetectionArray (input, default /isaac/tag_detections)
//     -> /tf                    camera_frame -> <tag_prefix><id>   (as apriltag_ros)
//     -> apriltag_msgs/AprilTagDetectionArray (output, default /detections)
//
// tag_rotation_correction [x, y, z, w] is right-multiplied onto every tag pose,
// in case cuAprilTags' tag-frame axes differ from apriltag_ros'. Leave it as
// identity unless a side-by-side comparison of both detectors shows a fixed
// rotation (see apriltag_bringup/README.md).

#include <memory>
#include <string>
#include <vector>

#include "apriltag_msgs/msg/april_tag_detection_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"

using IsaacArray = isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
using RosArray = apriltag_msgs::msg::AprilTagDetectionArray;

class IsaacTagBridge : public rclcpp::Node
{
public:
  IsaacTagBridge() : Node("isaac_tag_bridge")
  {
    tag_prefix_   = declare_parameter<std::string>("tag_prefix", "tag_");
    camera_frame_ = declare_parameter<std::string>("camera_frame", "");  // empty = use message frame
    publish_tf_   = declare_parameter<bool>("publish_tf", true);
    const auto corr = declare_parameter<std::vector<double>>(
      "tag_rotation_correction", {0.0, 0.0, 0.0, 1.0});
    if (corr.size() != 4) {
      throw std::invalid_argument("tag_rotation_correction must be [x, y, z, w]");
    }
    correction_ = tf2::Quaternion(corr[0], corr[1], corr[2], corr[3]).normalized();

    pub_ = create_publisher<RosArray>("detections_out", rclcpp::SensorDataQoS());
    sub_ = create_subscription<IsaacArray>(
      "detections_in", rclcpp::SensorDataQoS(),
      std::bind(&IsaacTagBridge::callback, this, std::placeholders::_1));
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    RCLCPP_INFO(get_logger(), "Bridging %s -> %s, TF frames '%s<id>'",
                sub_->get_topic_name(), pub_->get_topic_name(), tag_prefix_.c_str());
  }

private:
  void callback(const IsaacArray::SharedPtr msg)
  {
    RosArray out;
    out.header = msg->header;
    if (!camera_frame_.empty()) {
      out.header.frame_id = camera_frame_;
    }

    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    for (const auto & det : msg->detections) {
      apriltag_msgs::msg::AprilTagDetection d;
      d.family = det.family;
      d.id = det.id;
      d.centre.x = det.center.x;
      d.centre.y = det.center.y;
      for (size_t i = 0; i < 4; ++i) {
        d.corners[i].x = det.corners[i].x;
        d.corners[i].y = det.corners[i].y;
      }
      // Isaac provides no hamming / homography; left at zero.
      out.detections.push_back(d);

      const auto & p = det.pose.pose.pose;
      tf2::Quaternion q(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
      q = (q * correction_).normalized();

      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = msg->header.stamp;
      tf.header.frame_id = out.header.frame_id;
      tf.child_frame_id = tag_prefix_ + std::to_string(det.id);
      tf.transform.translation.x = p.position.x;
      tf.transform.translation.y = p.position.y;
      tf.transform.translation.z = p.position.z;
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
  bool publish_tf_;
  tf2::Quaternion correction_;
  rclcpp::Publisher<RosArray>::SharedPtr pub_;
  rclcpp::Subscription<IsaacArray>::SharedPtr sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IsaacTagBridge>());
  rclcpp::shutdown();
  return 0;
}
