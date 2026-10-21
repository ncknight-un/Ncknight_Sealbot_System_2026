// Publish each tag's calibrated zero frame as a static TF.
//
// Loads the YAML written by panel_calibration (pass it as a params file) and
// publishes camera_frame -> <tag>_zero. The deviation of a tag in its own zeroed
// frame is then just the TF lookup <tag>_zero -> <tag>, e.g.
// `ros2 run tf2_ros tf2_echo tag_1_zero tag_1`.

#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

class TagZeroFrames : public rclcpp::Node
{
public:
  TagZeroFrames()
  : Node("tag_zero_frames",
         rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
  {
    std::string camera_frame = "camera_optical_frame";
    get_parameter("camera_frame", camera_frame);
    std::vector<std::string> tag_frames;
    if (!get_parameter("tag_frames", tag_frames) || tag_frames.empty()) {
      RCLCPP_ERROR(get_logger(), "No tag_frames found; pass the calibration file as a params file.");
      return;
    }

    const auto stamp = now();
    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    std::string names;
    for (const auto & frame : tag_frames) {
      std::vector<double> t, q;
      std::string zero_frame = frame + "_zero";
      get_parameter("tags." + frame + ".zero_frame", zero_frame);
      if (!get_parameter("tags." + frame + ".translation", t) || t.size() != 3 ||
          !get_parameter("tags." + frame + ".rotation", q) || q.size() != 4)
      {
        RCLCPP_WARN(get_logger(), "%s: missing or malformed translation/rotation, skipping.",
                    frame.c_str());
        continue;
      }

      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = stamp;
      tf.header.frame_id = camera_frame;
      tf.child_frame_id = zero_frame;
      tf.transform.translation.x = t[0];
      tf.transform.translation.y = t[1];
      tf.transform.translation.z = t[2];
      tf.transform.rotation.x = q[0];
      tf.transform.rotation.y = q[1];
      tf.transform.rotation.z = q[2];
      tf.transform.rotation.w = q[3];
      transforms.push_back(tf);
      names += (names.empty() ? "" : ", ") + zero_frame;
    }

    broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    broadcaster_->sendTransform(transforms);
    RCLCPP_INFO(get_logger(), "Publishing %zu zero frames under %s: %s",
                transforms.size(), camera_frame.c_str(), names.c_str());
  }

private:
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> broadcaster_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TagZeroFrames>());
  rclcpp::shutdown();
  return 0;
}
