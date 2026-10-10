#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "apriltag_msgs/msg/april_tag_detection_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

class TagCylinders : public rclcpp::Node
{
public:
  TagCylinders() : Node("tag_viz_node")
  {
    detections_topic_ = declare_parameter<std::string>("detections_topic", "/detections");
    marker_topic_     = declare_parameter<std::string>("marker_topic", "tag_cylinders");
    frame_prefix_ = declare_parameter<std::string>("frame_prefix", "tag_");
    diameter_         = declare_parameter<double>("diameter", 0.001);          // m
    height_           = declare_parameter<double>("height", 0.05);            // m
    lifetime_         = declare_parameter<double>("lifetime", 0.3);           // s
    color_            = declare_parameter<std::vector<double>>(
                          "color", {0.2, 0.6, 1.0, 0.9});                     // r g b a

    if (color_.size() != 4) {
      RCLCPP_WARN(get_logger(), "'color' must have 4 elements; using default.");
      color_ = {0.1, 0.6, 1.0, 0.9};
    }

    pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);
    sub_ = create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>(
      detections_topic_, rclcpp::SensorDataQoS(),
      std::bind(&TagCylinders::callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Listening on %s, publishing %s",
                detections_topic_.c_str(), marker_topic_.c_str());
  }

private:
  void callback(const apriltag_msgs::msg::AprilTagDetectionArray::SharedPtr msg)
  {
    visualization_msgs::msg::MarkerArray out;
    const builtin_interfaces::msg::Duration life =
      rclcpp::Duration::from_seconds(lifetime_);

    for (const auto & det : msg->detections) {
      visualization_msgs::msg::Marker m;
      // apriltag_ros publishes tag frames as "<family>:<id>" by default
      m.header.frame_id = frame_prefix_ + std::to_string(det.id);
      m.header.stamp = msg->header.stamp;
      m.ns = "tag_cylinders";
      m.id = det.id;
      m.type = visualization_msgs::msg::Marker::CYLINDER;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.orientation.w = 1.0;
      // tag z-axis points out of the tag; lift so the cylinder sits on it
      m.pose.position.z = -height_ / 2.0;
      m.scale.x = diameter_;
      m.scale.y = diameter_;
      m.scale.z = height_;
      m.color.r = color_[0];
      m.color.g = color_[1];
      m.color.b = color_[2];
      m.color.a = color_[3];
      m.lifetime = life;   // marker expires on its own when the tag is lost
      out.markers.push_back(m);
    }

    if (!out.markers.empty()) {
      pub_->publish(out);
    }
  }

  std::string detections_topic_, marker_topic_, frame_prefix_;
  double diameter_, height_, lifetime_;
  std::vector<double> color_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
  rclcpp::Subscription<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TagCylinders>());
  rclcpp::shutdown();
  return 0;
}