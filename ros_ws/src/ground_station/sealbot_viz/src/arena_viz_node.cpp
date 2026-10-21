#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

// Publishes a rectangular arena as four wall markers. The arena is centered on
// the frame origin and the walls hang below the z = 0 plane (top edge at z = 0).
class ArenaViz : public rclcpp::Node
{
public:
  ArenaViz() : Node("arena_viz_node")
  {
    marker_topic_   = declare_parameter<std::string>("marker_topic", "arena");
    frame_id_       = declare_parameter<std::string>("frame_id", "world");
    width_          = declare_parameter<double>("width", 22.86);          // m, along x
    length_         = declare_parameter<double>("length", 50.0);          // m, along y
    wall_depth_     = declare_parameter<double>("wall_depth", 1.0);       // m, below z = 0
    wall_thickness_ = declare_parameter<double>("wall_thickness", 0.05);  // m
    publish_period_ = declare_parameter<double>("publish_period", 1.0);   // s
    color_          = declare_parameter<std::vector<double>>("color", {1.0, 1.0, 0.6, 0.5});  // r g b a
    
    // Transient local so RViz gets the arena even if it starts after this node
    pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      marker_topic_, rclcpp::QoS(1).transient_local());

    timer_ = create_wall_timer(
      std::chrono::duration<double>(publish_period_),
      std::bind(&ArenaViz::publish_arena, this));
    publish_arena();

    RCLCPP_INFO(get_logger(), "Publishing %.2f x %.2f m arena on %s in frame '%s'",
                width_, length_, marker_topic_.c_str(), frame_id_.c_str());
  }

private:
  visualization_msgs::msg::Marker make_wall(int id, double x, double y, double sx, double sy)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id_;
    m.header.stamp = now();
    m.ns = "arena_walls";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = x;
    m.pose.position.y = y;
    m.pose.position.z = -wall_depth_ / 2.0;  // top of wall sits on z = 0
    m.pose.orientation.w = 1.0;
    m.scale.x = sx;
    m.scale.y = sy;
    m.scale.z = wall_depth_;
    m.color.r = color_[0];
    m.color.g = color_[1];
    m.color.b = color_[2];
    m.color.a = color_[3];
    return m;
  }

  void publish_arena()
  {
    const double hw = width_ / 2.0;
    const double hl = length_ / 2.0;
    const double t = wall_thickness_;

    // Walls sit just outside the arena so the inner dimensions are width x length
    visualization_msgs::msg::MarkerArray out;
    out.markers.push_back(make_wall(0,  hw + t / 2.0, 0.0, t, length_ + 2 * t));
    out.markers.push_back(make_wall(1, -hw - t / 2.0, 0.0, t, length_ + 2 * t));
    out.markers.push_back(make_wall(2, 0.0,  hl + t / 2.0, width_, t));
    out.markers.push_back(make_wall(3, 0.0, -hl - t / 2.0, width_, t));
    pub_->publish(out);
  }

  std::string marker_topic_, frame_id_;
  double width_, length_, wall_depth_, wall_thickness_, publish_period_;
  std::vector<double> color_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArenaViz>());
  rclcpp::shutdown();
  return 0;
}
