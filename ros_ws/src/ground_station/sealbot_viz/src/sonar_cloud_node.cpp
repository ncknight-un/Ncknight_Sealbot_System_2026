#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ping360_sonar_msgs/msg/sonar_echo.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

// Converts raw Ping360 echoes (one beam per message) into a PointCloud2 for RViz.
// Keeps the latest beam at each angle, so the cloud shows the whole sweep, and
// drops returns below `intensity_threshold` or closer than `min_range`.
class SonarCloud : public rclcpp::Node
{
public:
  SonarCloud() : Node("sonar_cloud_node")
  {
    echo_topic_  = declare_parameter<std::string>("echo_topic", "/scan_echo");
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "sonar_cloud");
    threshold_   = declare_parameter<int>("intensity_threshold", 100);  // 0-255
    min_range_   = declare_parameter<double>("min_range", 0.3);         // m, skips transducer ringing

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic_, 10);
    // SensorDataQoS matches the driver (best effort) and also works with bag playback
    sub_ = create_subscription<ping360_sonar_msgs::msg::SonarEcho>(
      echo_topic_, rclcpp::SensorDataQoS(),
      std::bind(&SonarCloud::callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Listening on %s, publishing %s",
                echo_topic_.c_str(), cloud_topic_.c_str());
  }

private:
  struct Point
  {
    float x, y, intensity;
  };

  void callback(const ping360_sonar_msgs::msg::SonarEcho::SharedPtr msg)
  {
    const auto n = msg->intensities.size();
    if (n == 0) {
      return;
    }

    // Same range model as the driver's LaserScan: sample i is at (i + 1) * range / n
    const double step = static_cast<double>(msg->range) / n;
    const double c = std::cos(msg->angle);
    const double s = std::sin(msg->angle);

    std::vector<Point> beam;
    for (size_t i = 0; i < n; ++i) {
      const double r = (i + 1) * step;
      if (msg->intensities[i] < threshold_ || r < min_range_) {
        continue;
      }
      beam.push_back({static_cast<float>(r * c), static_cast<float>(r * s),
                      static_cast<float>(msg->intensities[i])});
    }

    // The head steps in whole gradians (400 per turn), so this key is exact
    const long key = std::lround(msg->angle * 200.0 / M_PI);
    sweep_[key] = std::move(beam);

    publish(msg->header);
  }

  void publish(const std_msgs::msg::Header & header)
  {
    size_t count = 0;
    for (const auto & kv : sweep_) {
      count += kv.second.size();
    }

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = header;
    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2Fields(4,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);
    mod.resize(count);

    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z"),
      in(cloud, "intensity");
    for (const auto & kv : sweep_) {
      for (const auto & p : kv.second) {
        *x = p.x; *y = p.y; *z = 0.0f; *in = p.intensity;
        ++x; ++y; ++z; ++in;
      }
    }
    pub_->publish(cloud);
  }

  std::string echo_topic_, cloud_topic_;
  int threshold_;
  double min_range_;
  std::map<long, std::vector<Point>> sweep_;  // gradian -> latest beam

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<ping360_sonar_msgs::msg::SonarEcho>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SonarCloud>());
  rclcpp::shutdown();
  return 0;
}
