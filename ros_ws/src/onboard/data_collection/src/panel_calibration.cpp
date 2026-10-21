// Record the zero (rest) pose of every AprilTag on the panel.
//
// Listens to the camera -> tag transforms that apriltag_ros broadcasts on /tf,
// collects them for `duration` seconds (starting at the first detection), averages
// each tag's pose and writes a YAML config. The config holds, per tag, the averaged
// camera -> tag_N transform, i.e. the pose of the tag's zero frame `tag_N_zero` in
// the camera frame. Applying it:
//
//     T_zero_tag = inv(T_cam_zero) * T_cam_tag
//
// gives the tag's deviation expressed in its own zeroed frame (tag_zero_frames
// publishes the zero frames as static TFs so TF can do this for you).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

#ifndef DATA_COLLECTION_SOURCE_DIR
#define DATA_COLLECTION_SOURCE_DIR "."
#endif

namespace
{

struct Sample
{
  Eigen::Vector3d t;
  Eigen::Vector4d q;  // x y z w
};

// Markley's method: principal eigenvector of sum(q q^T), which is sign-invariant
Eigen::Vector4d average_quaternion(const std::vector<Sample> & samples)
{
  Eigen::Matrix4d m = Eigen::Matrix4d::Zero();
  for (const auto & s : samples) {
    m += s.q * s.q.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(m);
  Eigen::Vector4d mean = solver.eigenvectors().col(3);  // eigenvalues sorted ascending
  return mean(3) >= 0.0 ? mean : Eigen::Vector4d(-mean);
}

double quat_angle_deg(const Eigen::Vector4d & a, const Eigen::Vector4d & b)
{
  const double d = std::clamp(std::abs(a.dot(b)), 0.0, 1.0);
  return 2.0 * std::acos(d) * 180.0 / M_PI;
}

// Always prints a decimal point so ROS parses the lists as double arrays
std::string fmt_list(const double * v, int n, int precision)
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(precision) << "[";
  for (int i = 0; i < n; ++i) {
    ss << (i ? ", " : "") << v[i];
  }
  ss << "]";
  return ss.str();
}

}  // namespace

class PanelCalibration : public rclcpp::Node
{
public:
  PanelCalibration() : Node("panel_calibration")
  {
    rcl_interfaces::msg::ParameterDescriptor number;
    number.dynamic_typing = true;  // so `duration:=5` (an int) works as well as `duration:=5.0`

    camera_frame_    = declare_parameter<std::string>("camera_frame", "camera_optical_frame");
    tag_prefix_      = declare_parameter<std::string>("tag_prefix", "tag_");
    duration_        = as_double(declare_parameter("duration", rclcpp::ParameterValue(5.0), number));          // s of data to average
    startup_timeout_ = as_double(declare_parameter("startup_timeout", rclcpp::ParameterValue(30.0), number));  // s to wait for first tag
    min_samples_     = declare_parameter<int>("min_samples", 10);
    output_file_     = declare_parameter<std::string>("output_file", "");
    if (output_file_.empty()) {
      output_file_ = std::string(DATA_COLLECTION_SOURCE_DIR) + "/config/panel_zero.yaml";
    }

    sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf", rclcpp::QoS(100),
      std::bind(&PanelCalibration::tf_callback, this, std::placeholders::_1));
    start_time_ = std::chrono::steady_clock::now();
    timer_ = create_wall_timer(std::chrono::milliseconds(200),
                               std::bind(&PanelCalibration::check_done, this));

    RCLCPP_INFO(get_logger(),
      "Waiting for '%s*' transforms from '%s'. Keep the panel at rest; "
      "recording %.1f s once tags are seen.",
      tag_prefix_.c_str(), camera_frame_.c_str(), duration_);
  }

  bool succeeded() const { return succeeded_; }

private:
  static double as_double(const rclcpp::ParameterValue & v)
  {
    return v.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER ?
      static_cast<double>(v.get<int64_t>()) : v.get<double>();
  }

  void tf_callback(const tf2_msgs::msg::TFMessage::SharedPtr msg)
  {
    if (done_) {
      return;
    }
    for (const auto & tf : msg->transforms) {
      const auto & child = tf.child_frame_id;
      if (tf.header.frame_id != camera_frame_ || child.rfind(tag_prefix_, 0) != 0) {
        continue;
      }
      if (!recording_) {
        recording_ = true;
        first_sample_time_ = std::chrono::steady_clock::now();
        RCLCPP_INFO(get_logger(), "First tag seen, recording...");
      }
      const auto & t = tf.transform.translation;
      const auto & r = tf.transform.rotation;
      Eigen::Vector4d q(r.x, r.y, r.z, r.w);
      samples_[child].push_back({Eigen::Vector3d(t.x, t.y, t.z), q.normalized()});
    }
  }

  void check_done()
  {
    if (done_) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!recording_) {
      if (std::chrono::duration<double>(now - start_time_).count() > startup_timeout_) {
        RCLCPP_ERROR(get_logger(),
          "No tag transforms received within %.0f s; nothing written.", startup_timeout_);
        finish();
      }
      return;
    }
    if (std::chrono::duration<double>(now - first_sample_time_).count() >= duration_) {
      succeeded_ = write_config();
      finish();
    }
  }

  // Sort tag_2 before tag_10
  bool tag_less(const std::string & a, const std::string & b) const
  {
    const auto sa = a.substr(tag_prefix_.size());
    const auto sb = b.substr(tag_prefix_.size());
    const bool na = !sa.empty() && std::all_of(sa.begin(), sa.end(), ::isdigit);
    const bool nb = !sb.empty() && std::all_of(sb.begin(), sb.end(), ::isdigit);
    if (na && nb) {
      return std::stol(sa) < std::stol(sb);
    }
    if (na != nb) {
      return na;
    }
    return sa < sb;
  }

  bool write_config()
  {
    std::vector<std::string> frames;
    for (const auto & kv : samples_) {
      frames.push_back(kv.first);
    }
    std::sort(frames.begin(), frames.end(),
              [this](const auto & a, const auto & b) { return tag_less(a, b); });

    std::ostringstream tags_yaml;
    std::vector<std::string> written;
    for (const auto & frame : frames) {
      const auto & data = samples_[frame];
      const auto n = static_cast<int>(data.size());
      if (n < min_samples_) {
        RCLCPP_WARN(get_logger(), "%s: only %d samples (< %d), skipping.",
                    frame.c_str(), n, min_samples_);
        continue;
      }

      Eigen::Vector3d mean_t = Eigen::Vector3d::Zero();
      for (const auto & s : data) {
        mean_t += s.t;
      }
      mean_t /= n;
      Eigen::Vector3d var_t = Eigen::Vector3d::Zero();
      double ang_sq = 0.0;
      const Eigen::Vector4d mean_q = average_quaternion(data);
      for (const auto & s : data) {
        var_t += (s.t - mean_t).cwiseAbs2();
        ang_sq += std::pow(quat_angle_deg(s.q, mean_q), 2);
      }
      const Eigen::Vector3d std_t = (var_t / n).cwiseSqrt();
      const double ang_std = std::sqrt(ang_sq / n);

      tags_yaml
        << "      " << frame << ":\n"
        << "        zero_frame: " << frame << "_zero\n"
        << "        translation: " << fmt_list(mean_t.data(), 3, 6) << "\n"
        << "        rotation: " << fmt_list(mean_q.data(), 4, 8) << "\n"
        << "        samples: " << n << "\n"
        << "        translation_std: " << fmt_list(std_t.data(), 3, 6) << "\n"
        << "        rotation_std_deg: " << std::fixed << std::setprecision(4) << ang_std << "\n";
      written.push_back(frame);

      RCLCPP_INFO(get_logger(),
        "%s: %d samples, t_std = [%.3f %.3f %.3f] mm, rot_std = %.3f deg",
        frame.c_str(), n, std_t.x() * 1e3, std_t.y() * 1e3, std_t.z() * 1e3, ang_std);
    }

    if (written.empty()) {
      RCLCPP_ERROR(get_logger(), "No tag had enough samples; nothing written.");
      return false;
    }

    std::ostringstream frame_list;
    for (size_t i = 0; i < written.size(); ++i) {
      frame_list << (i ? ", " : "") << written[i];
    }
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

    // Stored as ROS parameters so the same file can be loaded by tag_zero_frames
    // or read directly in post-processing.
    std::filesystem::create_directories(std::filesystem::path(output_file_).parent_path());
    std::ofstream out(output_file_);
    if (!out) {
      RCLCPP_ERROR(get_logger(), "Could not open %s for writing.", output_file_.c_str());
      return false;
    }
    out << "# Panel zero calibration: camera_frame -> <tag>_zero for each tag.\n"
        << "# rotation is a quaternion [x, y, z, w]; translation in metres.\n"
        << "/**:\n"
        << "  ros__parameters:\n"
        << "    camera_frame: " << camera_frame_ << "\n"
        << "    calibrated_at: '" << stamp << "'\n"
        << "    duration: " << std::fixed << std::setprecision(1) << duration_ << "\n"
        << "    tag_frames: [" << frame_list.str() << "]\n"
        << "    tags:\n"
        << tags_yaml.str();

    RCLCPP_INFO(get_logger(), "Wrote zero poses for %zu tags to %s",
                written.size(), output_file_.c_str());
    RCLCPP_INFO(get_logger(), "Rebuild data_collection to install it for tag_zero_frames.");
    return true;
  }

  void finish()
  {
    done_ = true;
    timer_->cancel();
    rclcpp::shutdown();
  }

  std::string camera_frame_, tag_prefix_, output_file_;
  double duration_, startup_timeout_;
  int min_samples_;

  std::map<std::string, std::vector<Sample>> samples_;
  std::chrono::steady_clock::time_point start_time_, first_sample_time_;
  bool recording_ = false;
  bool done_ = false;
  bool succeeded_ = false;

  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PanelCalibration>();
  rclcpp::spin(node);
  const bool ok = node->succeeded();
  rclcpp::shutdown();
  return ok ? 0 : 1;
}
