#!/usr/bin/env python3
"""
Turn ping360_sonar /scan_echo beams into a PointCloud2 that RViz can show.

Each echo sample becomes a point at (distance, beam angle) in the sonar frame,
with its 0-255 echo strength in the "intensity" field. The node keeps the most
recent beam at every angle, so a full sweep builds up the familiar sonar
picture and old data is replaced as the head goes around again.

Run (with your workspace sourced so ping360_sonar_msgs is known):
    python3 sonar_echo_to_cloud.py
    python3 sonar_echo_to_cloud.py --ros-args -p threshold:=60 -p stride:=1

Parameters:
    input_topic   (/scan_echo)     SonarEcho topic to read
    output_topic  (/sonar_cloud)   PointCloud2 topic to publish
    threshold     (80)             drop samples weaker than this (0-255)
    stride        (2)              keep every Nth sample along a beam
    min_range     (0.3)            m; hides the ringing right at the sonar
    publish_rate  (5.0)            Hz
    use_msg_stamp (false)          true = stamp clouds with the beam time
                                   (needs matching TF time); false = now()

In RViz: Fixed Frame = sonar, Add > PointCloud2, topic /sonar_cloud,
Color Transformer = Intensity, Style = Points, Size (Pixels) = 2.
"""

import math

import numpy as np


def beam_points(angle, range_m, intensities, threshold, stride, min_range):
    """Return an (N, 4) float32 array of x, y, z, intensity for one beam."""
    data = np.asarray(intensities, dtype=np.float32)
    n = data.size
    if n == 0 or range_m <= 0:
        return np.empty((0, 4), dtype=np.float32)
    idx = np.arange(0, n, max(1, int(stride)))
    dist = (idx + 1) * (float(range_m) / n)
    val = data[idx]
    keep = (val >= threshold) & (dist >= min_range)
    dist, val = dist[keep], val[keep]
    out = np.empty((dist.size, 4), dtype=np.float32)
    out[:, 0] = dist * math.cos(angle)
    out[:, 1] = dist * math.sin(angle)
    out[:, 2] = 0.0
    out[:, 3] = val
    return out


def angle_key(angle):
    # Ping360 steps in gradians (400 per turn); key each beam by its gradian
    return int(round(angle * 200.0 / math.pi)) % 400


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    from ping360_sonar_msgs.msg import SonarEcho

    class EchoToCloud(Node):
        def __init__(self):
            super().__init__("sonar_echo_to_cloud")
            p = self.declare_parameter
            self.in_topic = p("input_topic", "/scan_echo").value
            self.out_topic = p("output_topic", "/sonar_cloud").value
            self.threshold = float(p("threshold", 80).value)
            self.stride = int(p("stride", 2).value)
            self.min_range = float(p("min_range", 0.3).value)
            rate = float(p("publish_rate", 5.0).value)
            self.use_msg_stamp = bool(p("use_msg_stamp", False).value)

            self.beams = {}          # gradian -> (N,4) points
            self.frame_id = "sonar"
            self.last_stamp = None
            self.last_range = None
            self.dirty = False

            self.pub = self.create_publisher(PointCloud2, self.out_topic, 1)
            self.create_subscription(SonarEcho, self.in_topic, self.on_echo, 50)
            self.create_timer(1.0 / rate, self.publish)
            self.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            self.get_logger().info(
                f"{self.in_topic} -> {self.out_topic} (threshold {self.threshold:g}, "
                f"stride {self.stride}, min_range {self.min_range:g} m)")

        def on_echo(self, msg):
            if self.last_range is not None and msg.range != self.last_range:
                self.beams.clear()   # range changed: old beams no longer match
                self.get_logger().info(f"range changed to {msg.range} m, clearing")
            self.last_range = msg.range
            self.frame_id = msg.header.frame_id or "sonar"
            self.last_stamp = msg.header.stamp
            self.beams[angle_key(msg.angle)] = beam_points(
                msg.angle, msg.range, msg.intensities,
                self.threshold, self.stride, self.min_range)
            self.dirty = True

        def publish(self):
            if not self.dirty:
                return
            self.dirty = False
            pts = (np.concatenate(list(self.beams.values()))
                   if self.beams else np.empty((0, 4), np.float32))
            header = Header()
            header.frame_id = self.frame_id
            header.stamp = (self.last_stamp if self.use_msg_stamp and self.last_stamp
                            else self.get_clock().now().to_msg())
            cloud = PointCloud2()
            cloud.header = header
            cloud.height = 1
            cloud.width = int(pts.shape[0])
            cloud.fields = self.fields
            cloud.is_bigendian = False
            cloud.point_step = 16
            cloud.row_step = 16 * cloud.width
            cloud.is_dense = True
            cloud.data = pts.astype(np.float32).tobytes()
            self.pub.publish(cloud)

    rclpy.init()
    node = EchoToCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
