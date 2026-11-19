# apriltag_bringup

Two interchangeable AprilTag pipelines. Both publish the same outputs, so
`panel_calibration`, `tag_zero_frames` and `sealbot_viz` work with either:

| output | contents |
|---|---|
| `/tf` | `camera_optical_frame -> tag_N` |
| `/detections` | `apriltag_msgs/AprilTagDetectionArray` |

| detector | launch file | notes |
|---|---|---|
| `cpu` (default) | `apriltag.launch.xml` | `apriltag_ros`, configured by `config/tags.yaml` |
| `isaac` (experimental) | `cam_isaac_tag.launch.xml use_apriltag:=true` | camera + rectify + GPU AprilTag + `isaac_tag_bridge` in one container. Isaac only finds the tags; the bridge refines the corners on `/camera/image_rect_mono` and solves the pose with PnP like apriltag_ros, then publishes the outputs above |

Isaac (cuAprilTags) misses and mis-localises the small, tilted panel tags, so it is
kept for experiments only. Pick it with `detector:=cpu|isaac` on `data_collection/panel_calibration.launch.xml`
and `data_collection/data_collection.launch.xml`. **Use the same detector for
both**: the zero poses are only valid for the detector that recorded them, and
`tag_zero_frames` warns at startup if they differ.

```bash
ros2 launch data_collection panel_calibration.launch.xml                  # CPU (default)
ros2 launch data_collection panel_calibration.launch.xml detector:=isaac  # experimental
colcon build --packages-select data_collection                            # install the new zero file
ros2 launch data_collection data_collection.launch.xml use_apriltag:=true [detector:=isaac]   # detection is off by default
```

## Checking Isaac's tag axes against apriltag_ros (one-off)

`tag_N_zero -> tag_N` is expressed in the tag's own frame, so its axes only
mean the same thing as with the CPU method if both detectors define the tag
frame the same way. To check, with the panel **not moving**:

```bash
ros2 launch data_collection panel_calibration.launch.xml detector:=cpu   output_file:=/tmp/zero_cpu.yaml
ros2 launch data_collection panel_calibration.launch.xml detector:=isaac output_file:=/tmp/zero_isaac.yaml
ros2 run apriltag_bringup compare_calibrations.py /tmp/zero_cpu.yaml /tmp/zero_isaac.yaml
```

If it reports a consistent fixed rotation, Isaac's corner order differs from
apriltag's. Set in `config/isaac_tags.yaml`, rebuild, and compare again:

- 90/180/270° about tag Z: `corner_shift: 1/2/3`
- Z flipped (180° about X or Y): `corner_reverse: true` (maybe plus a shift)
- small leftover rotation: `tag_rotation_correction`

Then re-run the Isaac calibration.

Keep `size` equal in `config/tags.yaml` and `config/isaac_tags.yaml`.
