# Ncknight_Sealbot_System_2026

ROS 2 system and data tools for the Sealbot, running across a PC, Jetson, and BlueROV.

## Project structure

- `ros_ws/src/` - ROS 2 Humble workspace. `onboard/` contains robot drivers and launch packages; `ground_station/` contains visualization tools.
- `docker/ground_station/` - Docker setup for the ROS 2 ground station.
- `parsers_rosbag/bagsync/` - Opens ROS 2 bags, synchronizes topics to camera frames, and exports CSV files.
- `ML_Models/RNN_10_26/` - Tools to split exported CSVs into windows and train an RNN to estimate IMU data and depth.
- `bags_out/` - Output location for tagged bag data.
- `CalibrationPhotos/` - Camera calibration files and photos.

## Common commands

### Build the ROS workspace

```bash
cd ros_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

### Build and start the ground-station container

From the repository root:

```bash
docker compose -f docker/ground_station/docker-compose.yml up --build
```

### Export a bag to CSV

See [the bagsync guide](parsers_rosbag/bagsync/README.md) for installation and viewer details. In short, install its requirements, then run:

```bash
cd parsers_rosbag/bagsync
python3 bag_viewer.py /path/to/bag
```

### Prepare and train the RNN

Split an exported CSV into windows:

```bash
python3 ML_Models/RNN_10_26/make_windows.py /path/to/export.csv
```

Train using separate training, validation, and test folders:

```bash
python3 ML_Models/RNN_10_26/train_rnn.py --train_dir data/train --val_dir data/val --test_dir data/test
```
