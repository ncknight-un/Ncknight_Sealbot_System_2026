# syntax=docker/dockerfile:1
# Base image already includes ros-humble-desktop (saves a large download/install)
FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Keep downloaded .deb files between builds (used with the cache mounts below)
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# All apt packages in one layer.
# MAVROS comes as a prebuilt binary instead of being compiled from source.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
      python3-pip \
      ros-humble-rmw-fastrtps-cpp \
      ros-humble-rmw-cyclonedds-cpp \
      ros-humble-rosbag2-storage-mcap \
      ros-humble-mavros \
      ros-humble-mavros-extras \
      ros-humble-apriltag-ros \
      ros-humble-apriltag-msgs \
      ros-humble-camera-info-manager \
      ros-humble-image-proc \
      mrcal \
      libmrcal-dev \
      python3-mrcal \
      mrgingham

# GeographicLib datasets required by MAVROS
RUN /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

# Ping360 sonar driver needs the Blue Robotics Python library
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install bluerobotics-ping

WORKDIR /ros_ws

# Automatic sourcing and workspace shortcut
RUN cat >> /root/.bashrc <<'EOF'
source /opt/ros/humble/setup.bash
alias humble-source='source /ros_ws/install/setup.bash'
EOF