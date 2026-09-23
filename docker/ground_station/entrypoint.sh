#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
# Workspace overlay may not exist yet on the very first build layer
source /ws/install/setup.bash 2>/dev/null || true

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Override at `docker compose run` time with -e ROS_DOMAIN_ID=<n> if sealbot
# uses a non-default domain. Must match every machine in the system.
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

exec "$@"
